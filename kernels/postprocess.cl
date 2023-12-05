// using MACRO to allocate memory inside kernel
#define NUM_3D_BOX_CORNERS_MACRO 8
#define NUM_2D_BOX_CORNERS_MACRO 4


float my_exp(float x) 
{
    x = 1.0 + x/256;

    x *= x;       
    x *= x;       
    x *= x;       
    x *= x;

    x *= x;       
    x *= x;       
    x *= x;       
    x *= x;
    // x = exp(x);
    return x;
}

float Sigmoid(float x) 
{ 
    return 1.0f / (1.0f + my_exp(-x)); 
}

__kernel void FilterKernel(const __global float* box_preds,
                           const __global float* cls_preds,
                           const __global float* dir_preds,
                           const __global float* anchor_mask,
                           const __global float* dev_anchors_px,
                           const __global float* dev_anchors_py,
                           const __global float* dev_anchors_pz,
                           const __global float* dev_anchors_dx,
                           const __global float* dev_anchors_dy,
                           const __global float* dev_anchors_dz,
                           const __global float* dev_anchors_ro,
                           __global float* filtered_box,
                           __global float* filtered_score,
                           __global float* multiclass_score,
                           __global int* filtered_dir,
                           __global int* dev_filtered_class_id,
                           __global float* box_for_nms,
                           __global int* filter_count,
                           const float float_min,
                           const float float_max,
                           const __global float* score_threshold,
                           const int num_box_corners,
                           const int num_output_box_feature,
                           const int num_cls,
                           const int max_index) 
{
    float class_score_cache[20];  // Asume maximum class size of 20 to avoid runtime allocations
    
    int tid = get_group_id(0) * get_local_size(0) + get_local_id(0);

    if (tid >= max_index) {
        return;
    }

    // Decode the class probabilities using the Sigmoid function
    float score = 0.f;
    int class_id = 0;
    for (size_t i = 0; i < num_cls; i++) {
        class_score_cache[i] = Sigmoid(cls_preds[tid * num_cls + i]);

        if (class_score_cache[i] > score) {
            score = class_score_cache[i];
            class_id = i;
        }
    }

    // if there is data inside the anchor
    if (anchor_mask[tid] == 1 && score > score_threshold[class_id]) {    
        int counter = atomic_add(filter_count, 1);
        float za = dev_anchors_pz[tid] + dev_anchors_dz[tid] / 2;

        // decode RPN output, the formulas are used according to the encoding used in the paper
        float diagonal = sqrt(dev_anchors_dx[tid] * dev_anchors_dx[tid] + dev_anchors_dy[tid] * dev_anchors_dy[tid]);
        float box_px = box_preds[tid * num_output_box_feature + 0] * diagonal + dev_anchors_px[tid];
        float box_py = box_preds[tid * num_output_box_feature + 1] * diagonal + dev_anchors_py[tid];
        float box_pz = box_preds[tid * num_output_box_feature + 2] * dev_anchors_dz[tid] + za;
        float box_dx = my_exp((float)(box_preds[tid * num_output_box_feature + 3])) * dev_anchors_dx[tid];
        float box_dy = my_exp((float)(box_preds[tid * num_output_box_feature + 4])) * dev_anchors_dy[tid];
        float box_dz = my_exp((float)(box_preds[tid * num_output_box_feature + 5])) * dev_anchors_dz[tid];
        float box_ro = box_preds[tid * num_output_box_feature + 6] + dev_anchors_ro[tid];

        box_pz = box_pz - box_dz / 2.f;

        // Store the detection x,y,z,l,w,h,theta coordinates
        filtered_box[counter * num_output_box_feature + 0] = box_px;
        filtered_box[counter * num_output_box_feature + 1] = box_py;
        filtered_box[counter * num_output_box_feature + 2] = box_pz;
        filtered_box[counter * num_output_box_feature + 3] = box_dx;
        filtered_box[counter * num_output_box_feature + 4] = box_dy;
        filtered_box[counter * num_output_box_feature + 5] = box_dz;
        filtered_box[counter * num_output_box_feature + 6] = box_ro;
        filtered_score[counter] = score;

        // Copy the class scores
        for (size_t i = 0; i < num_cls; i++) {
            multiclass_score[counter * num_cls + i] = class_score_cache[i];
        }

        // Decode the direction class specified in SecondNet: Sparsely Embedded Convolutional Detection
        int direction_label;
        if (dir_preds[tid * 2 + 0] < dir_preds[tid * 2 + 1]) {
            direction_label = 1;
        } else {
            direction_label = 0;
        }

        filtered_dir[counter] = direction_label;
        dev_filtered_class_id[counter] = class_id;
        // convert normal box(normal boxes: x, y, z, w, l, h, r) to box(xmin, ymin,
        // xmax, ymax) for nms calculation
        // First: dx, dy -> box(x0y0, x0y1, x1y0, x1y1)
        float corners[NUM_3D_BOX_CORNERS_MACRO] = {(float)(-0.5f * box_dx), (float)(-0.5f * box_dy), (float)(-0.5f * box_dx),
                                                   (float)(0.5f * box_dy),  (float)(0.5f * box_dx),  (float)(0.5f * box_dy),
                                                   (float)(0.5f * box_dx),  (float)(-0.5f * box_dy)};

        // Second: Rotate, Offset and convert to point(xmin. ymin, xmax, ymax)
        float rotated_corners[NUM_3D_BOX_CORNERS_MACRO];
        float offset_corners[NUM_3D_BOX_CORNERS_MACRO];
        float sin_yaw = sin(box_ro);
        float cos_yaw = cos(box_ro);

        float xmin = float_max;
        float ymin = float_max;
        float xmax = float_min;
        float ymax = float_min;
        for (size_t i = 0; i < num_box_corners; i++) {
            rotated_corners[i * 2 + 0] = cos_yaw * corners[i * 2 + 0] - sin_yaw * corners[i * 2 + 1];
            rotated_corners[i * 2 + 1] = sin_yaw * corners[i * 2 + 0] + cos_yaw * corners[i * 2 + 1];

            offset_corners[i * 2 + 0] = rotated_corners[i * 2 + 0] + box_px;
            offset_corners[i * 2 + 1] = rotated_corners[i * 2 + 1] + box_py;

            xmin = fmin(xmin, offset_corners[i * 2 + 0]);
            ymin = fmin(ymin, offset_corners[i * 2 + 1]);
            xmax = fmax(xmin, offset_corners[i * 2 + 0]);
            ymax = fmax(ymax, offset_corners[i * 2 + 1]);
        }

        box_for_nms[counter * num_box_corners + 0] = xmin;
        box_for_nms[counter * num_box_corners + 1] = ymin;
        box_for_nms[counter * num_box_corners + 2] = xmax;
        box_for_nms[counter * num_box_corners + 3] = ymax;
    }
}

__kernel void SortBoxesByIndexKernel(const __global float* filtered_box,
                                     const __global int* filtered_dir,
                                     const __global int* filtered_class_id,
                                     const __global float* dev_multiclass_score,
                                     const __global float* box_for_nms,
                                     const __global int* indexes,
                                     const int filter_count,
                                     __global float* sorted_filtered_boxes,
                                     __global int* sorted_filtered_dir,
                                     __global int* sorted_filtered_class_id,
                                     __global float* dev_sorted_multiclass_score,
                                     __global float* sorted_box_for_nms,
                                     const int num_box_corners,
                                     const int num_output_box_feature,
                                     const int num_cls) 
{
    int tid = get_group_id(0) * get_local_size(0) + get_local_id(0);
    
    if (tid < filter_count) {
        int sort_index = indexes[tid];
        sorted_filtered_boxes[tid * num_output_box_feature + 0] = filtered_box[sort_index * num_output_box_feature + 0];
        sorted_filtered_boxes[tid * num_output_box_feature + 1] = filtered_box[sort_index * num_output_box_feature + 1];
        sorted_filtered_boxes[tid * num_output_box_feature + 2] = filtered_box[sort_index * num_output_box_feature + 2];
        sorted_filtered_boxes[tid * num_output_box_feature + 3] = filtered_box[sort_index * num_output_box_feature + 3];
        sorted_filtered_boxes[tid * num_output_box_feature + 4] = filtered_box[sort_index * num_output_box_feature + 4];
        sorted_filtered_boxes[tid * num_output_box_feature + 5] = filtered_box[sort_index * num_output_box_feature + 5];
        sorted_filtered_boxes[tid * num_output_box_feature + 6] = filtered_box[sort_index * num_output_box_feature + 6];

        for (size_t i = 0; i < num_cls; ++i) {
            dev_sorted_multiclass_score[tid * num_cls + i] = dev_multiclass_score[sort_index * num_cls + i];
        }

        sorted_filtered_dir[tid] = filtered_dir[sort_index];
        sorted_filtered_class_id[tid] = filtered_class_id[sort_index];

        sorted_box_for_nms[tid * num_box_corners + 0] = box_for_nms[sort_index * num_box_corners + 0];
        sorted_box_for_nms[tid * num_box_corners + 1] = box_for_nms[sort_index * num_box_corners + 1];
        sorted_box_for_nms[tid * num_box_corners + 2] = box_for_nms[sort_index * num_box_corners + 2];
        sorted_box_for_nms[tid * num_box_corners + 3] = box_for_nms[sort_index * num_box_corners + 3];
    }
}

