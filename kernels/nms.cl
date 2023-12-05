// #include "cl_head.h"
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define threadsPerBlock sizeof(unsigned long) * 8
// const int threadsPerBlock = sizeof(unsigned long) * 8;

// Intersection over Union (IoU) calculation
// a and b are pointers to the input objects
// @return IoU value = Area of overlap / Area of union
// @details: https://en.wikipedia.org/wiki/Jaccard_index
inline float DevIoU(float const *const a, float const *const b) {
  float left = max(a[0], b[0]);
  float right = min(a[2], b[2]);
  float top = max(a[1], b[1]);
  float bottom = min(a[3], b[3]);
  float width = max((float)(right - left + 1), 0.f);
  float height = max((float)(bottom - top + 1), 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__kernel void nms_gpu_kernel(const int n_boxes,
                                     const float nms_overlap_thresh,
                                     const __global float* dev_boxes,
                                     __global unsigned long* dev_mask,
                                     const int num_box_corners) 
{
    const int tid = get_local_id(0);
    const unsigned long row_start = get_group_id(0);
    const unsigned long col_start = get_group_id(1);

    if (row_start > col_start) return;

    const unsigned long block_threads = get_local_size(0);
    const unsigned long row_size = min((unsigned long)(n_boxes - row_start * block_threads), block_threads);
    const unsigned long col_size = min((unsigned long)(n_boxes - col_start * block_threads), block_threads);

    __local float block_boxes[threadsPerBlock * 4];

    if (tid < col_size) {
        block_boxes[tid * num_box_corners + 0] = 
                    dev_boxes[(block_threads * col_start + tid) * num_box_corners + 0];
        block_boxes[tid * num_box_corners + 1] = 
                    dev_boxes[(block_threads * col_start + tid) * num_box_corners + 1];
        block_boxes[tid * num_box_corners + 2] = 
                    dev_boxes[(block_threads * col_start + tid) * num_box_corners + 2];
        block_boxes[tid * num_box_corners + 3] = 
                    dev_boxes[(block_threads * col_start + tid) * num_box_corners + 3];
    }

    if (tid < row_size) {
        const int cur_box_idx = block_threads * row_start + tid;
        const float *cur_box = dev_boxes + cur_box_idx * num_box_corners;
        unsigned long t = 0;
        int start = 0;
        if (row_start == col_start) {
            start = tid + 1;
        }
        for (size_t i = start; i < col_size; i++) {
            float iou = DevIoU(cur_box, block_boxes + i * num_box_corners);

            if (iou > nms_overlap_thresh) {
                t |= ((unsigned long)1) << i;
            }
        }
        const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
        
        dev_mask[cur_box_idx * col_blocks + col_start] = t;
    }
}