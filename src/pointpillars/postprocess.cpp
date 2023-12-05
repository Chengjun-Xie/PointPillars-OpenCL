#include "pointpillars/postprocess.hpp"

#include <algorithm>
#include <cmath>

namespace pointpillars {
void SortBoxesByIndexKernelCPU(
    const float* filtered_box, const int* filtered_dir,
    const int* filtered_class_id, const float* dev_multiclass_score,
    const float* box_for_nms, const std::vector<int>& indexes,
    const int filter_count, float* sorted_filtered_boxes,
    int* sorted_filtered_dir, int* sorted_filtered_class_id,
    float* dev_sorted_multiclass_score, float* sorted_box_for_nms,
    const size_t num_box_corners, const size_t num_output_box_feature,
    const size_t num_cls) {
#pragma unroll(8)
  for (int tid = 0; tid < filter_count; tid++) {
    int sort_index = indexes[tid];

    sorted_filtered_boxes[tid * num_output_box_feature + 0] =
        filtered_box[sort_index * num_output_box_feature + 0];
    sorted_filtered_boxes[tid * num_output_box_feature + 1] =
        filtered_box[sort_index * num_output_box_feature + 1];
    sorted_filtered_boxes[tid * num_output_box_feature + 2] =
        filtered_box[sort_index * num_output_box_feature + 2];
    sorted_filtered_boxes[tid * num_output_box_feature + 3] =
        filtered_box[sort_index * num_output_box_feature + 3];
    sorted_filtered_boxes[tid * num_output_box_feature + 4] =
        filtered_box[sort_index * num_output_box_feature + 4];
    sorted_filtered_boxes[tid * num_output_box_feature + 5] =
        filtered_box[sort_index * num_output_box_feature + 5];
    sorted_filtered_boxes[tid * num_output_box_feature + 6] =
        filtered_box[sort_index * num_output_box_feature + 6];

#pragma unroll
    for (size_t i = 0; i < num_cls; ++i) {
      dev_sorted_multiclass_score[tid * num_cls + i] =
          dev_multiclass_score[sort_index * num_cls + i];
    }

    sorted_filtered_dir[tid] = filtered_dir[sort_index];
    sorted_filtered_class_id[tid] = filtered_class_id[sort_index];

    sorted_box_for_nms[tid * num_box_corners + 0] =
        box_for_nms[sort_index * num_box_corners + 0];
    sorted_box_for_nms[tid * num_box_corners + 1] =
        box_for_nms[sort_index * num_box_corners + 1];
    sorted_box_for_nms[tid * num_box_corners + 2] =
        box_for_nms[sort_index * num_box_corners + 2];
    sorted_box_for_nms[tid * num_box_corners + 3] =
        box_for_nms[sort_index * num_box_corners + 3];
  }
}

PostProcess::PostProcess(const float float_min, const float float_max,
                         const size_t num_anchor_x_inds,
                         const size_t num_anchor_y_inds,
                         const size_t num_anchor_r_inds, const size_t num_cls,
                         const float* score_threshold, const size_t num_threads,
                         const float nms_overlap_threshold,
                         const size_t num_box_corners,
                         const size_t num_output_box_feature,
                         const std::string& opencl_kernel_path,
                         const std::shared_ptr<cl::Context>& context,
                         const std::shared_ptr<cl::CommandQueue>& command_queue,
                         const std::shared_ptr<cl::Device>& device)
    : float_min_(float_min),
      float_max_(float_max),
      num_anchor_x_inds_(num_anchor_x_inds),
      num_anchor_y_inds_(num_anchor_y_inds),
      num_anchor_r_inds_(num_anchor_r_inds),
      num_cls_(num_cls),
      score_threshold_(score_threshold),
      num_threads_(num_threads),
      num_box_corners_(num_box_corners),
      num_output_box_feature_(num_output_box_feature),
      opencl_kernel_path_{opencl_kernel_path},
      context_(context),
      command_queue_(command_queue),
      device_(device) {
  nms_ptr_ = std::make_unique<NMS>(num_threads, num_box_corners,
                                   nms_overlap_threshold, opencl_kernel_path_,
                                   context_, command_queue_, device_);

  OpenCLInit();
  MakeOCLKernel();
  multicls_score_threshold_ = cl::Buffer(
      *context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      num_cls_ * sizeof(float),
      static_cast<void*>(const_cast<float*>(score_threshold_)), &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "PostProcess::PostProcess buffer multicls_score_threshold_ "
                 "malloc fail: "
              << errCL << std::endl;
  }

  host_filtered_score_.resize(num_anchor_x_inds_ * num_anchor_y_inds_);
}

void PostProcess::DoPostProcess(
    const cl::Buffer& rpn_box_output, const cl::Buffer& rpn_cls_output,
    const cl::Buffer& rpn_dir_output, const cl::Buffer& dev_anchor_mask,
    const cl::Buffer& dev_anchors_px, const cl::Buffer& dev_anchors_py,
    const cl::Buffer& dev_anchors_pz, const cl::Buffer& dev_anchors_dx,
    const cl::Buffer& dev_anchors_dy, const cl::Buffer& dev_anchors_dz,
    const cl::Buffer& dev_anchors_ro, cl::Buffer& dev_multiclass_score,
    cl::Buffer& dev_filtered_box, cl::Buffer& dev_filtered_score,
    cl::Buffer& dev_filtered_dir, cl::Buffer& dev_filtered_class_id,
    cl::Buffer& dev_box_for_nms, cl::Buffer& dev_filter_count,
    std::vector<ObjectDetection>& detections) {
  // filter objects by applying a class confidence threshold
  // Calculate number of boxes in the feature map
  // Decode the output of the RegionProposalNetwork and store all the boxes with
  // score above the threshold
  const unsigned int length =
      num_anchor_x_inds_ * num_cls_ * num_anchor_r_inds_ * num_anchor_y_inds_;

  {
    std::string kernel_name{"FilterKernel"};
    auto kernel = name_2_kernel_[kernel_name];
    idx = 0;
    kernel.setArg(idx++, rpn_box_output);
    kernel.setArg(idx++, rpn_cls_output);
    kernel.setArg(idx++, rpn_dir_output);
    kernel.setArg(idx++, dev_anchor_mask);
    kernel.setArg(idx++, dev_anchors_px);
    kernel.setArg(idx++, dev_anchors_py);
    kernel.setArg(idx++, dev_anchors_pz);
    kernel.setArg(idx++, dev_anchors_dx);
    kernel.setArg(idx++, dev_anchors_dy);
    kernel.setArg(idx++, dev_anchors_dz);
    kernel.setArg(idx++, dev_anchors_ro);
    kernel.setArg(idx++, dev_filtered_box);
    kernel.setArg(idx++, dev_filtered_score);
    kernel.setArg(idx++, dev_multiclass_score);
    kernel.setArg(idx++, dev_filtered_dir);
    kernel.setArg(idx++, dev_filtered_class_id);
    kernel.setArg(idx++, dev_box_for_nms);
    kernel.setArg(idx++, dev_filter_count);
    kernel.setArg(idx++, float_min_);
    kernel.setArg(idx++, float_max_);
    kernel.setArg(idx++, multicls_score_threshold_);
    errCL = kernel.setArg(idx++, num_box_corners_);
    if (CL_SUCCESS != errCL) {
      std::cout << "PostProcess::DoPostProcess FilterKernel setArg "
                   "num_box_corners_ fail: "
                << errCL << std::endl;
    }
    kernel.setArg(idx++, num_output_box_feature_);
    if (CL_SUCCESS != errCL) {
      std::cout << "PostProcess::DoPostProcess FilterKernel setArg "
                   "num_output_box_feature_ fail: "
                << errCL << std::endl;
    }
    kernel.setArg(idx++, num_cls_);
    if (CL_SUCCESS != errCL) {
      std::cout
          << "PostProcess::DoPostProcess FilterKernel setArg num_cls_ fail: "
          << errCL << std::endl;
    }
    kernel.setArg(idx++, length);

    int nums_opencl = DIVUP(length, threads_for_opencl_) * threads_for_opencl_;
    auto global_ndrange = cl::NDRange(nums_opencl);
    auto local_ndrange = cl::NDRange(threads_for_opencl_);
    errCL = command_queue_->enqueueNDRangeKernel(
        kernel, cl::NullRange, global_ndrange, local_ndrange, NULL, &eventCL);
    if (CL_SUCCESS != errCL) {
      std::cout << "PostProcess::DoPostProcess enqueueNDRangeKernel "
                   "FilterKernel fail: "
                << errCL << std::endl;
    }

    eventCL.wait();
  }

  // remove no longer required memory

  int host_filter_count[1];
  errCL = command_queue_->enqueueReadBuffer(
      dev_filter_count, true, 0, sizeof(int),
      static_cast<void*>(host_filter_count), NULL, NULL);
  if (CL_SUCCESS != errCL) {
    std::cout << "PostProcess::DoPostProcess enqueueReadBuffer "
                 "dev_filter_count fail: "
              << errCL << std::endl;
  }

  if (host_filter_count[0] == 0) {
    return;
  }

  // Create variables to hold the sorted box arrays

  auto dev_sorted_box_for_nms = cl::Buffer(
      *context_, CL_MEM_READ_WRITE,
      num_box_corners_ * host_filter_count[0] * sizeof(float), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "PostProcess::DoPostProcess dev_sorted_box_for_nms buffer "
                 "malloc fail: "
              << errCL << std::endl;
  }

  errCL = command_queue_->enqueueReadBuffer(
      dev_filtered_score, true, 0, host_filter_count[0] * sizeof(float),
      static_cast<void*>(host_filtered_score_.data()), NULL, NULL);
  if (CL_SUCCESS != errCL) {
    std::cout << "PostProcess::DoPostProcess enqueueReadBuffer "
                 "dev_filtered_score fail: "
              << errCL << std::endl;
  }

  std::vector<nms_pre_sort> nms_sort;
  nms_sort.resize(host_filter_count[0]);
  for (int i = 0; i < host_filter_count[0]; i++) {
    nms_sort[i].index = i;
    nms_sort[i].score = host_filtered_score_[i];
  }

  // Sort the box indexes according to the boxes score

  std::sort(nms_sort.begin(), nms_sort.end(),
            [](const nms_pre_sort& left, const nms_pre_sort& right) {
              return left.score > right.score;
            });

  std::vector<int> host_indexes;
  std::vector<float> host_filtered_score;
  host_indexes.resize(host_filter_count[0]);
  host_filtered_score.resize(host_filter_count[0]);
  for (int i = 0; i < host_filter_count[0]; i++) {
    host_indexes[i] = nms_sort[i].index;
    host_filtered_score[i] = nms_sort[i].score;
  }

  // Create arrays to hold the detections in host memory
  std::vector<float> host_filtered_box;
  host_filtered_box.resize(host_filter_count[0] * num_output_box_feature_);
  std::vector<float> host_multiclass_score;
  host_multiclass_score.resize(host_filter_count[0] * num_cls_);
  std::vector<int> host_filtered_dir;
  host_filtered_dir.resize(host_filter_count[0]);
  std::vector<int> host_filtered_class_id;
  host_filtered_class_id.resize(host_filter_count[0]);

  auto host_box_for_nms = new float[host_filter_count[0] * num_box_corners_];

  {
    std::vector<float> unsorted_host_filtered_box;
    unsorted_host_filtered_box.resize(host_filter_count[0] *
                                      num_output_box_feature_);
    errCL = command_queue_->enqueueReadBuffer(
        dev_filtered_box, true, 0,
        host_filter_count[0] * num_output_box_feature_ * sizeof(float),
        static_cast<void*>(unsorted_host_filtered_box.data()), NULL, NULL);
    if (CL_SUCCESS != errCL) {
      std::cout << "PostProcess::DoPostProcess enqueueReadBuffer "
                   "dev_filtered_box fail: "
                << errCL << std::endl;
    }

    std::vector<int> unsorted_host_filtered_class_id;
    unsorted_host_filtered_class_id.resize(host_filter_count[0]);
    errCL = command_queue_->enqueueReadBuffer(
        dev_filtered_class_id, true, 0, host_filter_count[0] * sizeof(float),
        static_cast<void*>(unsorted_host_filtered_class_id.data()), NULL, NULL);
    if (CL_SUCCESS != errCL) {
      std::cout << "PostProcess::DoPostProcess enqueueReadBuffer "
                   "dev_filtered_class_id fail: "
                << errCL << std::endl;
    }

    std::vector<int> unsorted_host_filtered_dir;
    unsorted_host_filtered_dir.resize(host_filter_count[0]);
    errCL = command_queue_->enqueueReadBuffer(
        dev_filtered_dir, true, 0, host_filter_count[0] * sizeof(float),
        static_cast<void*>(unsorted_host_filtered_dir.data()), NULL, NULL);
    if (CL_SUCCESS != errCL) {
      std::cout << "PostProcess::DoPostProcess enqueueReadBuffer "
                   "dev_filtered_dir fail: "
                << errCL << std::endl;
    }

    std::vector<float> unsorted_host_multiclass_score;
    unsorted_host_multiclass_score.resize(host_filter_count[0] * num_cls_);
    errCL = command_queue_->enqueueReadBuffer(
        dev_multiclass_score, true, 0,
        host_filter_count[0] * num_cls_ * sizeof(float),
        static_cast<void*>(unsorted_host_multiclass_score.data()), NULL, NULL);
    if (CL_SUCCESS != errCL) {
      std::cout << "PostProcess::DoPostProcess enqueueReadBuffer "
                   "dev_multiclass_score fail: "
                << errCL << std::endl;
    }

    std::vector<float> unsorted_host_box_for_nms;
    unsorted_host_box_for_nms.resize(host_filter_count[0] * num_box_corners_);
    errCL = command_queue_->enqueueReadBuffer(
        dev_box_for_nms, true, 0,
        host_filter_count[0] * num_box_corners_ * sizeof(float),
        static_cast<void*>(unsorted_host_box_for_nms.data()), NULL, NULL);
    if (CL_SUCCESS != errCL) {
      std::cout << "PostProcess::DoPostProcess enqueueReadBuffer "
                   "dev_box_for_nms fail: "
                << errCL << std::endl;
    }

    SortBoxesByIndexKernelCPU(
        unsorted_host_filtered_box.data(), unsorted_host_filtered_dir.data(),
        unsorted_host_filtered_class_id.data(),
        unsorted_host_multiclass_score.data(), unsorted_host_box_for_nms.data(),
        host_indexes, host_filter_count[0], host_filtered_box.data(),
        host_filtered_dir.data(), host_filtered_class_id.data(),
        host_multiclass_score.data(), host_box_for_nms, num_box_corners_,
        num_output_box_feature_, num_cls_);
  }

  errCL = command_queue_->enqueueWriteBuffer(
      dev_sorted_box_for_nms, true, 0,
      host_filter_count[0] * num_box_corners_ * sizeof(float),
      static_cast<void*>(host_box_for_nms), NULL, NULL);
  if (CL_SUCCESS != errCL) {
    std::cout << "PostProcess::DoPostProcess enqueueWriteBuffer "
                 "dev_sorted_box_for_nms fail: "
              << errCL << std::endl;
  }

  // Apply NMS to the sorted boxes
  int keep_inds[host_filter_count[0]];
  size_t out_num_objects = 0;
#ifdef CPUNMS
  nms_ptr_->DoNMS(host_filter_count[0], host_box_for_nms, keep_inds,
                  out_num_objects);
#else
  nms_ptr_->DoNMS(host_filter_count[0], dev_sorted_box_for_nms, keep_inds,
                  out_num_objects);
#endif

  delete[] host_box_for_nms;

  // Convert the NMS filtered boxes defined by keep_inds to an array of
  // ObjectDetection
  for (size_t i = 0; i < out_num_objects; i++) {
    ObjectDetection detection;
    detection.x = host_filtered_box[keep_inds[i] * num_output_box_feature_ + 0];
    detection.y = host_filtered_box[keep_inds[i] * num_output_box_feature_ + 1];
    detection.z = host_filtered_box[keep_inds[i] * num_output_box_feature_ + 2];
    detection.length =
        host_filtered_box[keep_inds[i] * num_output_box_feature_ + 3];
    detection.width =
        host_filtered_box[keep_inds[i] * num_output_box_feature_ + 4];
    detection.height =
        host_filtered_box[keep_inds[i] * num_output_box_feature_ + 5];

    detection.class_id =
        static_cast<float>(host_filtered_class_id[keep_inds[i]]);
    detection.likelihood = host_filtered_score[keep_inds[i]];

    // Apply the direction label found by the direction classifier
    if (host_filtered_dir[keep_inds[i]] == 0) {
      detection.yaw =
          host_filtered_box[keep_inds[i] * num_output_box_feature_ + 6] + M_PI;
    } else {
      detection.yaw =
          host_filtered_box[keep_inds[i] * num_output_box_feature_ + 6];
    }

    for (size_t k = 0; k < num_cls_; k++) {
      detection.class_probabilities.push_back(
          host_multiclass_score[keep_inds[i] * num_cls_ + k]);
    }

    detections.push_back(detection);
  }
}

bool PostProcess::OpenCLInit() {
  ocl_tool_.reset(new OCLTOOL(opencl_kernel_path_, program_names_,
                              program_2_kernel_names_, context_, device_,
                              build_option_str_));

  return true;
}

void PostProcess::MakeOCLKernel() {
#ifdef OCLDEBUG
  ocl_tool_->MakeKernelSource(name_2_source_);
  ocl_tool_->MakeProgramFromSource(name_2_source_, name_2_program_);
  ocl_tool_->BuildProgramAndGetBinary(name_2_program_, name_2_binary_);
  ocl_tool_->SaveProgramBinary(name_2_binary_);
#endif
  ocl_tool_->LoadBinaryAndMakeProgram(name_2_binary_program_);
  ocl_tool_->MakeKernelFromBinary(name_2_binary_program_, name_2_kernel_);
}
}  // namespace pointpillars
