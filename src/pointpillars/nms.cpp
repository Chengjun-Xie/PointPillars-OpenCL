#include "pointpillars/nms.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

namespace pointpillars {

#ifdef CPUNMS
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
#endif

NMS::NMS(const int num_threads, const int num_box_corners,
         const float nms_overlap_threshold,
         const std::string &opencl_kernel_path,
         const std::shared_ptr<cl::Context> &context,
         const std::shared_ptr<cl::CommandQueue> &command_queue,
         const std::shared_ptr<cl::Device> &device)
    : num_threads_(num_threads),
      num_box_corners_(num_box_corners),
      nms_overlap_threshold_(nms_overlap_threshold),
      opencl_kernel_path_{opencl_kernel_path},
      context_(context),
      command_queue_(command_queue),
      device_(device) {
  OpenCLInit();
  MakeOCLKernel();
}

#ifdef CPUNMS
void NMS::DoNMS(size_t host_filter_count, float *dev_sorted_box_for_nms,
                int *out_keep_inds, size_t &out_num_to_keep) {
#else
void NMS::DoNMS(size_t host_filter_count,
                const cl::Buffer& dev_sorted_box_for_nms, int* out_keep_inds,
                size_t& out_num_to_keep) {
#endif
  // Currently the parallel implementation of NMS only works on the GPU
  // Therefore, in case of a CPU or Host device, we use the sequential
  // implementation
#ifdef CPUNMS
  std::cout << "use SequentialNMS" << std::endl;
  SequentialNMS(host_filter_count, dev_sorted_box_for_nms, out_keep_inds,
                out_num_to_keep);
#else
  ParallelNMSOpenCL(host_filter_count, dev_sorted_box_for_nms, out_keep_inds,
                    out_num_to_keep);
#endif
}

#ifdef CPUNMS

void NMS::SequentialNMS(const size_t host_filter_count,
                        float *dev_sorted_box_for_nms, int *out_keep_inds,
                        size_t &out_num_to_keep) {
  std::vector<int>
      keep_inds_vec;  // vector holding the object indexes that should be kept
  keep_inds_vec.resize(
      host_filter_count);  // resize vector to maximum possible length

  // fill vector with default indexes 0, 1, 2, ...
  std::iota(keep_inds_vec.begin(), keep_inds_vec.end(), 0);

  // Convert vector to a C++ set
  std::set<int> keep_inds(keep_inds_vec.begin(), keep_inds_vec.end());

  // Filtering overlapping boxes
  for (size_t i = 0; i < host_filter_count; ++i) {
    for (size_t j = i + 1; j < host_filter_count; ++j) {
      auto iou_value = DevIoU(dev_sorted_box_for_nms + i * num_box_corners_,
                              dev_sorted_box_for_nms + j * num_box_corners_);
      if (iou_value > nms_overlap_threshold_) {
        // if IoU value to too high, remove the index from the set
        keep_inds.erase(j);
      }
    }
  }

  // fill output data, with the kept indexes
  out_num_to_keep = keep_inds.size();
  int keep_counter = 0;
  for (auto ind : keep_inds) {
    out_keep_inds[keep_counter] = ind;
    keep_counter++;
  }
}

#endif

void NMS::ParallelNMSOpenCL(const int host_filter_count,
                            const cl::Buffer &dev_sorted_box_for_nms,
                            int *out_keep_inds, size_t &out_num_to_keep) {
  const unsigned long col_blocks =
      DIVUP(host_filter_count, threads_for_opencl_);
  std::vector<unsigned long long> host_mask(host_filter_count * col_blocks,
                                            0LL);

  auto dev_mask =
      cl::Buffer(*context_, CL_MEM_READ_WRITE,
                 host_filter_count * col_blocks * sizeof(unsigned long long),
                 NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "NMS::ParallelNMSOpenCL dev_mask buffer malloc fail: " << errCL
              << std::endl;
  }

  errCL = command_queue_->enqueueWriteBuffer(
      dev_mask, true, 0,
      host_filter_count * col_blocks * sizeof(unsigned long long),
      static_cast<void *>(host_mask.data()), NULL, NULL);
  if (CL_SUCCESS != errCL) {
    std::cout << "NMS::ParallelNMSOpenCL enqueueWriteBuffer dev_mask fail: "
              << errCL << std::endl;
  }

  std::string kernel_name{"nms_gpu_kernel"};
  auto kernel = name_2_kernel_[kernel_name];
  idx = 0;
  kernel.setArg(idx++, host_filter_count);
  kernel.setArg(idx++, nms_overlap_threshold_);
  kernel.setArg(idx++, dev_sorted_box_for_nms);
  kernel.setArg(idx++, dev_mask);
  kernel.setArg(idx++, num_box_corners_);

  auto global_ndrange =
      cl::NDRange(col_blocks * threads_for_opencl_, col_blocks);
  auto local_ndrange = cl::NDRange(threads_for_opencl_);
  errCL = command_queue_->enqueueNDRangeKernel(
      kernel, cl::NullRange, global_ndrange, local_ndrange, NULL, &eventCL);
  if (CL_SUCCESS != errCL) {
    std::cout <<  "NMS::ParallelNMSOpenCL enqueueNDRangeKernel nms_rotated_gpu_kernel fail: "
        << errCL << std::endl;
  }

  eventCL.wait();
  // postprocess for nms output

  errCL = command_queue_->enqueueReadBuffer(
      dev_mask, true, 0,
      host_filter_count * col_blocks * sizeof(unsigned long long),
      static_cast<void *>(host_mask.data()), NULL, NULL);
  if (CL_SUCCESS != errCL) {
    std::cout << "NMS::ParallelNMSOpenCL enqueueReadBuffer dev_mask fail: " << errCL << std::endl;
  }

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  for (size_t i = 0; i < host_filter_count; i++) {
    int nblock = i / num_threads_;
    int inblock = i % num_threads_;

    if (!(remv[nblock] & (1ULL << inblock))) {
      out_keep_inds[out_num_to_keep++] = i;
      unsigned long long *p = &host_mask[0] + i * col_blocks;
      for (size_t j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
}

bool NMS::OpenCLInit() {
  ocl_tool_.reset(new OCLTOOL(opencl_kernel_path_, program_names_,
                              program_2_kernel_names_, context_, device_,
                              build_option_str_));

  return true;
}

void NMS::MakeOCLKernel() {
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
