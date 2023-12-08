#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cl_utils.hpp"

namespace pointpillars {

/**
 * Non-Maximum-Suppression
 *
 * Non-maximum suppression (NMS) is a way to eliminate points that do not lie in
 * the important edges of detected data. Here NMS is used to filter out
 * overlapping object detections. Therefore, an intersection-over-union (IOU)
 * approach is used to caculate the overlap of two objects. At the end, only the
 * most relevant objects are kept.
 */
class NMS {
 private:
  const int num_threads_;  // Number of threads used to execute the NMS kernel
  const int num_box_corners_;  // Number of corners of a 2D box
  const float
      nms_overlap_threshold_;  // Threshold below which objects are discarded

  // for opencl
  std::string opencl_kernel_path_;
  cl_int errCL;
  cl::Event eventCL;
  int idx;

  std::shared_ptr<cl::Context> context_ = nullptr;
  std::shared_ptr<cl::CommandQueue> command_queue_ = nullptr;
  std::shared_ptr<cl::Device> device_ = nullptr;

  uint64_t gpu_global_memory_cache_size;
  uint32_t gpu_compute_unit;
  uint32_t gpu_max_frequent;

  std::vector<std::string> program_names_{"nms.cl"};

  std::map<std::string, std::vector<std::string>> program_2_kernel_names_{
      {"nms.cl", {"nms_gpu_kernel"}}};

  std::map<std::string, std::string> name_2_source_;
  std::map<std::string, cl::Program> name_2_program_;
  std::map<std::string, std::vector<unsigned char>> name_2_binary_;
  std::map<std::string, cl::Program> name_2_binary_program_;
  std::map<std::string, cl::Kernel> name_2_kernel_;

  std::string build_option_str_{
      "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math"};

  std::shared_ptr<OCLTOOL> ocl_tool_ = nullptr;

  int threads_for_opencl_ = 64;

  // for opencl
  bool OpenCLInit();
  void MakeOCLKernel();

 public:
  /**
   * @brief Constructor
   * @param[in] num_threads Number of threads when launching kernel
   * @param[in] num_box_corners Number of corners for 2D box
   * @param[in] nms_overlap_threshold IOU threshold for NMS
   */
  NMS(const int num_threads, const int num_box_corners,
      const float nms_overlap_threshold, const std::string &,
      const std::shared_ptr<cl::Context> &,
      const std::shared_ptr<cl::CommandQueue> &,
      const std::shared_ptr<cl::Device> &);

  /**
   * @brief Execute Non-Maximum Suppresion for network output
   * @param[in] host_filter_count Number of filtered output
   * @param[in] dev_sorted_box_for_nms Bounding box output sorted by score
   * @param[out] out_keep_inds Indexes of selected bounding box
   * @param[out] out_num_to_keep Number of kept bounding boxes
   */
#ifdef CPUNMS
  void DoNMS(size_t host_filter_count, float *dev_sorted_box_for_nms,
             int *out_keep_inds, size_t &out_num_to_keep);
#else
  void DoNMS(size_t host_filter_count, const cl::Buffer &dev_sorted_box_for_nms,
             int *out_keep_inds, size_t &out_num_to_keep);
#endif

 private:
  /**
   * @brief Parallel Non-Maximum Suppresion for network output
   * @details Parallel NMS and postprocessing for selecting box
   */
  void ParallelNMS(const size_t host_filter_count,
                   float *dev_sorted_box_for_nms, int *out_keep_inds,
                   size_t &out_num_to_keep);

  void ParallelNMSOpenCL(const int host_filter_count,
                         const cl::Buffer &dev_sorted_box_for_nms,
                         int *out_keep_inds, size_t &out_num_to_keep);

#ifdef CPUNMS
  /**
   * @brief Sequential Non-Maximum Suppresion for network output in CPU or Host
   * device
   */
  void SequentialNMS(const size_t host_filter_count,
                     float *dev_sorted_box_for_nms, int *out_keep_inds,
                     size_t &out_num_to_keep);
#endif
};
}  // namespace pointpillars
