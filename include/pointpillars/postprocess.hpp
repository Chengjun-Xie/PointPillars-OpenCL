#pragma once

#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "pointpillars/nms.hpp"
#include "pointpillars/pointpillars_util.hpp"

// for opencl
#include "cl_utils.hpp"

namespace pointpillars {

struct nms_pre_sort {
  float score;
  int index;
};

/**
 * PointPillar's PostProcessing
 *
 * Use the output of the RegionProposalNetwork and the
 * anchors generated by the AnchorGrid to decode the
 * object position, dimension and class, filter out
 * redundant/clutter objects using NMS and sort them
 * according to likelihood. Finally convert into
 * object representation.
 */
class PostProcess {
 private:
  const float float_min_;
  const float float_max_;
  const size_t num_anchor_x_inds_;
  const size_t num_anchor_y_inds_;
  const size_t num_anchor_r_inds_;
  const int num_cls_;
  const float* score_threshold_;
  const size_t num_threads_;
  const int num_box_corners_;
  const int num_output_box_feature_;

  std::unique_ptr<NMS> nms_ptr_;

  // for opencl
  std::string opencl_kernel_path_;
  cl::Event eventCL;
  cl_int errCL;
  int idx;

  std::shared_ptr<cl::Context> context_;
  std::shared_ptr<cl::CommandQueue> command_queue_;
  std::shared_ptr<cl::Device> device_;

  std::vector<std::string> program_names_{"postprocess.cl"};

  std::map<std::string, std::vector<std::string>> program_2_kernel_names_{
      {"postprocess.cl", {"FilterKernel", "SortBoxesByIndexKernel"}}};

  std::map<std::string, std::string> name_2_source_;
  std::map<std::string, cl::Program> name_2_program_;
  std::map<std::string, std::vector<unsigned char>> name_2_binary_;
  std::map<std::string, cl::Program> name_2_binary_program_;
  std::map<std::string, cl::Kernel> name_2_kernel_;

  std::string build_option_str_{
      "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math"};

  std::shared_ptr<OCLTOOL> ocl_tool_ = nullptr;

  int threads_for_opencl_ = 64;

  cl::Buffer multicls_score_threshold_;
  std::vector<float> host_filtered_score_;

  // for opencl
  bool OpenCLInit();
  void MakeOCLKernel();

 public:
  /**
   * @brief Constructor
   * @param[in] float_min The lowest float value
   * @param[in] float_max The maximum float value
   * @param[in] num_anchor_x_inds Number of x-indexes for anchors
   * @param[in] num_anchor_y_inds Number of y-indexes for anchors
   * @param[in] num_anchor_r_inds Number of rotation-indexes for anchors
   * @param[in] score_threshold Score threshold for filtering output
   * @param[in] num_threads Number of threads when launching kernel
   * @param[in] nms_overlap_threshold IOU threshold for NMS
   * @param[in] num_box_corners Number of box's corner
   * @param[in] num_output_box_feature Number of output box's feature
   */
  PostProcess(const float float_min, const float float_max,
              const size_t num_anchor_x_inds, const size_t num_anchor_y_inds,
              const size_t num_anchor_r_inds, const size_t num_cls,
              const float* score_threshold, const size_t num_threads,
              const float nms_overlap_threshold, const size_t num_box_corners,
              const size_t num_output_box_feature, const std::string&,
              const std::shared_ptr<cl::Context>&,
              const std::shared_ptr<cl::CommandQueue>&,
              const std::shared_ptr<cl::Device>&);
  /**
   * @brief Postprocessing for the network output
   * @param[in] rpn_box_output Box predictions from the network output
   * @param[in] rpn_cls_output Class predictions from the network output
   * @param[in] rpn_dir_output Direction predictions from the network output
   * @param[in] dev_anchor_mask Anchor mask for filtering the network output
   * @param[in] dev_anchors_px X-coordinate values for corresponding anchors
   * @param[in] dev_anchors_py Y-coordinate values for corresponding anchors
   * @param[in] dev_anchors_pz Z-coordinate values for corresponding anchors
   * @param[in] dev_anchors_dx X-dimension values for corresponding anchors
   * @param[in] dev_anchors_dy Y-dimension values for corresponding anchors
   * @param[in] dev_anchors_dz Z-dimension values for corresponding anchors
   * @param[in] dev_anchors_ro Rotation values for corresponding anchors
   * @param[in] dev_filtered_box Filtered box predictions
   * @param[in] dev_filtered_score Filtered score predictions
   * @param[in] dev_filtered_dir Filtered direction predictions
   * @param[in] dev_box_for_nms Decoded boxes in min_x min_y max_x max_y
   * represenation from pose and dimension
   * @param[in] dev_filter_count The number of filtered output
   * @param[out] out_detection Output bounding boxes
   * @details dev_* represents device memory allocated variables
   */
  void DoPostProcess(
      const cl::Buffer& rpn_box_output, const cl::Buffer& rpn_cls_output,
      const cl::Buffer& rpn_dir_output, const cl::Buffer& dev_anchor_mask,
      const cl::Buffer& dev_anchors_px, const cl::Buffer& dev_anchors_py,
      const cl::Buffer& dev_anchors_pz, const cl::Buffer& dev_anchors_dx,
      const cl::Buffer& dev_anchors_dy, const cl::Buffer& dev_anchors_dz,
      const cl::Buffer& dev_anchors_ro, cl
      : Buffer& dev_multiclass_score, cl::Buffer& dev_filtered_box,
        cl::Buffer& dev_filtered_score, cl::Buffer& dev_filtered_dir,
        cl::Buffer& dev_filtered_class_id, cl::Buffer& dev_box_for_nms,
        cl::Buffer& dev_filter_count,
        std::vector<ObjectDetection>& out_detection);
};
}  // namespace pointpillars
