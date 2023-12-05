#ifndef __POINTPILLARS_FOR_LIDAR_3D_HPP__
#define __POINTPILLARS_FOR_LIDAR_3D_HPP__

#pragma once

#include <pthread.h>

#include <fstream>
#include <gpu/gpu_context_api_ocl.hpp>
#include <inference_engine.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "pointpillars/anchorgrid.hpp"
#include "pointpillars/pointpillars_config.hpp"
#include "pointpillars/pointpillars_util.hpp"
#include "pointpillars/postprocess.hpp"
#include "pointpillars/preprocess.hpp"
#include "pointpillars/scatter.hpp"
// for opencl
#include "cl_utils.hpp"

namespace pointpillars {
class PointPillars {
 protected:
  PointPillarsConfig config_;
  const float *score_threshold_;
  const float nms_overlap_threshold_;
  const std::string pfe_model_file_;
  const std::string rpn_model_file_;
  const int max_num_pillars_;
  const int max_num_points_per_pillar_;
  const int pfe_output_size_;
  const int grid_x_size_;
  const int grid_y_size_;
  const int grid_z_size_;
  const int rpn_input_size_;
  const int num_cls_;
  const int num_anchor_x_inds_;
  const int num_anchor_y_inds_;
  const int num_anchor_r_inds_;
  const int num_anchor_;
  const int rpn_box_output_size_;
  const int rpn_cls_output_size_;
  const int rpn_dir_output_size_;
  const float pillar_x_size_;
  const float pillar_y_size_;
  const float pillar_z_size_;
  const float min_x_range_;
  const float min_y_range_;
  const float min_z_range_;
  const float max_x_range_;
  const float max_y_range_;
  const float max_z_range_;
  const int batch_size_;
  const int num_features_;
  const int num_threads_;
  const int num_box_corners_;
  const int num_output_box_feature_;

  int host_pillar_count_[1];

  std::vector<int> max_num_pillars_init_;
  std::vector<int> grid_size_init_;
  std::vector<int> num_anchor_init_;

  std::fstream f_cost_time_;

  std::string opencl_kernel_path_;

  cl::Buffer dev_x_coors_;  // Array that holds the coordinates of corresponding
                            // pillar in x
  cl::Buffer dev_y_coors_;  // Array that holds the coordinates of corresponding
                            // pillar in y
  cl::Buffer dev_num_points_per_pillar_;  // Array that stores the number of
                                          // points in the corresponding pillar
  cl::Buffer
      dev_sparse_pillar_map_;  // Mask with values 0 or 1 that specifies if the
                               // corresponding pillar has points or not
  cl::Buffer dev_cumsum_workspace_;  // Device variable used as temporary
                                     // storage of the cumulative sum during the
                                     // anchor mask creation

  // variables to store the pillar's points
  cl::Buffer dev_pillar_x_;
  cl::Buffer dev_pillar_y_;
  cl::Buffer dev_pillar_z_;
  cl::Buffer dev_pillar_i_;

  // variables to store the pillar coordinates in the pillar grid
  cl::Buffer dev_x_coors_for_sub_shaped_;
  cl::Buffer dev_y_coors_for_sub_shaped_;

  // Pillar mask used to ignore the features generated with empty pillars
  cl::Buffer dev_pillar_feature_mask_;

  // Mask used to filter the anchors in regions with input points
  cl::Buffer dev_anchor_mask_;

  // Device memory used to store the RPN input feature map after Scatter
  cl::Buffer dev_scattered_feature_;

  // Device memory locations to store the object detections
  cl::Buffer dev_filtered_box_;
  cl::Buffer dev_filtered_score_;
  cl::Buffer dev_multiclass_score_;
  cl::Buffer dev_filtered_dir_;
  cl::Buffer dev_filtered_class_id_;
  cl::Buffer dev_box_for_nms_;
  cl::Buffer dev_filter_count_;

  std::unique_ptr<PreProcess> preprocess_points_ptr_;
  std::unique_ptr<Scatter> scatter_ptr_;
  std::unique_ptr<PostProcess> postprocess_ptr_;
  std::unique_ptr<AnchorGrid> anchor_grid_ptr_;

  // for opencl
  std::shared_ptr<cl::Context> context_ = nullptr;
  std::shared_ptr<cl::CommandQueue> command_queue_ = nullptr;
  std::shared_ptr<cl::CommandQueue> command_queue_pre_ = nullptr;
  std::shared_ptr<cl::CommandQueue> command_queue_infer_post_ = nullptr;
  std::shared_ptr<cl::Device> device_ = nullptr;

  uint64_t gpu_global_memory_cache_size_;
  uint32_t gpu_compute_unit_;
  uint32_t gpu_max_frequent_;

  std::vector<std::string> program_names_{"memset.cl"};

  std::map<std::string, std::vector<std::string>> program_2_kernel_names_{
      {"memset.cl", {"PointpillarMemset", "ScatterMemset"}}};

  std::map<std::string, std::string> name_2_source_;
  std::map<std::string, cl::Program> name_2_program_;
  std::map<std::string, std::vector<unsigned char>> name_2_binary_;
  std::map<std::string, cl::Program> name_2_binary_program_;
  std::map<std::string, cl::Kernel> name_2_kernel_;

  std::string build_option_str_{
      "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math"};

  std::shared_ptr<OCLTOOL> ocl_tool_ = nullptr;

  int threads_for_opencl_ = 64;

 public:
  PointPillars() = delete;

  /**
   * @brief Constructor
   * @param[in] score_threshold Score threshold for filtering output
   * @param[in] nms_overlap_threshold IOU threshold for NMS
   * @param[in] config PointPillars net configuration file
   */
  PointPillars(const float *score_threshold, const float *nms_overlap_threshold,
               const PointPillarsConfig &config);

  ~PointPillars();

  /**
   * @brief Call PointPillars to perform the end-to-end object detection chain
   * @param[in] in_points_array Pointcloud array
   * @param[in] in_num_points Number of points
   * @param[in] detections Network output bounding box list
   * @details This is the main public interface to run the algorithm
   */
  void Detect(const float *in_points_array, const int in_num_points,
              std::vector<ObjectDetection> &detections);

  /**
   * @brief Preprocess points
   * @param[in] in_points_array pointcloud array
   * @param[in] in_num_points Number of points
   * @details Call oneAPI preprocess
   */
  void PreProcessing(const float *in_points_array, const int in_num_points);
  void CreateAnchorMask();
  void PfeInfer();
  void ScatterBack();
  void RpnInfer();
  void PostProcessing(std::vector<ObjectDetection> &detections);

 private:
  // TODO: ov::CompiledModel
  InferenceEngine::ExecutableNetwork pfe_exe_network_;
  std::map<std::string, cl::Buffer> pfe_input_map_;
  std::map<std::string, cl::Buffer> rpn_output_map_;
  InferenceEngine::ExecutableNetwork rpn_exe_network_;

  cl::Buffer pfe_output_;
  cl::Buffer rpn_1_output_;
  cl::Buffer rpn_2_output_;
  cl::Buffer rpn_3_output_;

  InferenceEngine::InferRequest::Ptr pfe_infer_request_ptr_;
  InferenceEngine::InferRequest::Ptr rpn_infer_request_ptr_;

  void InitComponents();

  /**
   * @brief Memory allocation for device memory
   * @details Called in the constructor
   */
  void DeviceMemoryMalloc();

  /**
   * @brief Setup the PFE executable network
   * @details Setup the PFE network
   */
  void SetupPfeNetwork();

  /**
   * @brief Setup the RPN executable network
   * @param[in] resizeInput If false, the network is not adapted to input size
   * changes
   * @details Setup the RPN network
   */
  void SetupRpnNetwork(bool resize_input);

  // for opencl
  bool OpenCLInit();
  void MakeOCLKernel();
};
}  // namespace pointpillars

#endif  //__POINTPILLARS_FOR_LIDAR_3D_HPP__
