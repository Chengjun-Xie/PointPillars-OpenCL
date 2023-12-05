#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cl_utils.hpp"

namespace pointpillars {

/**
 * PointPillar's Scatter.
 * Converts learned features (output from PFE network = 1st CNN) from dense
 * tensors to sparse pseudo image.
 */
class Scatter {
 private:
  const int num_features_;     // The number of features per pillar
  const int max_num_pillars_;  // Maximum number of pillars
  const int grid_x_size_;      // Number of pillars in x-coordinate
  const int grid_y_size_;      // Number of pillars in x-coordinate

  // for opencl
  std::string opencl_kernel_path_;
  std::shared_ptr<cl::Context> context_;
  std::shared_ptr<cl::CommandQueue> command_queue_;
  std::shared_ptr<cl::Device> device_;

  std::vector<std::string> program_names_{"scatter.cl"};

  std::map<std::string, std::vector<std::string>> program_2_kernel_names_{
      {"scatter.cl", {"ScatterKernel"}}};

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
   * @param[in] num_features The number of features per pillar
   * @param[in] max_num_pillars Maximum number of pillars
   * @param[in] grid_x_size Number of pillars in x-coordinate
   * @param[in] grid_y_size Number of pillars in y-coordinate
   */
  Scatter(const int num_features, const int max_num_pillars,
          const int grid_x_size, const int grid_y_size, const std::string&,
          const std::shared_ptr<cl::Context>&,
          const std::shared_ptr<cl::CommandQueue>&,
          const std::shared_ptr<cl::Device>&);

  /**
   * @brief Call scatter kernel
   * @param[in] pillar_count The valid number of pillars
   * @param[in] x_coors X-coordinate indexes for corresponding pillars
   * @param[in] y_coors Y-coordinate indexes for corresponding pillars
   * @param[in] pfe_output Output from Pillar Feature Extractor
   * @param[out] scattered_feature Gridmap representation for pillars' feature
   * @details Allocate pillars in gridmap based on index(coordinates)
   * information
   */
  // void DoScatter(const int pillar_count, int *x_coors, int *y_coors, float
  // *pfe_output, float *scattered_feature);
  void DoScatter(const int pillar_count, cl::Buffer& x_coors,
                 cl::Buffer& y_coors, cl::Buffer& pfe_output,
                 cl::Buffer& scattered_feature);
};
}  // namespace pointpillars
