#pragma once

#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cl_utils.hpp"
#include "pointpillars/pointpillars_config.hpp"

namespace pointpillars {

/**
 * AnchorGrid
 *
 * The AnchorGrid class generates anchors in different sizes and orientations
 * for every location in the grid. Anchor based methods are used in object
 * detection in which a list of predefined boxes are refined by a CNN.
 *
 */
class AnchorGrid {
 public:
  /**
   * @brief Constructor
   *
   * Class used to generate the anchor grid which is used as a prior box list
   * during object detection
   *
   * @param[in] config Configuration used to generate anchors
   */
  AnchorGrid(AnchorGridConfig &config, const std::string &opencl_kernel_path,
             const std::shared_ptr<cl::Context> &context,
             const std::shared_ptr<cl::CommandQueue> &command_queue,
             const std::shared_ptr<cl::Device> &device);
  ~AnchorGrid();

  AnchorGridConfig config_;

  // Pointers to device memory locations for the anchors
  cl::Buffer dev_anchors_px_;
  cl::Buffer dev_anchors_py_;
  cl::Buffer dev_anchors_pz_;
  cl::Buffer dev_anchors_dx_;
  cl::Buffer dev_anchors_dy_;
  cl::Buffer dev_anchors_dz_;
  cl::Buffer dev_anchors_ro_;

  // Get size/number of anchors
  std::size_t size() { return num_anchors_; }

  // Generate default anchors
  void GenerateAnchors();

  // Creates an anchor mask that can be used to ignore anchors in regions
  // without points Input is the current pillar map (map, width, height, size in
  // x, size in y, size in z) Output are the created anchors
  void CreateAnchorMask(int *dev_pillar_map, const int pillar_map_w,
                        const int pillar_map_h, const float pillar_size_x,
                        const float pillar_size_y, int *dev_anchor_mask,
                        int *dev_pillar_workspace);

 private:
  std::size_t num_anchors_{0u};
  std::size_t mh_{0u};
  std::size_t mw_{0u};
  std::size_t mc_{0u};
  std::size_t mr_{0u};

  // Anchor pointers on the host
  // Only required for initialization
  float *dev_anchors_rad_{nullptr};
  float *host_anchors_px_{nullptr};
  float *host_anchors_py_{nullptr};
  float *host_anchors_pz_{nullptr};
  float *host_anchors_dx_{nullptr};
  float *host_anchors_dy_{nullptr};
  float *host_anchors_dz_{nullptr};
  float *host_anchors_ro_{nullptr};
  float *host_anchors_rad_;

  // Clear host memory
  void ClearHostMemory();

  // Allocate host memory
  void AllocateHostMemory();

  // Allocate device memory
  void AllocateDeviceMemory();

  // Move anchors from the host system to the target execution device
  void MoveAnchorsToDevice();

  // Internal function to create anchor mask
  void MaskAnchors(const float *dev_anchors_px, const float *dev_anchors_py,
                   const int *dev_pillar_map, int *dev_anchor_mask,
                   const float *dev_anchors_rad, const float min_x_range,
                   const float min_y_range, const float pillar_x_size,
                   const float pillar_y_size, const int grid_x_size,
                   const int grid_y_size, const int c, const int r, const int h,
                   const int w);

  // for opencl
  std::string opencl_kernel_path_;
  std::shared_ptr<cl::Context> context_;
  std::shared_ptr<cl::CommandQueue> command_queue_;
  std::shared_ptr<cl::Device> device_;

  uint64_t gpu_global_memory_cache_size_;
  uint32_t gpu_compute_unit_;
  uint32_t gpu_max_frequent_;

  std::vector<std::string> program_names_{"anchorgrid.cl"};

  std::map<std::string, std::vector<std::string>> program_2_kernel_names_{
      {"anchorgrid.cl", {"MaskAnchorsSimpleKernel"}}};

  std::map<std::string, std::string> name_2_source_;
  std::map<std::string, cl::Program> name_2_program_;
  std::map<std::string, std::vector<unsigned char>> name_2_binary_;
  std::map<std::string, cl::Program> name_2_binary_program_;
  std::map<std::string, cl::Kernel> name_2_kernel_;

  std::string build_option_str_{
      "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math"};

  std::shared_ptr<OCLTOOL> ocl_tool_ = nullptr;

  int threads_for_opencl_ = 64;

  bool OpenCLInit();
  void MakeOCLKernel();
};

}  // namespace pointpillars