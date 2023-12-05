#pragma once
#include <fstream>

#include "CL/cl2.hpp"
#include "cl_utils.hpp"

namespace pointpillars {

/**
 * PointPillar's PreProcessing
 *
 * Convert 3D point cloud data into 2D-grid/pillar form
 * to be able to feed it into the PillarFeatureNetwork
 */
class PreProcess {
 private:
  // initialzer list
  const int max_num_pillars_;
  const int max_num_points_per_pillar_;
  const int grid_x_size_;
  const int grid_y_size_;
  const int grid_z_size_;
  const float pillar_x_size_;
  const float pillar_y_size_;
  const float pillar_z_size_;
  const float min_x_range_;
  const float min_y_range_;
  const float min_z_range_;
  // end initalizer list

  std::string opencl_kernel_path_;

  cl::Buffer dev_pillar_x_in_coors_;
  cl::Buffer dev_pillar_y_in_coors_;
  cl::Buffer dev_pillar_z_in_coors_;
  cl::Buffer dev_pillar_i_in_coors_;

  cl::Buffer dev_pillar_count_histo_;

  cl::Buffer dev_counter_;
  cl::Buffer dev_pillar_count_;
  cl::Buffer dev_x_coors_for_sub_;
  cl::Buffer dev_y_coors_for_sub_;

  // for opencl
  std::vector<int> dev_pillar_count_histo_init_;
  int dev_counter_init_[1] = {0};

  const std::shared_ptr<cl::Context> context_;
  const std::shared_ptr<cl::CommandQueue> command_queue_;
  const std::shared_ptr<cl::Device> device_;

  std::vector<std::string> program_names{"memset.cl", "preprocess.cl"};

  std::map<std::string, std::vector<std::string>> program_2_kernel_names{
      {"memset.cl", {"PreprocessMemset"}},
      {"preprocess.cl",
       {"MakePillarHistoKernel", "MakePillarIndexKernel",
        "MakePillarFeatureKernel", "MakeExtraNetworkInputKernel"}}};

  std::map<std::string, std::string> name_2_source;
  std::map<std::string, cl::Program> name_2_program;
  std::map<std::string, std::vector<unsigned char>> name_2_binary;
  std::map<std::string, cl::Program> name_2_binary_program;
  std::map<std::string, cl::Kernel> name_2_kernel;

  std::string build_option_str{
      "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math"};

  std::shared_ptr<OCLTOOL> ocl_tool_ = nullptr;
  cl_int errCL;

  int threads_for_opencl = 64;

  // for opencl
  bool OpenCLInit();
  void MakeOCLKernel();

 public:
  /**
   * @brief Constructor
   * @param[in] max_num_pillars Maximum number of pillars
   * @param[in] max_points_per_pillar Maximum number of points per pillar
   * @param[in] grid_x_size Number of pillars in x-coordinate
   * @param[in] grid_y_size Number of pillars in y-coordinate
   * @param[in] grid_z_size Number of pillars in z-coordinate
   * @param[in] pillar_x_size Size of x-dimension for a pillar
   * @param[in] pillar_y_size Size of y-dimension for a pillar
   * @param[in] pillar_z_size Size of z-dimension for a pillar
   * @param[in] min_x_range Minimum x value for pointcloud
   * @param[in] min_y_range Minimum y value for pointcloud
   * @param[in] min_z_range Minimum z value for pointcloud
   * @param[in] num_box_corners Number of corners for 2D box
   */
  PreProcess(const int max_num_pillars, const int max_points_per_pillar,
             const int grid_x_size, const int grid_y_size,
             const int grid_z_size, const float pillar_x_size,
             const float pillar_y_size, const float pillar_z_size,
             const float min_x_range, const float min_y_range,
             const float min_z_range, const std::string& opencl_kernel_path,
             const std::shared_ptr<cl::Context>& context,
             const std::shared_ptr<cl::CommandQueue>& command_queue,
             const std::shared_ptr<cl::Device>& device);
  ~PreProcess();

  /**
   * @brief Preprocessing for input pointcloud
   * @param[in] dev_points Pointcloud array
   * @param[in] in_num_points The number of points
   * @param[in] dev_x_coors X-coordinate indexes for corresponding pillars
   * @param[in] dev_y_coors Y-coordinate indexes for corresponding pillars
   * @param[in] dev_num_points_per_pillar Number of points in corresponding
   * pillars
   * @param[in] dev_pillar_x X-coordinate values for points in each pillar
   * @param[in] dev_pillar_y Y-coordinate values for points in each pillar
   * @param[in] dev_pillar_z Z-coordinate values for points in each pillar
   * @param[in] dev_pillar_i Intensity values for points in each pillar
   * @param[in] dev_x_coors_for_sub_shaped Array for x substraction in the
   * network
   * @param[in] dev_y_coors_for_sub_shaped Array for y substraction in the
   * network
   * @param[in] dev_pillar_feature_mask Mask to make pillars' feature zero where
   * no points in the pillars
   * @param[in] dev_sparse_pillar_map Grid map representation for
   * pillar-occupancy
   * @param[in] host_pillar_count The numnber of valid pillars for an input
   * pointcloud
   * @details Convert pointcloud to pillar representation
   */
  void DoPreProcess(const cl::Buffer& dev_points, const int in_num_points,
                    cl::Buffer& dev_x_coors, cl::Buffer& dev_y_coors,
                    cl::Buffer& dev_num_points_per_pillar,
                    cl::Buffer& dev_pillar_x, cl::Buffer& dev_pillar_y,
                    cl::Buffer& dev_pillar_z, cl::Buffer& dev_pillar_i,
                    cl::Buffer& dev_x_coors_for_sub_shaped,
                    cl::Buffer& dev_y_coors_for_sub_shaped,
                    cl::Buffer& dev_pillar_feature_mask,
                    cl::Buffer& dev_sparse_pillar_map, int* host_pillar_count);
};
}  // namespace pointpillars
