#include "pointpillars/preprocess.hpp"

#include <algorithm>
#include <iostream>

#include "pointpillars/common.hpp"

namespace pointpillars {

PreProcess::PreProcess(const int max_num_pillars,
                       const int max_points_per_pillar, const int grid_x_size,
                       const int grid_y_size, const int grid_z_size,
                       const float pillar_x_size, const float pillar_y_size,
                       const float pillar_z_size, const float min_x_range,
                       const float min_y_range, const float min_z_range,
                       const std::string& opencl_kernel_path,
                       const std::shared_ptr<cl::Context>& context,
                       const std::shared_ptr<cl::CommandQueue>& command_queue,
                       const std::shared_ptr<cl::Device>& device)
    : max_num_pillars_(max_num_pillars),
      max_num_points_per_pillar_(max_points_per_pillar),
      grid_x_size_(grid_x_size),
      grid_y_size_(grid_y_size),
      grid_z_size_(grid_z_size),
      pillar_x_size_(pillar_x_size),
      pillar_y_size_(pillar_y_size),
      pillar_z_size_(pillar_z_size),
      min_x_range_(min_x_range),
      min_y_range_(min_y_range),
      min_z_range_(min_z_range),
      opencl_kernel_path_{opencl_kernel_path},
      context_(context),
      command_queue_(command_queue),
      device_(device) {
  // allocate memory
  dev_pillar_x_in_coors_ = cl::Buffer(
      *context_, CL_MEM_READ_WRITE,
      grid_y_size_ * grid_x_size_ * max_num_points_per_pillar_ * sizeof(float),
      NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_pillar_x_in_coors_ buffer malloc fail: " << errCL
              << std::endl;
  }

  dev_pillar_y_in_coors_ = cl::Buffer(
      *context_, CL_MEM_READ_WRITE,
      grid_y_size_ * grid_x_size_ * max_num_points_per_pillar_ * sizeof(float),
      NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_pillar_y_in_coors_ buffer malloc fail: " << errCL
              << std::endl;
  }

  dev_pillar_z_in_coors_ = cl::Buffer(
      *context_, CL_MEM_READ_WRITE,
      grid_y_size_ * grid_x_size_ * max_num_points_per_pillar_ * sizeof(float),
      NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_pillar_z_in_coors_ buffer malloc fail: " << errCL
              << std::endl;
  }

  dev_pillar_i_in_coors_ = cl::Buffer(
      *context_, CL_MEM_READ_WRITE,
      grid_y_size_ * grid_x_size_ * max_num_points_per_pillar_ * sizeof(float),
      NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_pillar_i_in_coors_ buffer malloc fail: " << errCL
              << std::endl;
  }

  dev_pillar_count_histo_ =
      cl::Buffer(*context_, CL_MEM_READ_WRITE,
                 grid_y_size_ * grid_x_size_ * sizeof(int), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_pillar_count_histo_ buffer malloc fail: " << errCL
              << std::endl;
  }

  dev_counter_ =
      cl::Buffer(*context_, CL_MEM_READ_WRITE, sizeof(int), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_counter_ buffer malloc fail: " << errCL << std::endl;
  }

  dev_pillar_count_ =
      cl::Buffer(*context_, CL_MEM_READ_WRITE, sizeof(int), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_pillar_count_ buffer malloc fail: " << errCL << std::endl;
  }

  dev_x_coors_for_sub_ =
      cl::Buffer(*context_, CL_MEM_READ_WRITE, max_num_pillars_ * sizeof(float),
                 NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_x_coors_for_sub_ buffer malloc fail: " << errCL
              << std::endl;
  }

  dev_y_coors_for_sub_ =
      cl::Buffer(*context_, CL_MEM_READ_WRITE, max_num_pillars_ * sizeof(float),
                 NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_y_coors_for_sub_ buffer malloc fail: " << errCL
              << std::endl;
  }

  OpenCLInit();
  MakeOCLKernel();

  dev_pillar_count_histo_init_ =
      std::move(std::vector<int>(grid_y_size_ * grid_x_size_, 0));
}

PreProcess::~PreProcess() {}

void PreProcess::DoPreProcess(
    const cl::Buffer& dev_points, const int in_num_points,
    cl::Buffer& dev_x_coors, cl::Buffer& dev_y_coors,
    cl::Buffer& dev_num_points_per_pillar, cl::Buffer& dev_pillar_x,
    cl::Buffer& dev_pillar_y, cl::Buffer& dev_pillar_z,
    cl::Buffer& dev_pillar_i, cl::Buffer& dev_x_coors_for_sub_shaped,
    cl::Buffer& dev_y_coors_for_sub_shaped, cl::Buffer& dev_pillar_feature_mask,
    cl::Buffer& dev_sparse_pillar_map, int* host_pillar_count) {
  // Set Pillar input features to 0
  {
    cl::Event eventCL;
    cl_int errCL;
    int idx = 0;
    int nums = grid_y_size_ * grid_x_size_ * max_num_points_per_pillar_;
    std::string kernel_name{"PreprocessMemset"};
    auto kernel = name_2_kernel[kernel_name];
    kernel.setArg(idx++, dev_pillar_x_in_coors_);
    kernel.setArg(idx++, dev_pillar_y_in_coors_);
    kernel.setArg(idx++, dev_pillar_z_in_coors_);
    kernel.setArg(idx++, dev_pillar_i_in_coors_);
    errCL = kernel.setArg(idx++, nums);
    if (CL_SUCCESS != errCL) {
      std::cout << "PreProcess::DoPreProcess PreprocessMemset setArg fail: "
                << errCL << std::endl;
    }

    size_t nums_opencl = DIVUP(nums, threads_for_opencl) * threads_for_opencl;
    auto global_ndrange = cl::NDRange(nums_opencl);
    auto local_ndrange = cl::NDRange(threads_for_opencl);
    errCL = command_queue_->enqueueNDRangeKernel(
        kernel, cl::NullRange, global_ndrange, local_ndrange, NULL, &eventCL);
    if (CL_SUCCESS != errCL) {
      std::cout << "PreProcess::DoPreProcess enqueueNDRangeKernel "
                   "PreprocessMemset fail: "
                << errCL << std::endl;
    }
    eventCL.wait();

    errCL = command_queue_->enqueueWriteBuffer(
        dev_pillar_count_histo_, true, 0,
        grid_y_size_ * grid_x_size_ * sizeof(int),
        static_cast<void*>(dev_pillar_count_histo_init_.data()), NULL, NULL);
    if (CL_SUCCESS != errCL) {
      std::cout << "PreProcess::DoPreProcess enqueueWriteBuffer "
                   "dev_pillar_count_histo_ fail: "
                << errCL << std::endl;
    }

    errCL = command_queue_->enqueueWriteBuffer(
        dev_counter_, true, 0, sizeof(int),
        static_cast<void*>(dev_counter_init_), NULL, NULL);
    if (CL_SUCCESS != errCL) {
      std::cout
          << "PreProcess::DoPreProcess enqueueWriteBuffer dev_counter_ fail: "
          << errCL << std::endl;
    }

    errCL = command_queue_->enqueueWriteBuffer(
        dev_pillar_count_, true, 0, sizeof(int),
        static_cast<void*>(dev_counter_init_), NULL, NULL);
    if (CL_SUCCESS != errCL) {
      std::cout << "PreProcess::DoPreProcess enqueueWriteBuffer "
                   "dev_pillar_count_ fail: "
                << errCL << std::endl;
    }
  }

  // Use the point cloud data to generate the pillars
  // This will create create assign the point to the corresponding pillar in the
  // grid. A maximum number of points can be assigned to a single pillar.
  {
    cl::Event eventCL;
    int idx = 0;
    cl_int errCL;
    std::string kernel_name{"MakePillarHistoKernel"};
    auto kernel = name_2_kernel[kernel_name];
    kernel.setArg(idx++, dev_points);
    kernel.setArg(idx++, dev_pillar_x_in_coors_);
    kernel.setArg(idx++, dev_pillar_y_in_coors_);
    kernel.setArg(idx++, dev_pillar_z_in_coors_);
    kernel.setArg(idx++, dev_pillar_i_in_coors_);
    kernel.setArg(idx++, dev_pillar_count_histo_);
    kernel.setArg(idx++, in_num_points);
    kernel.setArg(idx++, max_num_points_per_pillar_);
    kernel.setArg(idx++, grid_x_size_);
    kernel.setArg(idx++, grid_y_size_);
    kernel.setArg(idx++, grid_z_size_);
    kernel.setArg(idx++, min_x_range_);
    kernel.setArg(idx++, min_y_range_);
    kernel.setArg(idx++, min_z_range_);
    kernel.setArg(idx++, pillar_x_size_);
    kernel.setArg(idx++, pillar_y_size_);
    kernel.setArg(idx++, pillar_z_size_);

    size_t num_opencl = DIVUP(in_num_points, 256) * 256;
    auto global_ndrange = cl::NDRange(num_opencl);
    auto local_ndrange = cl::NDRange(256);
    errCL = command_queue_->enqueueNDRangeKernel(
        kernel, cl::NullRange, global_ndrange, local_ndrange, NULL, &eventCL);
    if (CL_SUCCESS != errCL) {
      std::cout << "PreProcess::DoPreProcess enqueueNDRangeKernel "
                   "MakePillarHistoKernel fail: "
                << errCL << std::endl;
    }

    eventCL.wait();

    std::vector<int> point_counter;
    int point_total = 0;
    point_counter.resize(grid_y_size_ * grid_x_size_);

    errCL = command_queue_->enqueueReadBuffer(
        dev_pillar_count_histo_, true, 0,
        grid_x_size_ * grid_y_size_ * sizeof(int),
        static_cast<void*>(point_counter.data()), NULL, NULL);
    if (CL_SUCCESS != errCL) {
      std::cout << "PreProcess::DoPreProcess enqueueReadBuffer "
                   "dev_pillar_count_histo_ fail: "
                << errCL << std::endl;
    }

    for (auto item : point_counter) {
      point_total += item;
    }

    std::cout << "PreProcess::DoPreProcess total point cloud num  " << 
            point_total << std::endl;;
  }

  // Check which pillars contain points and mark them for use during feature
  // extraction.
  {
    cl::Event eventCL;
    cl_int errCL;
    int idx = 0;
    std::string kernel_name{"MakePillarIndexKernel"};
    auto kernel = name_2_kernel[kernel_name];
    kernel.setArg(idx++, dev_pillar_count_histo_);
    kernel.setArg(idx++, dev_counter_);
    kernel.setArg(idx++, dev_pillar_count_);
    kernel.setArg(idx++, dev_x_coors);
    kernel.setArg(idx++, dev_y_coors);
    kernel.setArg(idx++, dev_x_coors_for_sub_);
    kernel.setArg(idx++, dev_y_coors_for_sub_);
    kernel.setArg(idx++, dev_num_points_per_pillar);
    kernel.setArg(idx++, dev_sparse_pillar_map);
    kernel.setArg(idx++, max_num_pillars_);
    kernel.setArg(idx++, max_num_points_per_pillar_);
    kernel.setArg(idx++, grid_x_size_);
    kernel.setArg(idx++, grid_y_size_);
    kernel.setArg(idx++, min_x_range_);
    kernel.setArg(idx++, min_y_range_);
    kernel.setArg(idx++, pillar_x_size_);
    kernel.setArg(idx++, pillar_y_size_);

    size_t nums_opencl =
        DIVUP(grid_x_size_ * grid_y_size_, threads_for_opencl) *
        threads_for_opencl;
    auto global_ndrange = cl::NDRange(nums_opencl);
    auto local_ndrange = cl::NDRange(threads_for_opencl);
    errCL = command_queue_->enqueueNDRangeKernel(
        kernel, cl::NullRange, global_ndrange, local_ndrange, NULL, &eventCL);
    if (CL_SUCCESS != errCL) {
      std::cout << "PreProcess::DoPreProcess enqueueNDRangeKernel "
                   "MakePillarIndexKernel fail: "
                << errCL << std::endl;
    }

    eventCL.wait();
  }

  // Generate the first 4 pillar features in the input feature map.
  // This is a list of points up to max_num_points_per_pillar
  {
    command_queue_->enqueueReadBuffer(dev_pillar_count_, true, 0, sizeof(int),
                                      host_pillar_count, NULL, NULL);
    int total_point = host_pillar_count[0] * max_num_points_per_pillar_;

    cl::Event eventCL;
    cl_int errCL;
    int idx = 0;
    std::string kernel_name{"MakePillarFeatureKernel"};
    auto kernel = name_2_kernel[kernel_name];
    kernel.setArg(idx++, dev_pillar_x_in_coors_);
    kernel.setArg(idx++, dev_pillar_y_in_coors_);
    kernel.setArg(idx++, dev_pillar_z_in_coors_);
    kernel.setArg(idx++, dev_pillar_i_in_coors_);
    kernel.setArg(idx++, dev_pillar_x);
    kernel.setArg(idx++, dev_pillar_y);
    kernel.setArg(idx++, dev_pillar_z);
    kernel.setArg(idx++, dev_pillar_i);
    kernel.setArg(idx++, dev_x_coors);
    kernel.setArg(idx++, dev_y_coors);
    kernel.setArg(idx++, dev_num_points_per_pillar);
    kernel.setArg(idx++, max_num_points_per_pillar_);
    kernel.setArg(idx++, grid_x_size_);
    kernel.setArg(idx++, total_point);

    size_t nums_opencl =
        DIVUP(total_point, threads_for_opencl) * threads_for_opencl;
    auto global_ndrange = cl::NDRange(nums_opencl);
    auto local_ndrange = cl::NDRange(threads_for_opencl);
    errCL = command_queue_->enqueueNDRangeKernel(
        kernel, cl::NullRange, global_ndrange, local_ndrange, NULL, &eventCL);
    if (CL_SUCCESS != errCL) {
      std::cout << "PreProcess::DoPreProcess enqueueNDRangeKernel "
                   "MakePillarFeatureKernel fail: "
                << errCL << std::endl;
    }

    eventCL.wait();
  }

  // Generate the next features in the pillar input feature map:
  // (pillar_center_x, pillar_center_y, pillar_mask)
  {
    cl::Event eventCL;
    cl_int errCL;
    int idx = 0;
    int total_point = max_num_pillars_ * max_num_points_per_pillar_;
    std::string kernel_name{"MakeExtraNetworkInputKernel"};
    auto kernel = name_2_kernel[kernel_name];
    kernel.setArg(idx++, dev_x_coors_for_sub_);
    kernel.setArg(idx++, dev_y_coors_for_sub_);
    kernel.setArg(idx++, dev_num_points_per_pillar);
    kernel.setArg(idx++, dev_x_coors_for_sub_shaped);
    kernel.setArg(idx++, dev_y_coors_for_sub_shaped);
    kernel.setArg(idx++, dev_pillar_feature_mask);
    kernel.setArg(idx++, max_num_points_per_pillar_);
    kernel.setArg(idx++, total_point);

    size_t nums_opencl =
        DIVUP(total_point, threads_for_opencl) * threads_for_opencl;
    auto global_ndrange = cl::NDRange(nums_opencl);
    auto local_ndrange = cl::NDRange(threads_for_opencl);
    errCL = command_queue_->enqueueNDRangeKernel(
        kernel, cl::NullRange, global_ndrange, local_ndrange, NULL, &eventCL);
    if (CL_SUCCESS != errCL) {
      std::cout << "PreProcess::DoPreProcess enqueueNDRangeKernel "
                   "MakeExtraNetworkInputKernel fail: "
                << errCL << std::endl;
    }

    eventCL.wait();
  }
}

bool PreProcess::OpenCLInit() {
  ocl_tool_.reset(new OCLTOOL(opencl_kernel_path_, program_names,
                              program_2_kernel_names, context_, device_,
                              build_option_str));

  return true;
}

void PreProcess::MakeOCLKernel() {
#ifdef OCLDEBUG
  ocl_tool_->MakeKernelSource(name_2_source);
  ocl_tool_->MakeProgramFromSource(name_2_source, name_2_program);
  ocl_tool_->BuildProgramAndGetBinary(name_2_program, name_2_binary);
  ocl_tool_->SaveProgramBinary(name_2_binary);
#endif
  ocl_tool_->LoadBinaryAndMakeProgram(name_2_binary_program);
  ocl_tool_->MakeKernelFromBinary(name_2_binary_program, name_2_kernel);
}

}  // namespace pointpillars
