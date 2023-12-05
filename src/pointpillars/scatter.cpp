#include "pointpillars/scatter.hpp"

#include <algorithm>

namespace pointpillars {

Scatter::Scatter(const int num_features, const int max_num_pillars,
                 const int grid_x_size, const int grid_y_size,
                 const std::string& opencl_kernel_path,
                 const std::shared_ptr<cl::Context>& context,
                 const std::shared_ptr<cl::CommandQueue>& command_queue,
                 const std::shared_ptr<cl::Device>& device)
    : num_features_(num_features),
      max_num_pillars_(max_num_pillars),
      grid_x_size_(grid_x_size),
      grid_y_size_(grid_y_size),
      opencl_kernel_path_{opencl_kernel_path},
      context_(context),
      command_queue_(command_queue),
      device_(device) {
  OpenCLInit();
  MakeOCLKernel();
}

void Scatter::DoScatter(const int pillar_count, cl::Buffer& x_coors,
                        cl::Buffer& y_coors, cl::Buffer& pfe_output,
                        cl::Buffer& scattered_feature) {
  // Launch the scatter kernel on each (n-pillar , m-feature)
  cl::Event eventCL;
  cl_int errCL;
  int idx = 0;
  int length = pillar_count * num_features_;
  std::string kernel_name{"ScatterKernel"};
  auto kernel = name_2_kernel_[kernel_name];
  kernel.setArg(idx++, x_coors);
  kernel.setArg(idx++, y_coors);
  kernel.setArg(idx++, pfe_output);
  kernel.setArg(idx++, scattered_feature);
  kernel.setArg(idx++, max_num_pillars_);
  kernel.setArg(idx++, grid_x_size_);
  kernel.setArg(idx++, grid_y_size_);
  kernel.setArg(idx++, num_features_);
  kernel.setArg(idx++, length);
  int nums_opencl = DIVUP(length, threads_for_opencl_) * threads_for_opencl_;
  auto global_ndrange = cl::NDRange(nums_opencl);
  auto local_ndrange = cl::NDRange(threads_for_opencl_);
  errCL = command_queue_->enqueueNDRangeKernel(
      kernel, cl::NullRange, global_ndrange, local_ndrange, NULL, &eventCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "Scatter::DoScatter enqueueNDRangeKernel ScatterKernel fail: "
              << errCL << std::endl;
  }

  eventCL.wait();
}

bool Scatter::OpenCLInit() {
  ocl_tool_.reset(new OCLTOOL(opencl_kernel_path_, program_names_,
                              program_2_kernel_names_, context_, device_,
                              build_option_str_));

  return true;
}

void Scatter::MakeOCLKernel() {
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
