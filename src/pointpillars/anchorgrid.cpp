#include "pointpillars/anchorgrid.hpp"

#include <algorithm>

namespace pointpillars {

AnchorGrid::AnchorGrid(AnchorGridConfig &config,
                       const std::string &opencl_kernel_path,
                       const std::shared_ptr<cl::Context> &context,
                       const std::shared_ptr<cl::CommandQueue> &command_queue,
                       const std::shared_ptr<cl::Device> &device)
    : config_{config},
      opencl_kernel_path_{opencl_kernel_path},
      context_(context),
      command_queue_(command_queue),
      device_(device),
      dev_anchors_px_(nullptr), 
      dev_anchors_py_(nullptr),
      dev_anchors_pz_(nullptr),
      dev_anchors_dx_(nullptr), 
      dev_anchors_dy_(nullptr),
      dev_anchors_dz_(nullptr), 
      dev_anchors_ro_(nullptr),
      dev_anchors_rad_(nullptr),
      host_anchors_px_(nullptr),
      host_anchors_py_(nullptr),
      host_anchors_pz_(nullptr),
      host_anchors_dx_(nullptr),
      host_anchors_dy_(nullptr),
      host_anchors_dz_(nullptr),
      host_anchors_ro_(nullptr),
      host_anchors_rad_(nullptr) {
  mc_ = config_.anchors.size();
  mr_ = config_.rotations.size();

  // In the anchor map the Width is along y and the Height along x
  mh_ = static_cast<std::size_t>((config_.max_x_range - config_.min_x_range) /
                                 config_.x_stride);
  mw_ = static_cast<std::size_t>((config_.max_y_range - config_.min_y_range) /
                                 config_.y_stride);
  num_anchors_ = mw_ * mh_ * mc_ * mr_;

  OpenCLInit();
  MakeOCLKernel();
}

AnchorGrid::~AnchorGrid() { ClearHostMemory(); }

void AnchorGrid::GenerateAnchors() {
  AllocateHostMemory();

  // Minimum (x, y) anchor grid coordinates + pillar center offset
  float x_offset = config_.min_x_range + 0.5f * config_.x_stride;
  float y_offset = config_.min_y_range + 0.5f * config_.y_stride;

  // In the anchor map the Width is along y and the Height along x, c is the
  // class, r is the rotation
  for (size_t y = 0; y < mw_; y++) {
    for (size_t x = 0; x < mh_; x++) {
      for (size_t c = 0; c < mc_; c++) {
        for (size_t r = 0; r < mr_; r++) {
          std::size_t index = y * mh_ * mc_ * mr_ + x * mc_ * mr_ + c * mr_ + r;

          // Set anchor grid locations at the center of the pillar
          host_anchors_px_[index] =
              static_cast<float>(x) * config_.x_stride + x_offset;
          host_anchors_py_[index] =
              static_cast<float>(y) * config_.y_stride + y_offset;

          // Assign z as dz
          host_anchors_pz_[index] = config_.anchors[c].dz;

          // Assign current anchor rotation r
          host_anchors_ro_[index] = config_.rotations[r];

          // Assign anchors sizes for the given class c
          host_anchors_dx_[index] = config_.anchors[c].x;
          host_anchors_dy_[index] = config_.anchors[c].y;
          host_anchors_dz_[index] = config_.anchors[c].z;
        }
      }
    }
  }

  // host_anchors_rad_ is used to optimize the decoding by precalculating an
  // effective radius around the anchor
  for (std::size_t c = 0; c < mc_; c++) {
    host_anchors_rad_[c] = std::min(config_.anchors[c].x, config_.anchors[c].y);
  }

  MoveAnchorsToDevice();
}

void AnchorGrid::AllocateHostMemory() {
  host_anchors_px_ = new float[num_anchors_];
  host_anchors_py_ = new float[num_anchors_];
  host_anchors_pz_ = new float[num_anchors_];
  host_anchors_dx_ = new float[num_anchors_];
  host_anchors_dy_ = new float[num_anchors_];
  host_anchors_dz_ = new float[num_anchors_];
  host_anchors_ro_ = new float[num_anchors_];

#pragma unroll(8)
  for (std::size_t i = 0; i < num_anchors_; i++) {
    host_anchors_px_[i] = 0.f;
    host_anchors_py_[i] = 0.f;
    host_anchors_pz_[i] = 0.f;
    host_anchors_dx_[i] = 0.f;
    host_anchors_dy_[i] = 0.f;
    host_anchors_dz_[i] = 0.f;
    host_anchors_ro_[i] = 0.f;
  }

  host_anchors_rad_ = new float[mc_];

  for (std::size_t i = 0; i < mc_; i++) {
    host_anchors_rad_[i] = 0.f;
  }
}

void AnchorGrid::ClearHostMemory() {
  delete[] host_anchors_px_;
  delete[] host_anchors_py_;
  delete[] host_anchors_pz_;
  delete[] host_anchors_dx_;
  delete[] host_anchors_dy_;
  delete[] host_anchors_dz_;
  delete[] host_anchors_ro_;

  delete[] host_anchors_rad_;

  host_anchors_px_ = nullptr;
  host_anchors_py_ = nullptr;
  host_anchors_pz_ = nullptr;
  host_anchors_dx_ = nullptr;
  host_anchors_dy_ = nullptr;
  host_anchors_dz_ = nullptr;
  host_anchors_ro_ = nullptr;
  host_anchors_rad_ = nullptr;
}

void AnchorGrid::AllocateDeviceMemory() {
  dev_anchors_px_ = cl::Buffer(*context_, CL_MEM_READ_WRITE,
                               num_anchors_ * sizeof(float), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "PointPillars::AnchorGrid dev_anchors_px_ buffer malloc fail: "
              << errCL << std::endl;
  }

  dev_anchors_py_ = cl::Buffer(*context_, CL_MEM_READ_WRITE,
                               num_anchors_ * sizeof(float), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "PointPillars::AnchorGrid dev_anchors_py_ buffer malloc fail: "
              << errCL << std::endl;
  }

  dev_anchors_pz_ = cl::Buffer(*context_, CL_MEM_READ_WRITE,
                               num_anchors_ * sizeof(float), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "PointPillars::AnchorGrid dev_anchors_pz_ buffer malloc fail: "
              << errCL << std::endl;
  }

  dev_anchors_dx_ = cl::Buffer(*context_, CL_MEM_READ_WRITE,
                               num_anchors_ * sizeof(float), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "PointPillars::AnchorGrid dev_anchors_px_ buffer malloc fail: "
              << errCL << std::endl;
  }

  dev_anchors_dy_ = cl::Buffer(*context_, CL_MEM_READ_WRITE,
                               num_anchors_ * sizeof(float), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "PointPillars::AnchorGrid dev_anchors_dy_ buffer malloc fail: "
              << errCL << std::endl;
  }

  dev_anchors_dz_ = cl::Buffer(*context_, CL_MEM_READ_WRITE,
                               num_anchors_ * sizeof(float), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "PointPillars::AnchorGrid dev_anchors_dz_ buffer malloc fail: "
              << errCL << std::endl;
  }

  dev_anchors_ro_ = cl::Buffer(*context_, CL_MEM_READ_WRITE,
                               num_anchors_ * sizeof(float), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "PointPillars::AnchorGrid dev_anchors_ro_ buffer malloc fail: "
              << errCL << std::endl;
  }

  dev_anchors_rad_ = cl::Buffer(*context_, CL_MEM_READ_WRITE,
                                mc_ * sizeof(float), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout
        << "PointPillars::AnchorGrid dev_anchors_rad_ buffer malloc fail: "
        << errCL << std::endl;
  }
}

void AnchorGrid::MoveAnchorsToDevice() {
  AllocateDeviceMemory();

  errCL = command_queue_->enqueueWriteBuffer(
      dev_anchors_px_, true, 0, num_anchors_ * sizeof(float),
      static_cast<void *>(host_anchors_px_), NULL, NULL);
  if (CL_SUCCESS != errCL) {
    std::cout
        << "PointPillars::AnchorGrid enqueueWriteBuffer dev_anchors_px_ fail: "
        << errCL << std::endl;
  }

  errCL = command_queue_->enqueueWriteBuffer(
      dev_anchors_py_, true, 0, num_anchors_ * sizeof(float),
      static_cast<void *>(host_anchors_py_), NULL, NULL);
  if (CL_SUCCESS != errCL) {
    std::cout
        << "PointPillars::AnchorGrid enqueueWriteBuffer dev_anchors_py_ fail: "
        << errCL << std::endl;
  }

  errCL = command_queue_->enqueueWriteBuffer(
      dev_anchors_pz_, true, 0, num_anchors_ * sizeof(float),
      static_cast<void *>(host_anchors_pz_), NULL, NULL);
  if (CL_SUCCESS != errCL) {
    std::cout
        << "PointPillars::AnchorGrid enqueueWriteBuffer dev_anchors_pz_ fail: "
        << errCL << std::endl;
  }

  errCL = command_queue_->enqueueWriteBuffer(
      dev_anchors_dx_, true, 0, num_anchors_ * sizeof(float),
      static_cast<void *>(host_anchors_dx_), NULL, NULL);
  if (CL_SUCCESS != errCL) {
    std::cout
        << "PointPillars::AnchorGrid enqueueWriteBuffer dev_anchors_dx_ fail: "
        << errCL << std::endl;
  }

  errCL = command_queue_->enqueueWriteBuffer(
      dev_anchors_dy_, true, 0, num_anchors_ * sizeof(float),
      static_cast<void *>(host_anchors_dy_), NULL, NULL);
  if (CL_SUCCESS != errCL) {
    std::cout
        << "PointPillars::AnchorGrid enqueueWriteBuffer dev_anchors_dy_ fail: "
        << errCL << std::endl;
  }

  errCL = command_queue_->enqueueWriteBuffer(
      dev_anchors_dz_, true, 0, num_anchors_ * sizeof(float),
      static_cast<void *>(host_anchors_dz_), NULL, NULL);
  if (CL_SUCCESS != errCL) {
    std::cout
        << "PointPillars::AnchorGrid enqueueWriteBuffer dev_anchors_dz_ fail: "
        << errCL << std::endl;
  }

  errCL = command_queue_->enqueueWriteBuffer(
      dev_anchors_ro_, true, 0, num_anchors_ * sizeof(float),
      static_cast<void *>(host_anchors_ro_), NULL, NULL);
  if (CL_SUCCESS != errCL) {
    std::cout
        << "PointPillars::AnchorGrid enqueueWriteBuffer dev_anchors_ro_ fail: "
        << errCL << std::endl;
  }

  errCL = command_queue_->enqueueWriteBuffer(
      dev_anchors_rad_, true, 0, mc_ * sizeof(float),
      static_cast<void *>(host_anchors_rad_), NULL, NULL);
  if (CL_SUCCESS != errCL) {
    std::cout
        << "PointPillars::AnchorGrid enqueueWriteBuffer dev_anchors_rad_ fail: "
        << errCL << std::endl;
  }

  ClearHostMemory();
}

void AnchorGrid::CreateAnchorMask(const cl::Buffer& dev_pillar_map, const int pillar_map_w,
                                  const int pillar_map_h,
                                  const float pillar_size_x,
                                  const float pillar_size_y,
                                  const cl::Buffer& dev_anchor_mask,
                                  const cl::Buffer& dev_pillar_workspace) {
  // Calculate the cumulative sum over the 2D grid dev_pillar_map in both X and
  // Y

  // Calculate an N greater than the current pillar map size enough to hold the
  // cummulative sum matrix
  const std::size_t n =
      NextPower(static_cast<std::size_t>(std::max(pillar_map_h, pillar_map_w)));

  // Calculate the cumulative sum
  ScanX(dev_pillar_workspace, dev_pillar_map, pillar_map_h, pillar_map_w, n);
  ScanY(dev_pillar_map, dev_pillar_workspace, pillar_map_h, pillar_map_w, n);

  // Mask anchors only where input data is found
  MaskAnchors(dev_anchors_px_, dev_anchors_py_, dev_pillar_map, dev_anchor_mask,
              dev_anchors_rad_, config_.min_x_range, config_.min_y_range,
              pillar_size_x, pillar_size_y, pillar_map_h, pillar_map_w, mc_,
              mr_, mh_, mw_);
}

void AnchorGrid::ScanX(const cl::Buffer& dev_input, 
                      const cl::Buffer& dev_output,
                      const int w, const int h, const int n){
  int idx = 0;
  std::string kernel_name{"ScanXKernel"};
  auto kernel = name_2_kernel_[kernel_name];
  cl::LocalSpaceArg local_mem_arg = cl::Local(sizeof(int) * n);
  kernel.setArg(idx++, dev_input);
  kernel.setArg(idx++, dev_output);
  kernel.setArg(idx++, n);
  kernel.setArg(idx++, local_mem_arg);
  auto global_ndrange = cl::NDRange(w * h / 2);
  auto local_ndrange = cl::NDRange(w / 2);
  errCL = command_queue_->enqueueNDRangeKernel(
    kernel, cl::NullRange, global_ndrange, local_ndrange, NULL, &eventCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "AnchorGrid::ScanX enqueueNDRangeKernel "
                 "ScanXKernel fail: "
              << errCL << std::endl;
  }
  eventCL.wait();
}

void AnchorGrid::ScanY(const cl::Buffer& dev_input, 
                      const cl::Buffer& dev_output,
                      const int w, const int h, const int n){
  int idx = 0;
  std::string kernel_name{"ScanYKernel"};
  auto kernel = name_2_kernel_[kernel_name];
  cl::LocalSpaceArg local_mem_arg = cl::Local(sizeof(int) * n);
  kernel.setArg(idx++, dev_input);
  kernel.setArg(idx++, dev_output);
  kernel.setArg(idx++, n);
  kernel.setArg(idx++, local_mem_arg);
  auto global_ndrange = cl::NDRange(w * h / 2);
  auto local_ndrange = cl::NDRange(h / 2);
  errCL = command_queue_->enqueueNDRangeKernel(
    kernel, cl::NullRange, global_ndrange, local_ndrange, NULL, &eventCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "AnchorGrid::ScanY enqueueNDRangeKernel "
                 "ScanYKernel fail: "
              << errCL << std::endl;
  }
  eventCL.wait();
}

void AnchorGrid::MaskAnchors(const cl::Buffer& dev_anchors_px, const cl::Buffer& dev_anchors_py,
                             const cl::Buffer& dev_pillar_map, const cl::Buffer& dev_anchor_mask,
                             const cl::Buffer& dev_anchors_rad,
                             const float min_x_range, const float min_y_range,
                             const float pillar_x_size,
                             const float pillar_y_size, const int grid_x_size,
                             const int grid_y_size, const int C, const int R,
                             const int H, const int W) {
  int idx = 0;
  std::string kernel_name{"MaskAnchorsSimpleKernel"};
  auto kernel = name_2_kernel_[kernel_name];
  int length = C * R * H * W;
  kernel.setArg(idx++, dev_anchors_px);
  kernel.setArg(idx++, dev_anchors_py);
  kernel.setArg(idx++, dev_pillar_map);
  kernel.setArg(idx++, dev_anchor_mask);
  kernel.setArg(idx++, dev_anchors_rad);
  kernel.setArg(idx++, min_x_range);
  kernel.setArg(idx++, min_y_range);
  kernel.setArg(idx++, pillar_x_size);
  kernel.setArg(idx++, pillar_y_size);
  kernel.setArg(idx++, grid_x_size);
  kernel.setArg(idx++, grid_y_size);
  kernel.setArg(idx++, C);
  kernel.setArg(idx++, R);
  kernel.setArg(idx++, H);
  kernel.setArg(idx++, W);
  kernel.setArg(idx++, length);
  int nums_opencl = DIVUP(length, threads_for_opencl_) * threads_for_opencl_;
  auto global_ndrange = cl::NDRange(nums_opencl);
  auto local_ndrange = cl::NDRange(threads_for_opencl_);
  errCL = command_queue_->enqueueNDRangeKernel(
      kernel, cl::NullRange, global_ndrange, local_ndrange, NULL, &eventCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "AnchorGrid::MaskAnchors enqueueNDRangeKernel "
                 "MaskAnchorsSimpleKernel fail: "
              << errCL << std::endl;
  }

  eventCL.wait();
}

bool AnchorGrid::OpenCLInit() {
  ocl_tool_.reset(new OCLTOOL(opencl_kernel_path_, program_names_,
                              program_2_kernel_names_, context_, device_,
                              build_option_str_));

  return true;
}

void AnchorGrid::MakeOCLKernel() {
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