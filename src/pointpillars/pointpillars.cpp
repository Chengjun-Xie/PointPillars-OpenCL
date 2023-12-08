#include "pointpillars/pointpillars.hpp"

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <thread>

static constexpr auto INTEL_PLATFORM_VENDOR = "Intel(R) Corporation";
static constexpr auto INTEL_GPU_PLATFORM = "OpenCL HD Graphics";

namespace pointpillars {
PointPillars::PointPillars(const float *score_threshold,
                           const float nms_threshold,
                           const PointPillarsConfig &config)
    : config_(config),
      score_threshold_(score_threshold),
      nms_overlap_threshold_(nms_threshold),
      pfe_model_file_(config.pfe_model_file),
      rpn_model_file_(config.rpn_model_file),
      max_num_pillars_(config.max_num_pillars),
      max_num_points_per_pillar_(config.max_num_points_per_pillar),
      pfe_output_size_(config.max_num_pillars * config.pillar_features),
      grid_x_size_(config.grid_x_size),
      grid_y_size_(config.grid_y_size),
      grid_z_size_(config.grid_z_size),
      rpn_input_size_(config.pillar_features * config.grid_x_size *
                      config.grid_y_size),
      num_cls_(config.num_classes),
      num_anchor_x_inds_(config.grid_x_size * config.rpn_scale),
      num_anchor_y_inds_(config.grid_y_size * config.rpn_scale),
      num_anchor_r_inds_(2),
      num_anchor_(num_anchor_x_inds_ * num_anchor_y_inds_ * num_anchor_r_inds_ *
                  num_cls_),
      rpn_box_output_size_(num_anchor_ * 7),         // feature score
      rpn_cls_output_size_(num_anchor_ * num_cls_),  // classification score
      rpn_dir_output_size_(num_anchor_ * 2),         // orientation score
      pillar_x_size_(config.pillar_x_size),
      pillar_y_size_(config.pillar_y_size),
      pillar_z_size_(config.pillar_z_size),
      min_x_range_(config.min_x_range),
      min_y_range_(config.min_y_range),
      min_z_range_(config.min_z_range),
      max_x_range_(config.max_x_range),
      max_y_range_(config.max_y_range),
      max_z_range_(config.max_x_range),
      batch_size_(1),
      num_features_(64),  // number of pillar features
      num_threads_(64),
      num_box_corners_(4),
      num_output_box_feature_(7),
      opencl_kernel_path_{config.opencl_kernel_path} {
  max_num_pillars_init_ = std::move(std::vector<int>(max_num_pillars_, 0));
  grid_size_init_ = std::move(std::vector<int>(grid_y_size_ * grid_x_size_, 0));
  num_anchor_init_ = std::move(std::vector<int>(num_anchor_, 0));

  OpenCLInit();
  MakeOCLKernel();
  InitComponents();
  DeviceMemoryMalloc();

  SetupPfeNetwork();
  SetupRpnNetwork(true);

#ifdef DEBUGLOCAL
  f_cost_time_ = std::fstream("f_cost_time.txt", (std::fstream::out));
#endif
}

PointPillars::~PointPillars() {
#ifdef DEBUGLOCAL
  f_cost_time_.close();
#endif
}

void PointPillars::InitComponents() {
  // Setup anchor grid
  AnchorGridConfig anchor_grid_config;
  anchor_grid_config.min_x_range = config.min_x_range;
  anchor_grid_config.max_x_range = config.max_x_range;
  anchor_grid_config.min_y_range = config.min_y_range;
  anchor_grid_config.max_y_range = config.max_y_range;
  anchor_grid_config.min_z_range = config.min_z_range;
  anchor_grid_config.max_z_range = config.max_z_range;

  anchor_grid_config.x_stride = config.x_stride;
  anchor_grid_config.y_stride = config.y_stride;
  anchor_grid_config.anchors = config.anchors;
  anchor_grid_config.rotations = {0.f, M_PI_2};
  anchor_grid_ptr_ =
      std::make_unique<AnchorGrid>(anchor_grid_config, opencl_kernel_path_,
                                   context_, command_queue_pre_, device_);
  anchor_grid_ptr_->GenerateAnchors();

  // Setup preprocessing
  preprocess_points_ptr_ = std::make_unique<PreProcess>(
      max_num_pillars_, max_num_points_per_pillar_, grid_x_size_, grid_y_size_,
      grid_z_size_, pillar_x_size_, pillar_y_size_, pillar_z_size_,
      min_x_range_, min_y_range_, min_z_range_, opencl_kernel_path_, context_,
      command_queue_pre_, device_);

  // Setup scatter
  scatter_ptr_ = std::make_unique<Scatter>(
      num_features_, max_num_pillars_, grid_x_size_, grid_y_size_,
      opencl_kernel_path_, context_, command_queue_infer_post_, device_);

  const float float_min = std::numeric_limits<float>::lowest();
  const float float_max = std::numeric_limits<float>::max();

  // Setup postprocessing
  postprocess_ptr_ = std::make_unique<PostProcess>(
      float_min, float_max, num_anchor_x_inds_, num_anchor_y_inds_,
      num_anchor_r_inds_, num_cls_, score_threshold_, num_threads_,
      nms_overlap_threshold_, num_box_corners_, num_output_box_feature_,
      opencl_kernel_path_, context_, command_queue_infer_post_, device_);
}

void PointPillars::SetupPfeNetwork() {
  // create map of network inputs to memory objects
  pfe_input_map_.insert({"pillar_x", dev_pillar_x_});
  pfe_input_map_.insert({"pillar_y", dev_pillar_y_});
  pfe_input_map_.insert({"pillar_z", dev_pillar_z_});
  pfe_input_map_.insert({"pillar_i", dev_pillar_i_});
  pfe_input_map_.insert({"num_points_per_pillar", dev_num_points_per_pillar_});
  pfe_input_map_.insert({"x_sub_shaped", dev_x_coors_for_sub_shaped_});
  pfe_input_map_.insert({"y_sub_shaped", dev_y_coors_for_sub_shaped_});
  pfe_input_map_.insert({"mask", dev_pillar_feature_mask_});
  std::string device = "GPU";

  // Setup InferenceEngine and load the network file
  InferenceEngine::Core inference_engine;
  inference_engine.SetConfig({{CONFIG_KEY(CACHE_DIR), config_.pfe_cache_path}},
                             device);
  auto network = inference_engine.ReadNetwork(pfe_model_file_);

  auto remote_context = InferenceEngine::gpu::make_shared_context(
      inference_engine, device, (*context_).get());

  // Setup network input configuration
  // The PillarFeatureExtraction (PFE) network has multiple inputs, which all
  // use FP32 precision
  for (auto &item : network.getInputsInfo()) {
    item.second->setPrecision(InferenceEngine::Precision::FP32);
    if (item.first == "num_points_per_pillar") {
      item.second->setLayout(InferenceEngine::Layout::NC);
    } else {
      item.second->setLayout(InferenceEngine::Layout::NCHW);
    }
  }

  // Setup network output configuration
  // The PillarFeatureExtraction (PFE) network has one output, which uses FP32
  // precision
  for (auto &item : network.getOutputsInfo()) {
    item.second->setPrecision(InferenceEngine::Precision::FP32);
    item.second->setLayout(InferenceEngine::Layout::NCHW);
  }

  // Finally load the network onto the execution device
  // pfe_exe_network_ = inference_engine.LoadNetwork(network, device);
  pfe_exe_network_ = inference_engine.LoadNetwork(network, remote_context);

  // Create InferenceEngine InferRequest for PFE
  pfe_infer_request_ptr_ = pfe_exe_network_.CreateInferRequestPtr();

  auto inputs = pfe_exe_network_.GetInputsInfo();
  for (auto input : inputs) {
    auto input_name = input.first;
    auto input_info = input.second;
    auto shared_buffer =
        static_cast<cl_mem>((pfe_input_map_[input_name]).get());
    auto shared_blob = InferenceEngine::gpu::make_shared_blob(
        input_info->getTensorDesc(), pfe_exe_network_.GetContext(),
        shared_buffer);
    pfe_infer_request_ptr_->SetBlob(input_name, shared_blob);
  }

  auto outputs = pfe_exe_network_.GetOutputsInfo();
  if (outputs.size() > 1) {
    std::cout << "PointPillars::SetupPfeNetwork: outputs size error!!!"
              << std::endl;
  }
  for (auto output : outputs) {
    auto output_name = output.first;
    auto output_info = output.second;
    auto shared_buffer = pfe_output_.get();
    auto shared_blob = InferenceEngine::gpu::make_shared_blob(
        output_info->getTensorDesc(), pfe_exe_network_.GetContext(),
        shared_buffer);
    pfe_infer_request_ptr_->SetBlob(output_name, shared_blob);
  }
}

void PointPillars::SetupRpnNetwork(bool resize_input) {
  rpn_output_map_.insert({"184", rpn_1_output_});  // box output
  rpn_output_map_.insert({"185", rpn_2_output_});  // classification output
  rpn_output_map_.insert({"187", rpn_3_output_});  // direction output

  std::string device = "GPU";

  // Setup InferenceEngine and load the network file
  InferenceEngine::Core inference_engine;
  inference_engine.SetConfig({{CONFIG_KEY(CACHE_DIR), config_.rpn_cache_path}},
                             device);
  auto network = inference_engine.ReadNetwork(rpn_model_file_);

  auto remote_context = InferenceEngine::gpu::make_shared_context(
      inference_engine, device, (*context_).get());

  // Setup network input configuration
  // The RegionProposalNetwork (RPN) network has one input, which uses FP32
  // precision
  for (auto &item : network.getInputsInfo()) {
    item.second->setPrecision(InferenceEngine::Precision::FP32);
    item.second->setLayout(InferenceEngine::Layout::NCHW);
  }

  // A resizing of the RPN input is possible and required depening on the pillar
  // and LiDAR configuration
  if (resize_input) {
    auto input_shapes = network.getInputShapes();
    std::string input_name;
    InferenceEngine::SizeVector input_shape;
    std::tie(input_name, input_shape) =
        *input_shapes.begin();  // only one input
    input_shape[0] = 1;         // set batch size to the first input dimension
    input_shape[1] =
        config_.pillar_features;  // set batch size to the first input dimension
    input_shape[2] =
        config_.grid_y_size;  // changes input height to the image one
    input_shape[3] =
        config_.grid_x_size;  // changes input width to the image one

    input_shapes[input_name] = input_shape;
    network.reshape(input_shapes);
  }

  // Setup network output configuration
  // The PillarFeatureExtraction (PFE) network has multiple outputs, which all
  // use FP32 precision
  for (auto &item : network.getOutputsInfo()) {
    item.second->setPrecision(InferenceEngine::Precision::FP32);
    item.second->setLayout(InferenceEngine::Layout::NCHW);
  }

  // Finally load the network onto the execution device
  // rpn_exe_network_ = inference_engine.LoadNetwork(network, device);
  rpn_exe_network_ = inference_engine.LoadNetwork(network, remote_context);

  // Create InferenceEngine InferRequest for RPN
  rpn_infer_request_ptr_ = rpn_exe_network_.CreateInferRequestPtr();

  auto inputs = rpn_exe_network_.GetInputsInfo();
  if (inputs.size() > 1) {
    std::cout << "PointPillars::SetupRpnNetwork: inputs size error!!!"
              << std::endl;
  }
  for (auto input : inputs) {
    auto input_name = input.first;
    auto input_info = input.second;
    auto shared_buffer = dev_scattered_feature_.get();
    auto shared_blob = InferenceEngine::gpu::make_shared_blob(
        input_info->getTensorDesc(), rpn_exe_network_.GetContext(),
        shared_buffer);
    rpn_infer_request_ptr_->SetBlob(input_name, shared_blob);
  }

  auto outputs = rpn_exe_network_.GetOutputsInfo();
  std::vector<std::pair<std::string, InferenceEngine::CDataPtr>> output_blobs;
  for (auto &output : outputs) {
    output_blobs.push_back(output);
    std::cout << "PointPillars::SetupRpnNetwork: output name: " << output.first
              << std::endl;
  }

  if (outputs.size() != 5) {
    std::cout << "PointPillars::SetupRpnNetwork: outputs size error!!!"
              << std::endl;
  }

  {
    auto output_name = output_blobs[0].first;
    auto output_info = output_blobs[0].second;
    // auto shared_buffer = rpn_1_output_.get();
    auto shared_buffer = rpn_output_map_[output_name].get();
    auto shared_blob = InferenceEngine::gpu::make_shared_blob(
        output_info->getTensorDesc(), rpn_exe_network_.GetContext(),
        shared_buffer);
    rpn_infer_request_ptr_->SetBlob(output_name, shared_blob);
  }

  {
    auto output_name = output_blobs[1].first;
    auto output_info = output_blobs[1].second;
    // auto shared_buffer = rpn_2_output_.get();
    auto shared_buffer = rpn_output_map_[output_name].get();
    auto shared_blob = InferenceEngine::gpu::make_shared_blob(
        output_info->getTensorDesc(), rpn_exe_network_.GetContext(),
        shared_buffer);
    rpn_infer_request_ptr_->SetBlob(output_name, shared_blob);
  }

  {
    auto output_name = output_blobs[2].first;
    auto output_info = output_blobs[2].second;
    // auto shared_buffer = rpn_3_output_.get();
    auto shared_buffer = rpn_output_map_[output_name].get();
    auto shared_blob = InferenceEngine::gpu::make_shared_blob(
        output_info->getTensorDesc(), rpn_exe_network_.GetContext(),
        shared_buffer);
    rpn_infer_request_ptr_->SetBlob(output_name, shared_blob);
  }
}

void PointPillars::DeviceMemoryMalloc() {
  cl_int errCL;
  // Allocate all device memory vector
  dev_x_coors_ = cl::Buffer(*context_, CL_MEM_READ_WRITE,
                            max_num_pillars_ * sizeof(int), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_x_coors_ buffer malloc fail: " << errCL << std::endl;
  }

  dev_y_coors_ = cl::Buffer(*context_, CL_MEM_READ_WRITE,
                            max_num_pillars_ * sizeof(int), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_y_coors_ buffer malloc fail: " << errCL << std::endl;
  }

  dev_num_points_per_pillar_ =
      cl::Buffer(*context_, CL_MEM_READ_WRITE, max_num_pillars_ * sizeof(float),
                 NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_num_points_per_pillar_ buffer malloc fail: " << errCL
              << std::endl;
  }

  dev_sparse_pillar_map_ =
      cl::Buffer(*context_, CL_MEM_READ_WRITE,
                 grid_x_size_ * grid_y_size_ * sizeof(int), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_sparse_pillar_map_ buffer malloc fail: " << errCL
              << std::endl;
  }

  dev_pillar_x_ =
      cl::Buffer(*context_, CL_MEM_READ_WRITE,
                 max_num_pillars_ * max_num_points_per_pillar_ * sizeof(float),
                 NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_pillar_x_ buffer malloc fail: " << errCL << std::endl;
  }

  dev_pillar_y_ =
      cl::Buffer(*context_, CL_MEM_READ_WRITE,
                 max_num_pillars_ * max_num_points_per_pillar_ * sizeof(float),
                 NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_pillar_y_ buffer malloc fail: " << errCL << std::endl;
  }

  dev_pillar_z_ =
      cl::Buffer(*context_, CL_MEM_READ_WRITE,
                 max_num_pillars_ * max_num_points_per_pillar_ * sizeof(float),
                 NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_pillar_z_ buffer malloc fail: " << errCL << std::endl;
  }

  dev_pillar_i_ =
      cl::Buffer(*context_, CL_MEM_READ_WRITE,
                 max_num_pillars_ * max_num_points_per_pillar_ * sizeof(float),
                 NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_pillar_i_ buffer malloc fail: " << errCL << std::endl;
  }

  dev_x_coors_for_sub_shaped_ =
      cl::Buffer(*context_, CL_MEM_READ_WRITE,
                 max_num_pillars_ * max_num_points_per_pillar_ * sizeof(float),
                 NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_x_coors_for_sub_shaped_ buffer malloc fail: " << errCL
              << std::endl;
  }

  dev_y_coors_for_sub_shaped_ =
      cl::Buffer(*context_, CL_MEM_READ_WRITE,
                 max_num_pillars_ * max_num_points_per_pillar_ * sizeof(float),
                 NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_y_coors_for_sub_shaped_ buffer malloc fail: " << errCL
              << std::endl;
  }

  dev_pillar_feature_mask_ =
      cl::Buffer(*context_, CL_MEM_READ_WRITE,
                 max_num_pillars_ * max_num_points_per_pillar_ * sizeof(float),
                 NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_pillar_feature_mask_ buffer malloc fail: " << errCL
              << std::endl;
  }

  // cumsum kernel
  dev_cumsum_workspace_ =
      cl::Buffer(*context_, CL_MEM_READ_WRITE,
                 grid_x_size_ * grid_y_size_ * sizeof(int), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_cumsum_workspace_ buffer malloc fail: " << errCL
              << std::endl;
  }

  // for make anchor mask kernel
  dev_anchor_mask_ = cl::Buffer(*context_, CL_MEM_READ_WRITE,
                                num_anchor_ * sizeof(int), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_anchor_mask_ buffer malloc fail: " << errCL << std::endl;
  }

  // for scatter kernel
  dev_scattered_feature_ =
      cl::Buffer(*context_, CL_MEM_READ_WRITE,
                 num_features_ * grid_y_size_ * grid_x_size_ * sizeof(float),
                 NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_scattered_feature_ buffer malloc fail: " << errCL
              << std::endl;
  }

  // for filter
  dev_filtered_box_ = cl::Buffer(
      *context_, CL_MEM_READ_WRITE,
      num_anchor_ * num_output_box_feature_ * sizeof(float), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_filtered_box_ buffer malloc fail: " << errCL << std::endl;
  }

  dev_filtered_score_ = cl::Buffer(*context_, CL_MEM_READ_WRITE,
                                   num_anchor_ * sizeof(float), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_filtered_score_ buffer malloc fail: " << errCL
              << std::endl;
  }

  dev_multiclass_score_ =
      cl::Buffer(*context_, CL_MEM_READ_WRITE,
                 num_anchor_ * num_cls_ * sizeof(float), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_multiclass_score_ buffer malloc fail: " << errCL
              << std::endl;
  }

  dev_filtered_dir_ = cl::Buffer(*context_, CL_MEM_READ_WRITE,
                                 num_anchor_ * sizeof(int), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_filtered_dir_ buffer malloc fail: " << errCL << std::endl;
  }

  dev_filtered_class_id_ = cl::Buffer(*context_, CL_MEM_READ_WRITE,
                                      num_anchor_ * sizeof(int), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_filtered_class_id_ buffer malloc fail: " << errCL
              << std::endl;
  }

  dev_box_for_nms_ =
      cl::Buffer(*context_, CL_MEM_READ_WRITE,
                 num_anchor_ * num_box_corners_ * sizeof(float), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_box_for_nms_ buffer malloc fail: " << errCL << std::endl;
  }

  dev_filter_count_ =
      cl::Buffer(*context_, CL_MEM_READ_WRITE, sizeof(int), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "dev_filter_count_ buffer malloc fail: " << errCL << std::endl;
  }

  // CNN outputs
  pfe_output_ = cl::Buffer(*context_, CL_MEM_READ_WRITE,
                           pfe_output_size_ * sizeof(float), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "pfe_output_ buffer malloc fail: " << errCL << std::endl;
  }

  rpn_1_output_ =
      cl::Buffer(*context_, CL_MEM_READ_WRITE,
                 rpn_box_output_size_ * sizeof(float), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "rpn_1_output_ buffer malloc fail: " << errCL << std::endl;
  }

  rpn_2_output_ =
      cl::Buffer(*context_, CL_MEM_READ_WRITE,
                 rpn_cls_output_size_ * sizeof(float), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "rpn_2_output_ buffer malloc fail: " << errCL << std::endl;
  }

  rpn_3_output_ =
      cl::Buffer(*context_, CL_MEM_READ_WRITE,
                 rpn_dir_output_size_ * sizeof(float), NULL, &errCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "rpn_3_output_ buffer malloc fail: " << errCL << std::endl;
  }
}

void PointPillars::PreProcessing(const float *in_points_array,
                                 const int in_num_points) {
  std::cout << "Starting PointPillars\n";
  std::cout << "   PreProcessing";

  const auto start_time = std::chrono::high_resolution_clock::now();

  cl_int errCL;
  cl::Event eventCL;
  int idx = 0;

  auto dev_points = cl::Buffer(*context_, CL_MEM_READ_WRITE,
                               in_num_points * num_box_corners_ * sizeof(float),
                               NULL, &errCL);
  errCL = command_queue_->enqueueWriteBuffer(
      dev_points, true, 0, in_num_points * num_box_corners_ * sizeof(float),
      static_cast<const void *>(in_points_array), NULL, NULL);
  if (CL_SUCCESS != errCL) {
    std::cout
        << "PointPillars::PreProcessing enqueueWriteBuffer dev_points fail: "
        << errCL << std::endl;
  }

  errCL = command_queue_->enqueueWriteBuffer(
      dev_sparse_pillar_map_, true, 0,
      grid_y_size_ * grid_x_size_ * sizeof(int),
      static_cast<void *>(grid_size_init_.data()), NULL, NULL);
  if (CL_SUCCESS != errCL) {
    std::cout << "PointPillars::PreProcessing enqueueWriteBuffer "
                 "dev_sparse_pillar_map_ fail: "
              << errCL << std::endl;
  }

  errCL = command_queue_->enqueueWriteBuffer(
      dev_x_coors_, true, 0, max_num_pillars_ * sizeof(int),
      static_cast<void *>(max_num_pillars_init_.data()), NULL, NULL);
  if (CL_SUCCESS != errCL) {
    std::cout
        << "PointPillars::PreProcessing enqueueWriteBuffer dev_x_coors_ fail: "
        << errCL << std::endl;
  }

  errCL = command_queue_->enqueueWriteBuffer(
      dev_y_coors_, true, 0, max_num_pillars_ * sizeof(int),
      static_cast<void *>(max_num_pillars_init_.data()), NULL, NULL);
  if (CL_SUCCESS != errCL) {
    std::cout
        << "PointPillars::PreProcessing enqueueWriteBuffer dev_y_coors_ fail: "
        << errCL << std::endl;
  }

  errCL = command_queue_->enqueueWriteBuffer(
      dev_num_points_per_pillar_, true, 0, max_num_pillars_ * sizeof(int),
      static_cast<void *>(max_num_pillars_init_.data()), NULL, NULL);
  if (CL_SUCCESS != errCL) {
    std::cout << "PointPillars::PreProcessing enqueueWriteBuffer "
                 "dev_num_points_per_pillar_ fail: "
              << errCL << std::endl;
  }

  errCL = command_queue_->enqueueWriteBuffer(
      dev_anchor_mask_, true, 0, num_anchor_ * sizeof(int),
      static_cast<void *>(num_anchor_init_.data()), NULL, NULL);
  if (CL_SUCCESS != errCL) {
    std::cout << "PointPillars::PreProcessing enqueueWriteBuffer "
                 "dev_anchor_mask_ fail: "
              << errCL << std::endl;
  }

  errCL = command_queue_->enqueueWriteBuffer(
      dev_cumsum_workspace_, true, 0, grid_y_size_ * grid_x_size_ * sizeof(int),
      static_cast<void *>(grid_size_init_.data()), NULL, NULL);
  if (CL_SUCCESS != errCL) {
    std::cout << "PointPillars::PreProcessing enqueueWriteBuffer "
                 "dev_cumsum_workspace_ fail: "
              << errCL << std::endl;
  }

  // // Run the PreProcessing operations and generate the input feature map

  size_t nums = max_num_pillars_ * max_num_points_per_pillar_;
  std::string kernel_name{"PointpillarMemset"};
  auto kernel = name_2_kernel_[kernel_name];
  kernel.setArg(idx++, dev_pillar_x_);
  kernel.setArg(idx++, dev_pillar_y_);
  kernel.setArg(idx++, dev_pillar_z_);
  kernel.setArg(idx++, dev_pillar_i_);
  kernel.setArg(idx++, dev_x_coors_for_sub_shaped_);
  kernel.setArg(idx++, dev_y_coors_for_sub_shaped_);
  kernel.setArg(idx++, dev_pillar_feature_mask_);
  kernel.setArg(idx++, nums);

  size_t nums_opencl = DIVUP(nums, threads_for_opencl_) * threads_for_opencl_;
  auto global_ndrange = cl::NDRange(nums_opencl);
  auto local_ndrange = cl::NDRange(threads_for_opencl_);
  command_queue_->enqueueNDRangeKernel(kernel, cl::NullRange, global_ndrange,
                                       local_ndrange, NULL, &eventCL);
  if (CL_SUCCESS != errCL) {
    std::cout << "PointPillars::PreProcessing enqueueNDRangeKernel "
                 "PointpillarMemset fail: "
              << errCL << std::endl;
  }

  eventCL.wait();

  // Run the PreProcessing operations and generate the input feature map
  preprocess_points_ptr_->DoPreProcess(
      dev_points, in_num_points, dev_x_coors_, dev_y_coors_,
      dev_num_points_per_pillar_, dev_pillar_x_, dev_pillar_y_, dev_pillar_z_,
      dev_pillar_i_, dev_x_coors_for_sub_shaped_, dev_y_coors_for_sub_shaped_,
      dev_pillar_feature_mask_, dev_sparse_pillar_map_, host_pillar_count_);

  const auto end_time = std::chrono::high_resolution_clock::now();
  const auto pre_cost_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                            start_time)
          .count();
  std::cout << " - " << pre_cost_time << "(microseconds)" << std::endl;
}

void PointPillars::PostProcessing(std::vector<ObjectDetection> &detections) {
  std::cout << "   Postprocessing";
  const auto start_time = std::chrono::high_resolution_clock::now();
  cl_int errCL;
  detections.clear();

  int host_counter_init = 0;
  errCL = command_queue_->enqueueWriteBuffer(
      dev_filter_count_, true, 0, sizeof(int),
      static_cast<void *>(&host_counter_init), NULL, NULL);
  if (CL_SUCCESS != errCL) {
    std::cout << "PointPillars::PostProcessing enqueueWriteBuffer "
                 "dev_filter_count_ fail: "
              << errCL << std::endl;
  }

  postprocess_ptr_->DoPostProcess(
      rpn_1_output_, rpn_2_output_, rpn_3_output_, dev_multiclass_score_,
      dev_filtered_box_, dev_filtered_score_, dev_filtered_class_id_,
      dev_box_for_nms_, dev_filter_count_, detections);

  const auto end_time = std::chrono::high_resolution_clock::now();
  const auto post_cost_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                            start_time)
          .count();

  std::cout << " - " << post_cost_time << "(microseconds)" << std::endl;
}

void PointPillars::PfeInfer() {
  std::cout << "   PFE Inference infer";

  const auto start_time = std::chrono::high_resolution_clock::now();

  pfe_infer_request_ptr_->StartAsync();

  // Wait for the inference to finish
  auto inference_result_status = pfe_infer_request_ptr_->Wait(
      InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
  if (InferenceEngine::OK != inference_result_status) {
    throw std::runtime_error("PFE Inference failed");
  }
  const auto end_time = std::chrono::high_resolution_clock::now();
  const auto pfe_cost_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                            start_time)
          .count();
  std::cout << " - " << pfe_cost_time << "(microseconds)" << std::endl;
}

void PointPillars::ScatterBack() {
  cl_int errCL;
  cl::Event eventCL;

  std::cout << "   Scattering";

  const auto start_time = std::chrono::high_resolution_clock::now();

  std::string kernel_name{"ScatterMemset"};
  auto kernel = name_2_kernel_[kernel_name];
  int idx = 0;
  kernel.setArg(idx++, dev_scattered_feature_);
  kernel.setArg(idx++, rpn_input_size_);
  int nums_opencl =
      DIVUP(rpn_input_size_, threads_for_opencl_) * threads_for_opencl_;
  auto global_ndrange = cl::NDRange(nums_opencl);
  auto local_ndrange = cl::NDRange(threads_for_opencl_);
  errCL = command_queue_->enqueueNDRangeKernel(
      kernel, cl::NullRange, global_ndrange, local_ndrange, NULL, &eventCL);
  if (CL_SUCCESS != errCL) {
    std::cout
        << "PointPillars::Detect enqueueNDRangeKernel ScatterMemset fail: "
        << errCL << std::endl;
  }

  eventCL.wait();

  scatter_ptr_->DoScatter(host_pillar_count_[0], dev_x_coors_, dev_y_coors_,
                          pfe_output_, dev_scattered_feature_);

  const auto end_time = std::chrono::high_resolution_clock::now();
  const auto scatter_cost_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                            start_time)
          .count();
  std::cout << " - " << scatter_cost_time << "(microseconds)" << std::endl;
}

void PointPillars::RpnInfer() {
  std::cout << "   RPN Inference infer";

  const auto start_time = std::chrono::high_resolution_clock::now();

  rpn_infer_request_ptr_->StartAsync();
  auto inference_result_status = rpn_infer_request_ptr_->Wait(
      InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
  if (InferenceEngine::OK != inference_result_status) {
    throw std::runtime_error("RPN Inference failed");
  }

  const auto end_time = std::chrono::high_resolution_clock::now();
  const auto rpn_cost_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                            start_time)
          .count();

  std::cout << " - " << rpn_cost_time << "(microseconds)" << std::endl;
}

bool PointPillars::OpenCLInit() {
  cl_int res;

  std::vector<cl::Platform> platforms;
  res = cl::Platform::get(&platforms);
  if (platforms.empty() || res != CL_SUCCESS) {
    std::cout << "OpenCLInit: get platform fail" << std::endl;

    return false;
  }

  cl::Platform platform_intel;
  bool find_intel_gpu = false;
  for (auto item : platforms) {
    if (item.getInfo<CL_PLATFORM_VENDOR>() != INTEL_PLATFORM_VENDOR) {
      continue;
    }

    auto item_name = item.getInfo<CL_PLATFORM_NAME>();

    if (item_name.find(INTEL_GPU_PLATFORM) != std::string::npos) {
      platform_intel = item;
      find_intel_gpu = true;
      break;
    }
  }

  if (!find_intel_gpu) {
    std::cout << "PointPillars::OpenCLInit not found intel gpu" << std::endl;

    return find_intel_gpu;
  }

  cl::Platform::setDefault(platform_intel);

  std::vector<cl::Device> all_devices;
  res = platform_intel.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
  if (all_devices.empty() || res != CL_SUCCESS) {
    std::cout << "OpenCLInit: get device fail: " << res << std::endl;

    return false;
  }

  device_ = std::make_shared<cl::Device>(all_devices[0]);
  context_ =
      std::make_shared<cl::Context>(*device_, nullptr, nullptr, nullptr, &res);
  command_queue_ =
      std::make_shared<cl::CommandQueue>(*context_, *device_, 0, &res);
  command_queue_pre_ =
      std::make_shared<cl::CommandQueue>(*context_, *device_, 0, &res);
  if (CL_SUCCESS != res) {
    std::cout << "OpenCLInit command_queue_pre_ create fail" << res
              << std::endl;
  }
  command_queue_infer_post_ =
      std::make_shared<cl::CommandQueue>(*context_, *device_, 0, &res);
  if (CL_SUCCESS != res) {
    std::cout << "OpenCLInit command_queue_infer_post_ create fail" << res
              << std::endl;
  }
  const std::string device_name = device_->getInfo<CL_DEVICE_NAME>();
  const std::string vendor_name = device_->getInfo<CL_DEVICE_VENDOR>();
  device_->getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                   &gpu_global_memory_cache_size_);
  device_->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &gpu_compute_unit_);
  device_->getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &gpu_max_frequent_);

  std::cout << "device name: " << device_name.c_str()
            << " , device vendor name: " << vendor_name.c_str() << std::endl;
  std::cout << "CL_DEVICE_MAX_CLOCK_FREQUENCY: " << gpu_max_frequent_
            << std::endl;

  ocl_tool_.reset(new OCLTOOL(opencl_kernel_path_, program_names_,
                              program_2_kernel_names_, context_, device_,
                              build_option_str_));

  return true;
}

void PointPillars::CreateAnchorMask() {
  // 2nd step is to create the anchor mask used to optimize the decoding of the
  // RegionProposalNetwork output
  std::cout << "   AnchorMask";
  const auto start_time = std::chrono::high_resolution_clock::now();
  anchor_grid_ptr_->CreateAnchorMask(
      dev_sparse_pillar_map_, grid_y_size_, grid_x_size_, pillar_x_size_,
      pillar_y_size_, dev_anchor_mask_, dev_cumsum_workspace_);
  const auto end_time = std::chrono::high_resolution_clock::now();
  const auto create_anchor_cost_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                            start_time)
          .count();
  std::cout << " - " << create_anchor_cost_time << "(microseconds)"
            << std::endl;
}

void PointPillars::MakeOCLKernel() {
#ifdef OCLDEBUG
  ocl_tool_->MakeKernelSource(name_2_source_);
  ocl_tool_->MakeProgramFromSource(name_2_source_, name_2_program_);
  ocl_tool_->BuildProgramAndGetBinary(name_2_program_, name_2_binary_);
  ocl_tool_->SaveProgramBinary(name_2_binary_);
#endif
  ocl_tool_->LoadBinaryAndMakeProgram(name_2_binary_program_);
  ocl_tool_->MakeKernelFromBinary(name_2_binary_program_, name_2_kernel_);
}

void PointPillars::Detect(const float *in_points_array, const int in_num_points,
                          std::vector<ObjectDetection> &detections) {
  PreProcessing(in_points_array, in_num_points);
  CreateAnchorMask();
  PfeInfer();
  ScatterBack();
  RpnInfer();
  PostProcessing(detections);
}

}  // namespace pointpillars
