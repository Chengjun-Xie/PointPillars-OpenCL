#include "cl_utils.hpp"

namespace pointpillars {

int GetSource(const char* filename, std::string& s) {
  size_t size;
  char* str;
  std::fstream f(filename, (std::fstream::in | std::fstream::binary));

  if (f.is_open()) {
    size_t fileSize;
    f.seekg(0, std::fstream::end);
    size = fileSize = (size_t)f.tellg();
    f.seekg(0, std::fstream::beg);

    // str = new char[size+1];
    str = new char[size];

    if (!str) {
      f.close();
      return 0;
    }

    f.read(str, fileSize);
    f.close();

    // str[size] = '\0';
    // s = str;
    s.assign(str, size);

    delete[] str;
    return 0;
  }
  std::cout << "Error: failed to open file\n" << filename << std::endl;
  return -1;
}

OCLTOOL::OCLTOOL(const std::string& opencl_kernel_path,
                 const std::vector<std::string>& program_names,
                 const std::map<std::string, std::vector<std::string>>&
                     program_2_kernel_names,
                 const std::shared_ptr<cl::Context>& context,
                 const std::shared_ptr<cl::Device>& device,
                 const std::string build_option_str)
    : program_names_{program_names},
      program_2_kernel_names_{program_2_kernel_names},
      context_{context},
      device_{device},
      build_option_str_{build_option_str} {
  opencl_base_ = opencl_kernel_path;
}

void OCLTOOL::MakeKernelSource(
    std::map<std::string, std::string>& name_2_source) {
  for (auto cl_item : program_names_) {
    const auto cl_path = opencl_base_ + cl_item;

    std::cout << "OCLTOOL::MakeKernelSource cl_path: " << cl_path << std::endl;
    std::string cl_source;
    GetSource(cl_path.c_str(), cl_source);
    name_2_source[cl_item] = cl_source;
  }
}

void OCLTOOL::MakeProgramFromSource(
    const std::map<std::string, std::string>& name_2_source,
    std::map<std::string, cl::Program>& name_2_program) {
  for (const auto cl_source : name_2_source) {
    auto it_name = cl_source.first;
    auto it_source = cl_source.second;

    // cl::Program::Sources sources;
    // sources.push_back(it_source);
    cl::Program program = cl::Program(*context_, it_source, false, &errCL);
    if (CL_SUCCESS != errCL) {
      std::cout << "PointPillars::MakeProgram fail: " << it_name
                << ", err:" << errCL << std::endl;
    }

    name_2_program[it_name] = std::move(program);
  }
}

void OCLTOOL::BuildProgramAndGetBinary(
    const std::map<std::string, cl::Program>& name_2_program,
    std::map<std::string, std::vector<unsigned char>>& name_2_binary) {
  for (auto cl_program : name_2_program) {
    auto it_name = cl_program.first;
    auto it_program = cl_program.second;

    errCL = it_program.build({*device_}, build_option_str_.c_str());
    if (CL_SUCCESS != errCL) {
      std::cout << "PointPillars::BuildProgramAndGetBinary fail: " << it_name
                << ", err:" << errCL << std::endl;

      if (it_program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(*device_) ==
          CL_BUILD_ERROR) {
        std::string build_log =
            it_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*device_);
        std::cout << build_log << std::endl;
      }
    }

    auto binariesVectors = it_program.getInfo<CL_PROGRAM_BINARIES>(&errCL);
    if (CL_SUCCESS != errCL) {
      std::cout << "BuildProgramAndGetBinary getInfo CL_PROGRAM_BINARIES fail: "
                << errCL << std::endl;
    }

    name_2_binary[it_name] = std::move(binariesVectors[0]);
  }
}

void OCLTOOL::SaveProgramBinary(
    const std::map<std::string, std::vector<unsigned char>>& name_2_binary) {
  for (auto cl_program_binary : name_2_binary) {
    auto it_name = cl_program_binary.first;
    auto cl_binary = cl_program_binary.second;

    auto pos = it_name.find(".cl");
    auto file_name_prefix = it_name.substr(0, pos);
    auto binary_file = file_name_prefix + std::string{".binary"};
    auto binary_file_path = opencl_base_ + binary_file;
    std::fstream f(binary_file_path.c_str(),
                   (std::fstream::out | std::fstream::binary));
    f.write(reinterpret_cast<char*>(cl_binary.data()), cl_binary.size());
    f.close();
  }
}

void OCLTOOL::LoadBinaryAndMakeProgram(
    std::map<std::string, cl::Program>& name_2_binary_program) {
  for (auto program_name : program_names_) {
    auto pos = program_name.find(".cl");
    auto file_name_prefix = program_name.substr(0, pos);
    auto binary_file = file_name_prefix + std::string{".binary"};
    auto binary_file_path = opencl_base_ + binary_file;

    std::string binary_source_str;
    GetSource(binary_file_path.c_str(), binary_source_str);

    std::vector<unsigned char> binary_vec{binary_source_str.begin(),
                                          binary_source_str.end()};
    cl::Program::Binaries binaries;
    binaries.push_back(binary_vec);
    auto binary_program =
        cl::Program(*context_, {*device_}, binaries, NULL, &errCL);
    if (CL_SUCCESS != errCL) {
      std::cout << "PointPillars::LoadBinaryAndMakeProgram binary fail: "
                << program_name << ", err:" << errCL << std::endl;
    }

    errCL = binary_program.build({*device_}, build_option_str_.c_str());
    if (CL_SUCCESS != errCL) {
      std::cout << "PointPillars::BuildProgramAndGetBinary fail: "
                << program_name;
      std::cout << ", err:" << errCL << std::endl;
      if (binary_program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(*device_) ==
          CL_BUILD_ERROR) {
        std::string build_log =
            binary_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*device_);
        std::cout << build_log << std::endl;
      }
    }

    name_2_binary_program[program_name] = binary_program;
  }
}

void OCLTOOL::MakeKernelFromBinary(
    std::map<std::string, cl::Program>& name_2_binary_program,
    std::map<std::string, cl::Kernel>& name_2_kernel) {
  for (auto item : program_2_kernel_names_) {
    auto program_name = item.first;
    auto kernel_vec = item.second;

    auto binary_program = name_2_binary_program[program_name];
    for (auto kernel_name : kernel_vec) {
      auto kernel = cl::Kernel(binary_program, kernel_name.c_str(), &errCL);
      if (CL_SUCCESS != errCL) {
        std::cout << "PointPillars::MakeKernelFromBinary fail: " << kernel_name
                  << " ,err: " << errCL << std::endl;
      }
      name_2_kernel[kernel_name] = kernel;
    }
  }
}

}  // namespace pointpillars
