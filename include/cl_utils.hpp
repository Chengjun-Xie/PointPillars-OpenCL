#ifndef __CL_UTILS_HPP__
#define __CL_UTILS_HPP__

#include <string.h>

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "CL/cl2.hpp"
#include "pointpillars/common.hpp"

namespace pointpillars {
/** convert the kernel file into a string */
int GetSource(const char* filename, std::string& s);

class OCLTOOL {
 public:
  OCLTOOL(const std::string&, const std::vector<std::string>&,
          const std::map<std::string, std::vector<std::string>>&,
          const std::shared_ptr<cl::Context>&,
          const std::shared_ptr<cl::Device>&,
          const std::string build_option_str);

  void MakeKernelSource(std::map<std::string, std::string>&);
  void MakeProgramFromSource(const std::map<std::string, std::string>&,
                             std::map<std::string, cl::Program>&);
  void BuildProgramAndGetBinary(
      const std::map<std::string, cl::Program>&,
      std::map<std::string, std::vector<unsigned char>>&);
  void SaveProgramBinary(
      const std::map<std::string, std::vector<unsigned char>>&);
  void LoadBinaryAndMakeProgram(std::map<std::string, cl::Program>&);
  void MakeKernelFromBinary(std::map<std::string, cl::Program>&,
                            std::map<std::string, cl::Kernel>&);

 private:
  const std::vector<std::string> program_names_;
  const std::map<std::string, std::vector<std::string>> program_2_kernel_names_;

  const std::shared_ptr<cl::Context> context_;
  const std::shared_ptr<cl::Device> device_;

  std::string opencl_base_;
  const std::string build_option_str_;

  cl_int errCL;
};

}  // namespace pointpillars
#endif