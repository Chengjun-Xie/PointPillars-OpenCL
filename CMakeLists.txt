# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("pointpillars-opencl")

add_definitions(-DOCLDEBUG)

# include opencl
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

# include openvino
find_package(OpenVINO REQUIRED)
include_directories(${InferenceEngine_INCLUDE_DIRS})


# include opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# include boost
find_package(Boost REQUIRED COMPONENTS system filesystem thread program_options)
include_directories(${Boost_INCLUDE_DIRS})


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SOURCE_LIST)

add_executable(${PROJECT_NAME} main.cpp ${SOURCE_LIST})
target_link_libraries(${PROJECT_NAME} 
  ${OpenCL_LIBS} 
  ${InferenceEngine_LIBS} 
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  openvino::runtime
)

configure_file(${CMAKE_SOURCE_DIR}/data/example.pcd ${CMAKE_CURRENT_BINARY_DIR}/example.pcd COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/model/pfe.onnx ${CMAKE_CURRENT_BINARY_DIR}/pfe.onnx COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/model/rpn.onnx ${CMAKE_CURRENT_BINARY_DIR}/rpn.onnx COPYONLY)


