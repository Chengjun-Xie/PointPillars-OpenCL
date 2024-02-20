# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

project ("pointpillars-opencl")

add_definitions(-DOCLDEBUG)

# include openvino
find_package(OpenVINO REQUIRED)

# include boost
find_package(Boost REQUIRED COMPONENTS system filesystem thread program_options)
include_directories(${Boost_INCLUDE_DIRS})

# include opencl
add_library(OPENCL_SO SHARED IMPORTED)
set_property(TARGET OPENCL_SO PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libs/libOpenCL.so.1)

# build library for execution
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*)
add_library(${PROJECT_NAME}_LIB STATIC ${SRC_FILES})
target_link_libraries(${PROJECT_NAME}_LIB
  openvino::runtime
  OPENCL_SO
)

target_include_directories(${PROJECT_NAME}_LIB PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} 
  ${PROJECT_NAME}_LIB
  Boost::filesystem
  Boost::program_options
)

configure_file(${CMAKE_SOURCE_DIR}/data/example.pcd ${CMAKE_CURRENT_BINARY_DIR}/example.pcd COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/model/pfe.onnx ${CMAKE_CURRENT_BINARY_DIR}/pfe.onnx COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/model/rpn.onnx ${CMAKE_CURRENT_BINARY_DIR}/rpn.onnx COPYONLY)


