cmake_minimum_required(VERSION 3.16.3)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.9/bin/nvcc")
set(CMAKE_CXX_STANDARD 17)
set(TORCH_INSTALL_PREFIX "/mnt/disk4t/home/ljk/.local/lib/python3.8/site-packages/torch")
set(ENV{TORCH_CUDA_ARCH_LIST} "8.9")
set(CMAKE_CUDA_ARCHITECTURES 80-real;89-real)
set(Torch_DIR "/mnt/disk4t/home/ljk/.local/lib/python3.8/site-packages/torch/share/cmake/Torch")

enable_language(CUDA)
project(CudaLab LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_library(NVTOOLSEXT_LIBRARY
             NAMES nvToolsExt
             PATH_SUFFIXES lib)

if (NVTOOLSEXT_LIBRARY)
    message(STATUS "Found nvToolsExt library: ${NVTOOLSEXT_LIBRARY}")
else()
    message(FATAL_ERROR "Could not find nvToolsExt library")
endif()

add_library(CUDA::nvToolsExt SHARED IMPORTED)
set_target_properties(CUDA::nvToolsExt PROPERTIES
 IMPORTED_LOCATION ${NVTOOLSEXT_LIBRARY}
)

find_package(Torch CONFIG)
find_package(CUDA REQUIRED)

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS 
"${TORCH_INSTALL_PREFIX}/lib")

include_directories(/usr/include/python3.8)
include_directories(include)

file(GLOB_RECURSE SRCS CONFIGURE_DEPENDS 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_library(${PROJECT_NAME} SHARED ${SRCS})

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")

# 启用 CTest [也可以直接使用bash脚本]
enable_testing()

# 找 Python
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# 批量测试
add_test(
    NAME test_cuda_kernels
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_all.py
)

set_tests_properties(test_cuda_kernels PROPERTIES 
  ENVIRONMENT TORCH_CUDA_ARCH_LIST="8.9"  # rtx4090
  ENVIRONMENT CUDA_KERNEL_SRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}/src # source code dir
  ENVIRONMENT PYTHON_PATH=${CMAKE_SOURCE_DIR}/test
)