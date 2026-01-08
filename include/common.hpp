#pragma once

#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <cmath>

#define WARP_SIZE 32
#define BLOCK_SIZE 1024
#define WARP_NUM ((BLOCK_SIZE) + (WARP_SIZE) - 1) / WARP_SIZE

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                            \
  m.def(STRINGFY(func), &func);
  
#define TORCH_BINDING_COMMON_IMPL(func)                            \
  m.impl(STRINGFY(func), &(func));

// 检查是 CUDA tensor
#define CHECK_TORCH_TENSOR_CUDA(x)                                      \
  TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")

// 检查是 contiguous
#define CHECK_TORCH_TENSOR_CONTIGUOUS(x)                                \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

// 检查 dtype
#define CHECK_TORCH_TENSOR_DTYPE(x, dtype)                              \
  TORCH_CHECK((x).scalar_type() == (dtype),                             \
              #x " must have dtype " #dtype)
