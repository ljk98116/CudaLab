#pragma once
#include <common.hpp>

#include <add2/add2.hpp>
#include <reduce/block_reduce.hpp>

TORCH_LIBRARY(CudaLab, m) {
  TORCH_BINDING_COMMON_EXTENSION(add2_fp32_1d)
  TORCH_BINDING_COMMON_EXTENSION(reduce_fp32_1d)
}