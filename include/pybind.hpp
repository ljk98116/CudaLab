#pragma once
#include <common.hpp>

#include <add2/add2.hpp>
#include <reduce/block_reduce.hpp>
#include <softmax/softmax.hpp>
#include <self_attn/self_attn.hpp>

TORCH_LIBRARY(CudaLab, m) {
  TORCH_BINDING_COMMON_EXTENSION(add2_fp32_1d)
  TORCH_BINDING_COMMON_EXTENSION(reduce_fp32_1d)
  TORCH_BINDING_COMMON_EXTENSION(softmax_naive_fp32_1d)
  TORCH_BINDING_COMMON_EXTENSION(softmax_fp32_1d)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_fp32_1d)
  TORCH_BINDING_COMMON_EXTENSION(self_attn_fp32_v1)
}