#pragma once
#include <common.hpp>

// basic ops
#include <add2/add2.hpp>

// reduce
#include <reduce/block_reduce.hpp>

// softmax
#include <softmax/softmax.hpp>

// attn
#include <self_attn/self_attn.hpp>
#include <flash_attn/flash_attn.hpp>

// prefix sum
#include <prefix_sum/prefix_sum_baseline.hpp>
#include <prefix_sum/prefix_sum_v1.hpp>
#include <prefix_sum/prefix_sum_v2.hpp>
#include <prefix_sum/prefix_sum_v3.hpp>
#include <prefix_sum/prefix_sum_v4.hpp>

TORCH_LIBRARY(CudaLab, m) {
  TORCH_BINDING_COMMON_EXTENSION(add2_fp32_1d)
  TORCH_BINDING_COMMON_EXTENSION(reduce_fp32_1d)

  TORCH_BINDING_COMMON_EXTENSION(softmax_naive_fp32_1d)
  TORCH_BINDING_COMMON_EXTENSION(softmax_fp32_1d)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_fp32_1d)

  TORCH_BINDING_COMMON_EXTENSION(self_attn_fp32_v1)
  // TORCH_BINDING_COMMON_EXTENSION(flash_attn_fp32_v1)
  
  TORCH_BINDING_COMMON_EXTENSION(prefix_sum_baseline_fp32_1d)
  TORCH_BINDING_COMMON_EXTENSION(prefix_sum_v1_fp32_1d)
  TORCH_BINDING_COMMON_EXTENSION(prefix_sum_v2_fp32_1d)
  TORCH_BINDING_COMMON_EXTENSION(prefix_sum_v3_fp32_1d)
  TORCH_BINDING_COMMON_EXTENSION(prefix_sum_v4_fp32_1d)
}