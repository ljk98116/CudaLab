#pragma once

#include <common.hpp>

enum BlockState {
  INVALID  = 0,
  PARTIAL  = 1,
  LOOKBACK = 2,
  COMPLETE = 3
};

struct BlockPrefix {
  float sum;   // 这个 block 的 prefix-sum 结果
  int   state; // 当前状态
};

// 仅使用一个核函数
#define TORCH_BINDING_PREFIX_SUM_V4(tag, th_type, element_type, n_pack)         \
  torch::Tensor prefix_sum_v4_##tag(torch::Tensor x);

#define TORCH_BINDING_PREFIX_SUM_V4_IMPL(tag, th_type, element_type, n_pack)    \
  torch::Tensor prefix_sum_v4_##tag(torch::Tensor x) {                          \
    CHECK_TORCH_TENSOR_CUDA(x);                                                 \
    CHECK_TORCH_TENSOR_CONTIGUOUS(x);                                           \
    CHECK_TORCH_TENSOR_DTYPE(x, th_type);                                       \
    const int64_t N = x.numel();                                                \
    int part_size = BLOCK_SIZE;                                                 \
    int part_num = (N + part_size - 1) / part_size;                             \
    int block_num = part_num;                                                   \
    auto res = torch::zeros({N}, torch::TensorOptions().dtype(x.dtype()).device(x.device())); \
    auto buffer = torch::zeros({block_num}, torch::TensorOptions().dtype(x.dtype()).device(x.device())); \
    BlockPrefix *g_block_prefix;                                                \
    cudaMalloc(&g_block_prefix, sizeof(BlockPrefix) * block_num);               \
    cudaMemset(g_block_prefix, 0, sizeof(BlockPrefix) * block_num);             \
    ScanAndWritePartSumKernelV4<<<block_num, part_size>>>(                      \
      x.data_ptr<element_type>(),                                               \
      g_block_prefix,                                                           \
      res.data_ptr<element_type>(),                                             \
      N                                                                         \
    );                                                                          \
    cudaFree(g_block_prefix);                                                   \
    return res;                                                                 \
  }

TORCH_BINDING_PREFIX_SUM_V4(fp32_1d, torch::kFloat32, float, 1)