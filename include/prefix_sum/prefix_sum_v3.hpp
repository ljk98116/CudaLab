#pragma once

#include <common.hpp>

// block内使用共享内存
// block内使用reduce的方式求和
// warp内计算对共享内存访问使用zero padding方案,共享内存大小给成32x33x4字节,可以使得不同线程不会访问到共享内存内同一个32对齐的bank
// warp级计算并行化
#define TORCH_BINDING_PREFIX_SUM_V3(tag, th_type, element_type, n_pack)         \
  torch::Tensor prefix_sum_v3_##tag(torch::Tensor x);

#define TORCH_BINDING_PREFIX_SUM_V3_IMPL(tag, th_type, element_type, n_pack)    \
  torch::Tensor prefix_sum_v3_##tag(torch::Tensor x) {                          \
    CHECK_TORCH_TENSOR_CUDA(x);                                                 \
    CHECK_TORCH_TENSOR_CONTIGUOUS(x);                                           \
    CHECK_TORCH_TENSOR_DTYPE(x, th_type);                                       \
    const int64_t N = x.numel();                                                \
    int part_size = BLOCK_SIZE;                                                 \
    int part_num = (N + part_size - 1) / part_size;                             \
    int block_num = part_num;                                                   \
    auto res = torch::zeros({N}, torch::TensorOptions().dtype(x.dtype()).device(x.device())); \
    auto buffer = torch::zeros({block_num}, torch::TensorOptions().dtype(x.dtype()).device(x.device())); \
    size_t sram_size = sizeof(element_type) * (part_size + WARP_SIZE);          \
    ScanAndWritePartSumKernelV3<<<block_num, part_size, sram_size>>>(           \
      x.data_ptr<element_type>(),                                               \
      buffer.data_ptr<element_type>(),                                          \
      res.data_ptr<element_type>(),                                             \
      N                                                                         \
    );                                                                          \
    ScanPartSumKernelV3<<<1, 1>>>(buffer.data_ptr<element_type>(), part_num);   \
    AddBaseSumKernelV3<<<block_num, part_size>>>(                               \
      buffer.data_ptr<element_type>(), res.data_ptr<element_type>(), N          \
    );                                                                          \
    return res;                                                                 \
  }

TORCH_BINDING_PREFIX_SUM_V3(fp32_1d, torch::kFloat32, float, 1)