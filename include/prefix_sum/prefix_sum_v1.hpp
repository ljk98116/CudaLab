#pragma once

#include <common.hpp>

// block内使用共享内存
#define TORCH_BINDING_PREFIX_SUM_V1(tag, th_type, element_type, n_pack) \
  torch::Tensor prefix_sum_v1_##tag(torch::Tensor x);

#define TORCH_BINDING_PREFIX_SUM_V1_IMPL(tag, th_type, element_type, n_pack) \
  torch::Tensor prefix_sum_v1_##tag(torch::Tensor x) {                       \
    CHECK_TORCH_TENSOR_CUDA(x);                                                 \
    CHECK_TORCH_TENSOR_CONTIGUOUS(x);                                           \
    CHECK_TORCH_TENSOR_DTYPE(x, th_type);                                       \
    const int64_t N = x.numel();                                                \
    int part_size = BLOCK_SIZE;                                                 \
    int part_num = (N + part_size - 1) / part_size;                             \
    int block_num = part_num;                                                   \
    auto res = torch::zeros({N}, torch::TensorOptions().dtype(x.dtype()).device(x.device())); \
    auto buffer = torch::zeros({block_num}, torch::TensorOptions().dtype(x.dtype()).device(x.device())); \
    size_t sram_size = sizeof(element_type) * part_size;                        \
    ScanAndWritePartSumKernelV1<<<block_num, part_size, sram_size>>>(           \
      x.data_ptr<element_type>(),                                               \
      buffer.data_ptr<element_type>(),                                          \
      res.data_ptr<element_type>(),                                             \
      N, part_num                                                               \
    );                                                                          \
    ScanPartSumKernelV1<<<1, 1>>>(buffer.data_ptr<element_type>(), part_num);   \
    AddBaseSumKernelV1<<<block_num, part_size>>>(                               \
      buffer.data_ptr<element_type>(), res.data_ptr<element_type>(), N, part_num\
    );                                                                          \
    return res;                                                                 \
  }

TORCH_BINDING_PREFIX_SUM_V1(fp32_1d, torch::kFloat32, float, 1)