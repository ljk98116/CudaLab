#pragma once

#include <common.hpp>

#define TORCH_BINDING_ADD2_1D(tag, th_type, element_type, n_pack) \
  torch::Tensor add2_##tag(torch::Tensor x, torch::Tensor y);

#define TORCH_BINDING_IMPL_ADD2_1D(tag, th_type, element_type, n_pack)     \
torch::Tensor add2_##tag(torch::Tensor x, torch::Tensor y) {        \
  CHECK_TORCH_TENSOR_CUDA(x);                                        \
  CHECK_TORCH_TENSOR_CUDA(y);                                        \
  CHECK_TORCH_TENSOR_CONTIGUOUS(x);                                  \
  CHECK_TORCH_TENSOR_CONTIGUOUS(y);                                  \
  CHECK_TORCH_TENSOR_DTYPE(x, th_type);                              \
  CHECK_TORCH_TENSOR_DTYPE(y, th_type);                              \
                                                                     \
  const int64_t M = x.numel();                                       \
  TORCH_CHECK(y.numel() == M, "x/y size mismatch");                  \
                                                                     \
  auto res = torch::zeros_like(x);                                   \
                                                                     \
  constexpr int threadsPerBlock = BLOCK_SIZE;                        \
  const int elems_per_block = threadsPerBlock * n_pack;             \
  const int blocksPerGrid = (M + elems_per_block - 1) / elems_per_block; \
                                                                     \
  add2_##tag##_kernel<<<blocksPerGrid, threadsPerBlock>>>( \
      x.data_ptr<element_type>(),                                    \
      y.data_ptr<element_type>(),                                    \
      res.data_ptr<element_type>(),                                  \
      M                                                              \
  );                                                                 \
                                                                     \
  return res;                                                        \
}

TORCH_BINDING_ADD2_1D(fp32_1d, torch::kFloat32, float, 1)




