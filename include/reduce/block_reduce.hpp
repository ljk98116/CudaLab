#pragma once

#include <common.hpp>

#define TORCH_BINDING_REDUCE_1D(tag, th_type, element_type, n_pack) \
torch::Tensor reduce_##tag(torch::Tensor x);

#define TORCH_BINDING_IMPL_REDUCE_1D(tag, th_type, element_type, n_pack)  \
torch::Tensor reduce_##tag(torch::Tensor x) {                        \
  CHECK_TORCH_TENSOR_CUDA(x);                                        \
  CHECK_TORCH_TENSOR_CONTIGUOUS(x);                                  \
  CHECK_TORCH_TENSOR_DTYPE(x, th_type);                              \
                                                                     \
  const int64_t M = x.numel();                                       \
                                                                     \
  auto res = torch::zeros(                                           \
      {1},                                                           \
      torch::TensorOptions()                                         \
          .dtype(x.dtype())                                          \
          .device(x.device())                                        \
  );                                                                 \
                                                                     \
  constexpr int threadsPerBlock = BLOCK_SIZE;                        \
  const int elems_per_block = threadsPerBlock * n_pack;              \
  const int blocksPerGrid = (M + elems_per_block - 1) / elems_per_block; \
                                                                     \
  reduce_##tag##_kernel<<<blocksPerGrid, threadsPerBlock>>>(         \
      x.data_ptr<element_type>(),                                    \
      res.data_ptr<element_type>(),                                  \
      M                                                              \
  );                                                                 \
                                                                     \
  return res;                                                        \
}

TORCH_BINDING_REDUCE_1D(fp32_1d, torch::kFloat32, float, 1)