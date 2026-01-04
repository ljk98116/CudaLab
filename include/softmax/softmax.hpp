#pragma once

#include <common.hpp>

#define TORCH_BINDING_NAIVE_SOFTMAX_1D(tag, th_type, element_type, n_pack) \
  torch::Tensor softmax_naive_##tag(torch::Tensor x);

#define TORCH_BINDING_NAIVE_SOFTMAX_1D_IMPL(tag, th_type, element_type, n_pack) \
torch::Tensor softmax_naive_##tag(torch::Tensor x) {                          \
  CHECK_TORCH_TENSOR_CUDA(x);                                           \
  CHECK_TORCH_TENSOR_CONTIGUOUS(x);                                     \
  CHECK_TORCH_TENSOR_DTYPE(x, th_type);                                 \
                                                                        \
  const int64_t M = x.numel();                                          \
  auto res = torch::zeros_like(x);                                      \
                                                                        \
  constexpr int threadsPerBlock = 1;                           \
  const int elems_per_block = threadsPerBlock * n_pack;                 \
  const int blocksPerGrid = (M + elems_per_block - 1) / elems_per_block;\
                                                                        \
  softmax_naive_##tag##_kernel<<<blocksPerGrid, threadsPerBlock>>>(     \
    x.data_ptr<element_type>(),                                         \
    res.data_ptr<element_type>(),                                       \
    M                                                                   \
  );                                                                    \
                                                                        \
  return res;                                                           \
}

#define TORCH_BINDING_SOFTMAX_1D(tag, th_type, element_type, n_pack) \
  torch::Tensor softmax_##tag(torch::Tensor x);

#define TORCH_BINDING_SOFTMAX_1D_IMPL(tag, th_type, element_type, n_pack) \
torch::Tensor softmax_##tag(torch::Tensor x) {                          \
  CHECK_TORCH_TENSOR_CUDA(x);                                           \
  CHECK_TORCH_TENSOR_CONTIGUOUS(x);                                     \
  CHECK_TORCH_TENSOR_DTYPE(x, th_type);                                 \
                                                                        \
  const int64_t M = x.numel();                                          \
  auto res = torch::zeros_like(x);                                      \
                                                                        \
  constexpr int threadsPerBlock = BLOCK_SIZE;                           \
  const int elems_per_block = threadsPerBlock * n_pack;                 \
  const int blocksPerGrid = (M + elems_per_block - 1) / elems_per_block;\
                                                                        \
  softmax_##tag##_kernel<<<blocksPerGrid, threadsPerBlock>>>(           \
    x.data_ptr<element_type>(),                                         \
    res.data_ptr<element_type>(),                                       \
    M                                                                   \
  );                                                                    \
                                                                        \
  return res;                                                           \
}

#define TORCH_BINDING_SAFE_SOFTMAX_1D(tag, th_type, element_type, n_pack) \
  torch::Tensor safe_softmax_##tag(torch::Tensor x);

#define TORCH_BINDING_SAFE_SOFTMAX_1D_IMPL(tag, th_type, element_type, n_pack) \
torch::Tensor safe_softmax_##tag(torch::Tensor x) {                          \
  CHECK_TORCH_TENSOR_CUDA(x);                                           \
  CHECK_TORCH_TENSOR_CONTIGUOUS(x);                                     \
  CHECK_TORCH_TENSOR_DTYPE(x, th_type);                                 \
                                                                        \
  const int64_t M = x.numel();                                          \
  auto res = torch::zeros_like(x);                                      \
                                                                        \
  constexpr int threadsPerBlock = BLOCK_SIZE;                           \
  const int elems_per_block = threadsPerBlock * n_pack;                 \
  const int blocksPerGrid = (M + elems_per_block - 1) / elems_per_block;\
                                                                        \
  float *sumVal, *maxVal;                                               \
  cudaMalloc(&sumVal, sizeof(float));                                   \
  cudaMalloc(&maxVal, sizeof(float));                                   \
  cudaMemset(sumVal, 0.0f, sizeof(float));                              \
  cudaMemset(maxVal, FLT_MIN, sizeof(float));                           \
  softmax_block_reduce_max<<<blocksPerGrid, elems_per_block>>>(         \
    x.data_ptr<element_type>(),                                         \
    maxVal,                                                             \
    M                                                                   \
  );                                                                    \
                                                                        \
  softmax_block_reduce_sum<<<blocksPerGrid, elems_per_block>>>(         \
    x.data_ptr<element_type>(),                                         \
    sumVal,                                                             \
    maxVal,                                                             \
    M                                                                   \
  );                                                                    \
                                                                        \
  safe_softmax_##tag##_kernel<<<blocksPerGrid, elems_per_block>>>(      \
    x.data_ptr<element_type>(),                                         \
    maxVal,                                                             \
    sumVal,                                                             \
    res.data_ptr<element_type>(),                                       \
    M                                                                   \
  );                                                                    \
  cudaFree(sumVal);                                                     \
  cudaFree(maxVal);                                                     \
  cudaDeviceSynchronize();                                              \
                                                                        \
  return res;                                                           \
}

TORCH_BINDING_NAIVE_SOFTMAX_1D(fp32_1d, torch::kFloat32, float, 1)
TORCH_BINDING_SOFTMAX_1D(fp32_1d, torch::kFloat32, float, 1)
TORCH_BINDING_SAFE_SOFTMAX_1D(fp32_1d, torch::kFloat32, float, 1)