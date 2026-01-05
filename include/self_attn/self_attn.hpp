#pragma once

#include <common.hpp>

#define TORCH_BINDING_SELF_ATTN_V1(tag, th_type, element_type, n_pack) \
  torch::Tensor self_attn_##tag(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

#define TORCH_BINDING_SELF_ATTN_V1_IMPL(tag, th_type, element_type, n_pack) \
torch::Tensor self_attn_##tag(torch::Tensor Q, torch::Tensor K, torch::Tensor V) { \
  CHECK_TORCH_TENSOR_CUDA(Q);                                           \
  CHECK_TORCH_TENSOR_CONTIGUOUS(Q);                                     \
  CHECK_TORCH_TENSOR_DTYPE(Q, th_type);                                 \
  CHECK_TORCH_TENSOR_CUDA(K);                                           \
  CHECK_TORCH_TENSOR_CONTIGUOUS(K);                                     \
  CHECK_TORCH_TENSOR_DTYPE(K, th_type);                                 \
  CHECK_TORCH_TENSOR_CUDA(V);                                           \
  CHECK_TORCH_TENSOR_CONTIGUOUS(V);                                     \
  CHECK_TORCH_TENSOR_DTYPE(V, th_type);                                 \
                                                                        \
  TORCH_CHECK(Q.dim() == 2);                                            \
  TORCH_CHECK(K.dim() == 2);                                            \
  TORCH_CHECK(V.dim() == 2);                                            \
                                                                        \
  int M = Q.size(0);                                                    \
  int d = Q.size(1);                                                    \
  int N = K.size(0);                                                    \
                                                                        \
  TORCH_CHECK(K.size(1) == d);                                          \
  TORCH_CHECK(V.size(0) == N);                                          \
  auto res = torch::zeros(                                              \
      {M, d},                                                           \
      torch::TensorOptions()                                            \
          .dtype(Q.dtype())                                             \
          .device(Q.device())                                           \
  );                                                                    \
  auto l = torch::zeros({M});                                           \
  auto m = torch::zeros({M});                                           \
                                                                        \
  static constexpr int Br = 4;                                          \
  static constexpr int Bc = 4;                                          \
                                                                        \
  dim3 threadsPerBlock(WARP_SIZE * WARP_SIZE, 1);                       \
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,   \
                      (M + threadsPerBlock.y - 1) / threadsPerBlock.y); \
                                                                        \
  self_attn_##tag##_kernel<<<blocksPerGrid, threadsPerBlock>>>(         \
    Q.data_ptr<element_type>(),                                         \
    K.data_ptr<element_type>(),                                         \
    V.data_ptr<element_type>(),                                         \
    res.data_ptr<element_type>(),                                       \
    M, N, d                                                             \
  );                                                                    \
  return res;                                                           \
}

TORCH_BINDING_SELF_ATTN(fp32_v1, torch::kFloat32, float, 1)