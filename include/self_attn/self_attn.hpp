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
          .dtype(torch::kFloat32)                                       \
          .device(Q.device())                                           \
  );                                                                    \
  auto l = torch::zeros({M}, torch::TensorOptions().dtype(torch::kFloat32).device(Q.device())); \
  auto m = torch::full(                                                 \
    {M},                                                                \
    -FLT_MAX,                                                           \
    torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));  \
                                                                        \
  static constexpr int Br = 32;                                         \
  static constexpr int Bc = 1;                                          \
                                                                        \
  dim3 threadsPerBlock(Br, 1);                                          \
  dim3 blocksPerGrid((M + Br - 1) / Br, 1);                             \
                                                                        \
  size_t sMem = (((Bc << 1) + Br) * d + Bc * Br) * sizeof(float);       \
  self_attn_##tag##_kernel<Bc, Br><<<blocksPerGrid, threadsPerBlock, sMem>>>( \
    Q.data_ptr<element_type>(),                                         \
    K.data_ptr<element_type>(),                                         \
    V.data_ptr<element_type>(),                                         \
    res.data_ptr<element_type>(),                                       \
    M, N, d,                                                            \
    l.data_ptr<float>(),                                                \
    m.data_ptr<float>()                                                 \
  );                                                                    \
  return res;                                                           \
}

TORCH_BINDING_SELF_ATTN_V1(fp32_v1, torch::kFloat32, float, 1)