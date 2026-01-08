// #pragma once

// #include <common.hpp>

// #define TORCH_BINDING_FLASH_ATTN(tag, th_type, element_type, n_pack) \
//   torch::Tensor flash_attn_##tag(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

// // check Bc == Br
// #define TORCH_BINDING_FLASH_ATTN_IMPL(tag, th_type, element_type, n_pack) \
//   torch::Tensor flash_attn_##tag(torch::Tensor Q, torch::Tensor K, torch::Tensor V, int nh) { \
//     const int Bc = 32;                                                                \
//     const int Br = 32;                                                                \
//                                                                                       \
//     const int B = Q.size(0);                                                          \
//     const int N = Q.size(1);                                                          \
//     const int d = Q.size(2);                                                          \
//                                                                                       \
//     CHECK_TORCH_TENSOR_CUDA(Q);                                                       \
//     CHECK_TORCH_TENSOR_CONTIGUOUS(Q);                                                 \
//     CHECK_TORCH_TENSOR_DTYPE(Q, th_type);                                             \
//     CHECK_TORCH_TENSOR_CUDA(K);                                                       \
//     CHECK_TORCH_TENSOR_CONTIGUOUS(K);                                                 \
//     CHECK_TORCH_TENSOR_DTYPE(K, th_type);                                             \
//     CHECK_TORCH_TENSOR_CUDA(V);                                                       \
//     CHECK_TORCH_TENSOR_CONTIGUOUS(V);                                                 \
//     CHECK_TORCH_TENSOR_DTYPE(V, th_type);                                             \
//                                                                                       \
//     TORCH_CHECK(Q.dim() == 3);                                                        \
//     TORCH_CHECK(K.dim() == 3);                                                        \
//     TORCH_CHECK(V.dim() == 3);                                                        \
//                                                                                       \
//     TORCH_CHECK(K.size(2) == d);                                                      \
//     TORCH_CHECK(K.size(1) == N);                                                      \
//     TORCH_CHECK(K.size(0) == B);                                                      \
//     TORCH_CHECK(V.size(2) == d);                                                      \
//     TORCH_CHECK(V.size(1) == N);                                                      \
//     TORCH_CHECK(V.size(0) == B);                                                      \
//                                                                                       \
//     TORCH_CHECK(Bc == Br);                                                            \
//     const int Tc = (N + Bc - 1) / Bc;                                                 \
//     const int Tr = (N + Br - 1) / Br;                                                 \
//     const float softmax_scale = 1.0f / std::sqrt((float)d);                           \
//                                                                                       \
//     auto O = torch::zeros_like(Q);                                                    \
//     auto l = torch::zeros({B, nh, N});                                                \
//     auto m = torch::full({B, nh, N}, -FLT_MAX);                                       \
//                                                                                       \
//     torch::Device device(torch::kCUDA);                                               \
//     l = l.to(device);                                                                 \
//     m = m.to(device);                                                                 \
//     const int sram_size = Br * d * sizeof(float) + (2 * Bc * d * sizeof(float)) +     \
//       (Bc * Br * sizeof(float));                                                      \
//                                                                                       \
//     dim3 grid_dim(B, nh);                                                             \
//     dim3 block_dim(Bc);                                                               \
//     flash_attn_##tag##_kernel<<<grid_dim, block_dim, sram_size>>>(                    \
//         Q.data_ptr<element_type>(), K.data_ptr<element_type>(), V.data_ptr<element_type>(), \
//         N, d, Tc, Tr, Bc, Br, softmax_scale,                                          \
//         l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()                 \
//     );                                                                                \
//     return O;                                                                         \
//   }

// TORCH_BINDING_FLASH_ATTN(fp32_v1, torch::kFloat32, float, 1)