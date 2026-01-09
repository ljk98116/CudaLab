#pragma once

#include <common.hpp>

enum BlockState {
  INVALID  = 0,
  PARTIAL  = 1,
  COMPLETE = 2
};

// struct BlockPrefix {
//   float aggregate;   // 这个 block 的 prefix-sum 结果
//   float inclusive_prefix;
//   int   state; // 当前状态
// };

// 可以使用union的方式一次写入全局值+状态，避免全局memory fence的使用
// 使用全局计数器记录block的激活顺序，必须找前缀已经运行过的block进行更新，防止死锁[比较难理解]
// 使用torch workspace + cudastream优化核函数执行过程
union BlockPrefix {
  struct {
    float sum;
    int state;
  };
  uint64_t packed;
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
    int64_t workspace_bytes =                                                   \
        static_cast<int64_t>(block_num) * static_cast<int64_t>(sizeof(BlockPrefix)) \
      + static_cast<int64_t>(sizeof(int));                                      \
                                                                                \
    auto workspace = torch::empty(                                              \
      {workspace_bytes},                                                        \
      torch::TensorOptions()                                                    \
        .device(x.device())                                                     \
        .dtype(torch::kUInt8));                                                 \
                                                                                \
    BlockPrefix* g_block_prefix =                                               \
      reinterpret_cast<BlockPrefix*>(workspace.data_ptr<uint8_t>());            \
                                                                                \
    int* g_vp_counter =                                                         \
      reinterpret_cast<int*>(                                                   \
        workspace.data_ptr<uint8_t>() + block_num * sizeof(BlockPrefix));       \
                                                                                \
    cudaMemsetAsync(g_vp_counter, 0, sizeof(int), at::cuda::getCurrentCUDAStream());  \
    cudaMemsetAsync(                                                            \
      g_block_prefix, 0,                                                        \
      sizeof(BlockPrefix) * block_num,                                          \
      at::cuda::getCurrentCUDAStream());                                        \
    ScanAndWritePartSumKernelV4<<<block_num, part_size, 0, at::cuda::getCurrentCUDAStream()>>>( \
      x.data_ptr<element_type>(),                                               \
      g_block_prefix,                                                           \
      res.data_ptr<element_type>(),                                             \
      N,                                                                        \
      g_vp_counter                                                              \
    );                                                                          \
    return res;                                                                 \
  }

TORCH_BINDING_PREFIX_SUM_V4(fp32_1d, torch::kFloat32, float, 1)