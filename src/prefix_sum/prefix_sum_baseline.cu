#include <prefix_sum/prefix_sum_baseline.hpp>

// 这里gridDim.x都是1
// block内单线程处理
// 计算block内prefix sum存储到output中
// 存储block的和到buffer中
__global__ void ScanAndWritePartSumKernel(
  const float *input, float *buffer, float *output, int N, int part_num
) {
  for(int part_i = blockIdx.x; part_i<part_num; part_i += gridDim.x) {
    int part_begin = part_i * blockDim.x;
    int part_end = min((part_i + 1) * blockDim.x, N);
    if(threadIdx.x == 0) {
      float sum = 0.0f;
      for(int i=part_begin;i<part_end;++i) {
        sum += input[i];
        output[i] = sum;
      }
      buffer[part_i] = sum;
    }
  }
}

// 将block内结果转化为以block为单位的前缀和
__global__ void ScanPartSumKernel(float *buffer, int part_num) {
  float sum = 0.0f;
  for(int i=0;i<part_num;++i) {
    sum += buffer[i];
    buffer[i] = sum;
  }
}

// 每个结果使用前面的一个block的前缀和进行更新
__global__ void AddBaseSumKernel(float *buffer, float *output, int N, int part_num) {
  for(int part_i=blockIdx.x; part_i<part_num; part_i += gridDim.x) {
    if(part_i == 0) continue;
    int idx = part_i * blockDim.x + threadIdx.x;
    if(idx < N) {
      output[idx] += buffer[part_i - 1];
    } 
  }
}

TORCH_BINDING_PREFIX_SUM_BASELINE_IMPL(fp32_1d, torch::kFloat32, float, 1)