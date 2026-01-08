#include <prefix_sum/prefix_sum_v1.hpp>

static __device__ __forceinline__ void ScanBlock(float *shm) {
  if(threadIdx.x == 0) {
    float sum = 0.0f;
    for(int i=0;i<blockDim.x;++i) {
      sum += shm[i]; // shm[i]此时为输入值
      shm[i] = sum; // shm[i]改为前缀和
    }
  }
  __syncthreads(); // 更改共享内存需要同步
}

__global__ void ScanAndWritePartSumKernelV1(
  const float *input, float *buffer, float *output, int N, int part_num
) {
  extern __shared__ float shm[];
  for(int part_i=blockIdx.x;part_i<part_num;part_i+=gridDim.x) {
    int idx = part_i * blockDim.x + threadIdx.x;
    shm[threadIdx.x] = idx < N ? input[idx] : 0.0f;
    __syncthreads();
    ScanBlock(shm);
    if(idx < N) {
      output[idx] = shm[threadIdx.x];
    }
    if(threadIdx.x == blockDim.x - 1) {
      buffer[part_i] = shm[threadIdx.x];
    }
  }
}

// 将block内结果转化为以block为单位的前缀和
__global__ void ScanPartSumKernelV1(float *buffer, int part_num) {
  float sum = 0.0f;
  for(int i=0;i<part_num;++i) {
    sum += buffer[i];
    buffer[i] = sum;
  }
}

// 每个结果使用前面的一个block的前缀和进行更新
__global__ void AddBaseSumKernelV1(float *buffer, float *output, int N, int part_num) {
  for(int part_i=blockIdx.x; part_i<part_num; part_i += gridDim.x) {
    if(part_i == 0) continue;
    int idx = part_i * blockDim.x + threadIdx.x;
    if(idx < N) {
      output[idx] += buffer[part_i - 1];
    } 
  }
}

TORCH_BINDING_PREFIX_SUM_V1_IMPL(fp32_1d, torch::kFloat32, float, 1)
