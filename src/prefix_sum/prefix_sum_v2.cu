#include <prefix_sum/prefix_sum_v2.hpp>

static __device__ __forceinline__ void ScanWarp(float * shm, int lane_id) {
  if(lane_id == 0) {
    float sum = 0.0f;
  #pragma unroll
    for(int i=0;i<WARP_SIZE;++i) {
      sum += shm[i];
      shm[i] = sum;
    }
  }
}

static __device__ __forceinline__ void ScanBlock(float *shm) {
  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;
  // 每个warp的前缀和,初始为每个warp内的值的和
  __shared__ float warp_sum[WARP_NUM];
  // 计算每个warp内的前缀和, lane_id == 0的那个线程做加法，更新整个warp的结果
  ScanWarp(&shm[threadIdx.x], lane_id);
  __syncthreads(); // 更改共享内存需要同步
  // warp内最后一个位置是这个warp内的和
  if(lane_id == 31) {
    warp_sum[warp_id] = shm[threadIdx.x];
  }
  __syncthreads();
  // warp_sum做前缀和
  if(warp_id == 0) {
    ScanWarp(warp_sum, lane_id);
  }
  __syncthreads();
  // 此时warp_sum为warp级别的前缀和
  // add base，在warp级别更新外围share_mem的前缀和
  if(warp_id > 0) {
    shm[threadIdx.x] += warp_sum[warp_id - 1];
  }
  __syncthreads(); //设备函数，更改共享内存，需要同步
}

__global__ void ScanAndWritePartSumKernelV2(
  const float *input, float *buffer, float *output, int N, int part_num
) {
  extern __shared__ float shm[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  shm[threadIdx.x] = idx < N ? input[idx] : 0.0f;
  __syncthreads();
  ScanBlock(shm);
  if(idx < N) {
    output[idx] = shm[threadIdx.x];
  }
  if(threadIdx.x == blockDim.x - 1) {
    buffer[blockIdx.x] = shm[threadIdx.x];
  }
}

// 将block内结果转化为以block为单位的前缀和
__global__ void ScanPartSumKernelV2(float *buffer, int part_num) {
  float sum = 0.0f;
  for(int i=0;i<part_num;++i) {
    sum += buffer[i];
    buffer[i] = sum;
  }
}

// 每个结果使用前面的一个block的前缀和进行更新
__global__ void AddBaseSumKernelV2(float *buffer, float *output, int N, int part_num) {
  if(blockIdx.x == 0) return;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N) {
    output[idx] += buffer[blockIdx.x - 1];
  }
}

TORCH_BINDING_PREFIX_SUM_V2_IMPL(fp32_1d, torch::kFloat32, float, 1)
