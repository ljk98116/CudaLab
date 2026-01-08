#include <prefix_sum/prefix_sum_v3.hpp>

static __device__ __forceinline__ float ScanWarp(float v) {
  int lane = threadIdx.x & 31;
  #pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    float y = __shfl_up_sync(0xffffffff, v, offset);
    // 必须保证y值是有意义的，否则取0
    if (lane >= offset) {
      v += y;
    }
  }
  return v;
}

static __device__ __forceinline__ void ScanBlock(float v) {
  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;
  // 每个warp的前缀和,初始为每个warp内的值的和
  __shared__ float warp_sum[WARP_NUM];
  // 暂时加载输入值到shm
  extern __shared__ float shm[];
  // 计算每个warp内的前缀和，此时warp内的前缀和都已经在输入位置就位
  v = ScanWarp(v);
  // warp内最后一个位置是这个warp内的和
  if(lane_id == 31) {
    warp_sum[warp_id] = v;
  }
  __syncthreads();

  // warp_sum做前缀和, 算出block内warp级别的前缀和数组,并填充回warp_sum
  float sum = lane_id < WARP_NUM ? warp_sum[lane_id] : 0.0f;
  if(warp_id == 0 && lane_id < WARP_NUM) {
    sum = ScanWarp(sum);
    warp_sum[lane_id] = sum; 
  }
  __syncthreads();

  // 所有warp的和存储在最后一个warp中
  // 此时warp_sum为warp级别的前缀和
  // add base，在warp级别更新外围share_mem的前缀和
  if(warp_id > 0) {
    v += warp_sum[warp_id - 1];
  }
  shm[threadIdx.x] = v;
  __syncthreads(); //设备函数，更改共享内存，需要同步
}

__global__ void ScanAndWritePartSumKernelV3(
  const float *input, float *buffer, float *output, int N
) {
  extern __shared__ float shm[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float v = idx < N ? input[idx] : 0.0f;
  ScanBlock(v);
  if(idx < N) {
    output[idx] = shm[threadIdx.x];
  }
  if(threadIdx.x == blockDim.x - 1) {
    buffer[blockIdx.x] = shm[threadIdx.x];
  }
}

// 将block内结果转化为以block为单位的前缀和
__global__ void ScanPartSumKernelV3(float *buffer, int part_num) {
  float sum = 0.0f;
  for(int i=0;i<part_num;++i) {
    sum += buffer[i];
    buffer[i] = sum;
  }
}

// 每个结果使用前面的一个block的前缀和进行更新
__global__ void AddBaseSumKernelV3(float *buffer, float *output, int N) {
  if(blockIdx.x == 0) return;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N) {
    output[idx] += buffer[blockIdx.x - 1];
  }
}

TORCH_BINDING_PREFIX_SUM_V3_IMPL(fp32_1d, torch::kFloat32, float, 1)
