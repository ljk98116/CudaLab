#include <softmax/softmax.hpp>

template <int STRIDE=32>
static __device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for(int stride=STRIDE >> 1; stride > 0; stride >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, stride));
  }
  return v;
}

template <int STRIDE=32>
static __device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for(int stride=STRIDE >> 1; stride > 0; stride >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, stride);
  }
  return v;
}

static __device__ float __softmax_block_reduce_max(const float *input, int N) {
  int tid = threadIdx.x;
  int off = tid + blockIdx.x * blockDim.x;
  // 计算block内最大值
  __shared__ float sData[WARP_NUM];
  float v = off < N ? input[off] : FLT_MIN;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  v = warp_reduce_max(v);
  if(lane_id == 0) {
    sData[warp_id] = v;
  }
  __syncthreads();
  v = lane_id < WARP_NUM ? sData[lane_id] : FLT_MIN;
  v = warp_reduce_max<WARP_NUM>(v);
  if(tid == 0) {
    sData[tid] = v;
  }
  __syncthreads();
  return sData[0];
}

static __device__ float __softmax_block_reduce_sum(const float *input, float maxval, int N) {
  int tid = threadIdx.x;
  int off = tid + blockIdx.x * blockDim.x;
  // 计算block内最大值
  __shared__ float sData[WARP_NUM];
  float v = off < N ? exp(input[off] - maxval) : 0;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  v = warp_reduce_sum(v);
  if(lane_id == 0) {
    sData[warp_id] = v;
  }
  __syncthreads();
  v = lane_id < WARP_NUM ? sData[lane_id] : 0;
  v = warp_reduce_sum<WARP_NUM>(v);
  if(tid == 0) {
    sData[tid] = v;
  }
  __syncthreads();
  return sData[0];
}

// 一行是一个线程
__global__ void softmax_naive_fp32_1d_kernel(const float *input, float *output, int N) {
  // 求行最大值
  float maxval = FLT_MIN;
  for(int i=0;i<N;++i) {
    maxval = fmaxf(maxval, input[i]);
  }
  float sumval = 0.0f;
  for(int i=0;i<N;++i) {
    sumval += exp(input[i] - maxval);
  }
  // 每个元素计算结果
  for(int i=0;i<N;++i) {
    output[i] = exp(input[i] - maxval) / sumval;
  }
}

__global__ void softmax_block_reduce_max(const float *input, float *output, int N) {
  float maxVal = __softmax_block_reduce_max(input, N);
  *output = maxVal;
}

__global__ void softmax_block_reduce_sum(const float *input, float *output, float * maxVal, int N) {
  float sumVal = __softmax_block_reduce_sum(input, *maxVal, N);
  *output = sumVal;
}

// 一行是一个block, 行元素数有限制，必须小于BLOCK_SIZE
__global__ void softmax_fp32_1d_kernel(const float *input, float *output, int N) {
  float maxval = __softmax_block_reduce_max(input, N);
  float expsum = __softmax_block_reduce_sum(input, maxval, N);
  int tid = threadIdx.x;
  int off = tid + blockIdx.x * blockDim.x;
  if(off < N) {
    output[off] = exp(input[off] - maxval) / expsum;
  }
}

// 大数组多次调用kernel
__global__ void safe_softmax_fp32_1d_kernel(
  const float *input, const float *maxVal, const float *sumVal, float *output, int N
) 
{
  int off = threadIdx.x + blockIdx.x * blockDim.x;
  if(off < N) {
    output[off] = exp(input[off] - *maxVal) / *sumVal;
  }
}

// to do: 大数组一行是一个block，但中间结果存储在全局内存中


TORCH_BINDING_NAIVE_SOFTMAX_1D_IMPL(fp32_1d, torch::kFloat32, float, 1)
TORCH_BINDING_SOFTMAX_1D_IMPL(fp32_1d, torch::kFloat32, float, 1)
TORCH_BINDING_SAFE_SOFTMAX_1D_IMPL(fp32_1d, torch::kFloat32, float, 1)