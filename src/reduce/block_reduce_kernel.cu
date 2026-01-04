#include <reduce/block_reduce.hpp>

template <const int STRIDE=32>
static __device__ __forceinline__ double warp_reduce(double v) {
#pragma unroll
  for(int stride = STRIDE >> 1; stride > 0; stride >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, stride);
  }
  return v;
}

__global__ void reduce_fp32_1d_kernel(const float *input, float *output, int N) {
  int tid = threadIdx.x;
  int off = tid + blockIdx.x * blockDim.x;
  __shared__ double warp_sum[WARP_NUM];
  double v = 0.0;
  // ✅ grid-stride loop
  for (int i = off; i < N; i += gridDim.x * blockDim.x) {
    v += input[i];
  }
  __syncthreads();
  v = warp_reduce<WARP_SIZE>(v);
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  if(lane_id == 0) {
    warp_sum[warp_id] = v;
  }
  __syncthreads();
  v = lane_id < WARP_NUM ? warp_sum[lane_id] : 0.0f;
  if(warp_id == 0) {
    v = warp_reduce<WARP_NUM>(v);
  }
  if(tid == 0) {
    atomicAdd(output, v);
  }
}

TORCH_BINDING_IMPL_REDUCE_1D(fp32_1d, torch::kFloat32, float, 1)
