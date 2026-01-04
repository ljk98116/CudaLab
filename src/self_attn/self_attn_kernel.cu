#include <self_attn/self_attn.hpp>

static __device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for(int stride = WARP_SIZE >> 1; stride > 0; stride >>= 1) {
        v = fmax(v, __shfl_xor_sync(0xffffffff, v, stride));
    }
    return v;
}

static __device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for(int stride = WARP_SIZE >> 1; stride > 0; stride >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, stride);
    }
    return v;
}

static __device__ float block_reduce_max_Y(float v, int N, float *gBlkData) {
  if(N <= BLOCK_SIZE) {
    __shared__ float sData[WARP_NUM];
    int tidY = threadIdx.x;
    int warp_id = tidY / WARP_SIZE;
    int lane_id = tidY % WARP_SIZE;
    v = warp_reduce_max(v);
    if(lane_id == 0) {
        sData[warp_id] = v;
    }
    __syncthreads();
    v = lane_id < WARP_NUM ? sData[lane_id] : FLT_MIN;
    if(warp_id == 0) {
      v = warp_reduce_max(v);
    }
    if(tidY == 0) {
      sData[tidY] = v;
    }
    __syncthreads();
    return sData[0];
  }
  int tidY = threadIdx.x;
  int offX = threadIdx.y + blockDim.y * blockIdx.y;
  
  const int warp_num = (N + WARP_SIZE - 1) / WARP_SIZE;
  const int off = offX * warp_num;

  int warp_id = tidY / WARP_SIZE;
  int lane_id = tidY % WARP_SIZE;

  v = warp_reduce_max(v);
  // 全局store
  if(lane_id == 0) {
    gBlkData[off + warp_id] = v;
  }
  // 全局Load
  // 不同warp的lane_id=0都是v
  for(int stride = warp_num >> 1; stride > 0; stride >>= 1) {
    gBlkData[off + warp_id] = fmaxf(gBlkData[off + warp_id], gBlkData[off + warp_id + stride]);
  }
  return gBlkData[off];    
}

static __device__ float block_reduce_sum_Y(float v, float maxVal, int N, float *gBlkData) {
  v = exp(v - maxVal);
  if(N <= BLOCK_SIZE) {
    __shared__ float sData[WARP_NUM];
    int tidY = threadIdx.x;
    int warp_id = tidY / WARP_SIZE;
    int lane_id = tidY % WARP_SIZE;
    v = warp_reduce_sum(v);
    if(lane_id == 0) {
      sData[warp_id] = v;
    }
    __syncthreads();
    v = lane_id < WARP_NUM ? sData[lane_id] : 0.0f;
    if(warp_id == 0) {
      v = warp_reduce_sum(v);
    }
    if(tidY == 0) {
      sData[tidY] = v;
    }
    __syncthreads();
    return sData[0];
  }
  int tidY = threadIdx.x;
  int offX = threadIdx.y + blockDim.y * blockIdx.y;
  
  const int warp_sum = (N + WARP_SIZE - 1) / WARP_SIZE;
  const int off = offX * warp_sum;

  int warp_id = tidY / WARP_SIZE;
  int lane_id = tidY % WARP_SIZE;

  v = warp_reduce_sum(v);
  // 全局store
  if(lane_id == 0) {
    gBlkData[off + warp_id] = v;
  }
  // 全局Load
  // 不同warp的lane_id=0都是v
  for(int stride = warp_sum >> 1; stride > 0; stride >>= 1) {
    gBlkData[off + warp_id] += gBlkData[off + warp_id + stride];
  }
  return gBlkData[off];    
}

// softmax以行为单位, 对[M,N]矩阵逐行进行
// 以行为一个block, 求Y向的softmax
// 
__global__ void self_attn_fp32_kernel(
    const float *Q, const float *K, const float *V,
    float *output,
    int M, int N, int d,
    float *gBlkData
) {
  int tidX = threadIdx.y;
  int tidY = threadIdx.x;
  int offX = tidX + blockDim.y * blockIdx.y;
  int offY = tidY + blockDim.x * blockIdx.x;
  if(offX >= M || offY >= N) return;
  // 计算[M,N]矩阵的初始值, KT[x,y] = K[y,x]
  // 使用KT的[i, offY]即K[offY, i]
  float fac = 1.0f / sqrt((float)d);
  float qkt = 0.0f;
  for(int i=0;i<d;++i) {
    int offQ = offX * d + i;
    int offKT = offY * d + i;
    if(offQ < M * d && offKT < N * d) {
      qkt += Q[offQ] * K[offKT];
    }         
  }
  qkt /= sqrt((float)d);
  // [offX, offY]进行Y方向block的softmax
  float maxVal = block_reduce_max_Y(qkt, N, gBlkData);
  float sumVal = block_reduce_sum_Y(qkt, maxVal, N, gBlkData);
  // printf("loc: (%d, %d), qkt: %.9f maxVal: %.9f sumVal: %.9f\n", offX, offY, qkt, maxVal, sumVal);
  // printf("sumVal: %.9f\n", sumVal);
  float softmax_qkt = exp(qkt - maxVal) / sumVal;
  // 计算结果
  for(int i=0;i<d;++i) {
    int offV = offY * d + i;
    int output_off = offX * d + i;
    if(output_off < M * d) {
      atomicAdd(&output[output_off], V[offV] * softmax_qkt);
    }
  }
}

TORCH_BINDING_SELF_ATTN_IMPL(fp32, torch::kFloat32, float, 1)