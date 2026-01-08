#include <prefix_sum/prefix_sum_v4.hpp>

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

static __device__ __forceinline__ float ScanBlock(float v) {
  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;
  // 每个warp的前缀和,初始为每个warp内的值的和
  __shared__ float warp_sum[WARP_NUM];
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
  return v;
}

static __device__ __forceinline__
float WarpLookbackPipeline(int bid, BlockPrefix* g) {
  float local = 0.0f;
  int lane = threadIdx.x & 31;

  // 每个 warp 处理一个 window
  int look = bid - 1 - (threadIdx.x >> 5) * 32;

  while (look >= 0) {
    int idx = look - lane;

    int state = INVALID;
    float val = 0.0f;

    if (idx >= 0) {
      state = g[idx].state;
      val   = g[idx].sum;
    }

    unsigned mask_complete =
      __ballot_sync(0xffffffff, state == COMPLETE);

    if (mask_complete) {
      int leader = __ffs(mask_complete) - 1;
      if (lane == leader)
        local += val;
      break;
    }

    if (state == PARTIAL)
      local += val;

    look -= 32 * (blockDim.x >> 5);
  }

  // warp reduction
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    local += __shfl_down_sync(0xffffffff, local, off);

  return local;
}


__global__ void ScanAndWritePartSumKernelV4(
  const float *input, BlockPrefix *g_block_prefix, float *output, int N
) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float v = idx < N ? input[idx] : 0.0f;
  // 计算block内前缀和
  float block_prefix = ScanBlock(v);

  // warp0 lane31 拿最终 block_sum
  float block_sum;
  if ((threadIdx.x >> 5) == 0 && (threadIdx.x & 31) == WARP_NUM - 1)
    block_sum = block_prefix;

  // broadcast
  block_sum = __shfl_sync(0xffffffff, block_sum, WARP_NUM - 1);

  // 线程0写全局数组，写partial
  if(threadIdx.x == 0) {
    g_block_prefix[blockIdx.x].sum = block_sum;
    __threadfence(); //保证此时这个block的sum属于全局可见的状态,注意区分__threadfence_block()对block内可见
    g_block_prefix[blockIdx.x].state = PARTIAL;
  }
  __syncthreads();

  __shared__ bool do_lookback;
  if (tid == 0) {
    int old = atomicCAS(&g_block_prefix[bid].state,
                        PARTIAL,
                        LOOKBACK);
    do_lookback = (old == PARTIAL);
  }
  __syncthreads();

  // look back, 将全局block前缀和数组进行更新
  __shared__ float base;
  float warp_base = WarpLookbackPipeline(bid, g_block_prefix);

  if ((threadIdx.x & 31) == 0) {
    if ((threadIdx.x >> 5) == 0)
      base = warp_base;
    else
      atomicAdd(&base, warp_base);
  }
  __syncthreads();

  // 完成 lookback
  if (do_lookback && tid == 0) {
    g_block_prefix[bid].sum += base;
    __threadfence();
    g_block_prefix[bid].state = COMPLETE;
  }
  __syncthreads();

  // 非 lookback block：等前一个 COMPLETE
  if (!do_lookback && bid > 0 && tid == 0) {
    while (g_block_prefix[bid - 1].state != COMPLETE) {
      // very short spin
    }
  }
  __syncthreads();
  // 累加前面的block的sum
  if(blockIdx.x > 0) block_prefix += g_block_prefix[blockIdx.x - 1].sum;
  if(idx < N) {
    output[idx] = block_prefix;
  }
}

TORCH_BINDING_PREFIX_SUM_V4_IMPL(fp32_1d, torch::kFloat32, float, 1)
