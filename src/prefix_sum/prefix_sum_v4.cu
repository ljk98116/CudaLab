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


__global__ void ScanAndWritePartSumKernelV4(
  const float *input, BlockPrefix *g_block_prefix, float *output, int N, int *g_vp_counter
) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float v = idx < N ? input[idx] : 0.0f;
  // 计算block内前缀和
  float aggregate = ScanBlock(v);
  __shared__ int vp_id;
  // 线程0写全局数组，写partial
  if(threadIdx.x == blockDim.x - 1) {
    vp_id = atomicAdd(g_vp_counter, 1);
    // g_block_prefix[blockIdx.x].aggregate = aggregate;
    // __threadfence(); //保证此时这个block的sum属于全局可见的状态,注意区分__threadfence_block()对block内可见
    // g_block_prefix[blockIdx.x].state = PARTIAL;
    BlockPrefix v;
    v.sum = aggregate;
    v.state = PARTIAL;
    atomicExch(
      reinterpret_cast<unsigned long long*>(&g_block_prefix[vp_id].packed),
      v.packed
    );
  }
  __syncthreads();
  
  // float block_exclusive = 0.0f;
  // int lane = threadIdx.x & 31;
  // int lookback = vp_id - 1;
  // 串行执行look back
  // if(tid == 31 && vp_id > 0) {
  //   while(lookback >= 0) {
  //     int state = g_block_prefix[lookback].state;
  //     if(state == INVALID) continue;
  //     if(state == PARTIAL) {
  //       block_exclusive += g_block_prefix[lookback].sum;
  //       --lookback;
  //     }
  //     else if(state == COMPLETE) {
  //       block_exclusive += g_block_prefix[lookback].sum;
  //       break;
  //     }
  //   }
  //   // g_block_prefix[bid].inclusive_prefix = aggregate + block_exclusive;
  //   // __threadfence();
  //   // g_block_prefix[bid].state = COMPLETE;
  //   BlockPrefix v_acc;
  //   v_acc.sum = aggregate + block_exclusive;
  //   v_acc.state = COMPLETE;
  //   atomicExch(
  //     reinterpret_cast<unsigned long long*>(&g_block_prefix[vp_id].packed),
  //     v_acc.packed
  //   );
  // }

  // 并行执行lookback
  // 循环往前推一组线程
  int state = INVALID; // 当前线程映射到对应的之前的part的状态
  float sum = 0.0f; // 当前线程迭代过程产生的累加和
  
  int lookback = vp_id - 1;
  lookback -= WARP_SIZE - 1 - tid;
  if(tid < WARP_SIZE) {
    while(lookback >= 0) {
      BlockPrefix b_prefix;
      unsigned long long packed;
      do {
        // packed = atomicAdd(
        //   reinterpret_cast<unsigned long long*>(&g_block_prefix[lookback].packed),
        //   0ull
        // );
        packed = __ldg(&g_block_prefix[lookback].packed);
      } while((packed >> 32) == INVALID);

      b_prefix.packed = packed;

      state = b_prefix.state;
      sum += b_prefix.sum;

      if(state == COMPLETE) break;

      lookback -= WARP_SIZE; // 向前看一个WARP_SIZE
    }
  }
  // // 对sum值做warp reduce add
  float block_exclusive = ScanWarp(sum);
  // 取lane = 31时的结果
  block_exclusive = __shfl_sync(0xffffffff, block_exclusive, 31);
  // 只在tid = 31时,用block_exclusive更新状态
  if(tid == WARP_SIZE - 1) {
    // 更新block状态
    BlockPrefix v_acc;
    v_acc.sum = aggregate + block_exclusive;
    v_acc.state = COMPLETE;
    atomicExch(
      reinterpret_cast<unsigned long long*>(&g_block_prefix[vp_id].packed),
      v_acc.packed
    );
  }
  // 累加前面的block的sum
  aggregate += block_exclusive;
  if(idx < N) {
    output[idx] = aggregate;
  }
}

TORCH_BINDING_PREFIX_SUM_V4_IMPL(fp32_1d, torch::kFloat32, float, 1)
