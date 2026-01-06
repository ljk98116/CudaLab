#include <self_attn/self_attn.hpp>

// static __device__ __forceinline__ float warp_reduce_max(float v) {
// #pragma unroll
//     for(int stride = WARP_SIZE >> 1; stride > 0; stride >>= 1) {
//         v = fmax(v, __shfl_xor_sync(0xffffffff, v, stride));
//     }
//     return v;
// }

// static __device__ __forceinline__ float warp_reduce_sum(float v) {
// #pragma unroll
//     for(int stride = WARP_SIZE >> 1; stride > 0; stride >>= 1) {
//         v += __shfl_xor_sync(0xffffffff, v, stride);
//     }
//     return v;
// }

// 单核的HMem空间复杂度 2 * Bc * d + Br * d + Br * Bc
// Bc这里给大小为1, 相当于空间复杂度只有O(M), 符合flashattention的复杂度描述
// 每行是一个线程，按Br的大小分成Tr份，在列方向可能迭代N次，运行速度会受影响
template <int Bc, int Br>
__global__ void self_attn_fp32_v1_kernel(
  const float *Q, const float *K, const float *V,
  float *output,
  int M, int N, int d,
  float *l, float *m
) {
  // tile内线程的位置[行方向]
  int tx = threadIdx.x;
  int Tc = (N + Bc - 1) / Bc; // N方向的tile个数
  int row_tile = blockIdx.x;
  int row = row_tile * Br + tx;
  if(row >= M) return;
  extern __shared__ float sMem[];
  int K_off = Bc * d;
  int V_off = K_off;
  int Q_off = Br * d;
  float *Kj = sMem; // Bc * d
  float *Vj = &sMem[K_off]; // Bc * d
  float *Qi = &sMem[K_off + V_off]; // Br * d
  float *S = &sMem[K_off + V_off + Q_off]; //Bc * Br个, 表示一行的qkt/sqrt(d), softmax(qkt/sqrt(d))
  // 遍历所有的列块，列方向迭代
  for(int j=0;j<Tc;++j) {
    // Load Kj, Vj, 防止多线程读取，因为Q的每一个行块都会与K，V的所有行块进行乘法
    // 可以使用矩阵型的load优化
    if(tx == 0) {
      for(int x=0;x<Bc;++x) {
        for(int y=0;y<d;++y) {
          int k_row = j * Bc + x;
          if (k_row < N) {
            Kj[x * d + y] = K[k_row * d + y];
            Vj[x * d + y] = V[k_row * d + y];
          } else {
            Kj[x * d + y] = 0.f;
            Vj[x * d + y] = 0.f;
          }
        }
      }
    }
    __syncthreads();
    // 对Q的每个行块计算tile内QKT，统计行方向最大值,这里的行代表全局结果的第row行
    // 预先读取Q的行块到sram，提高计算速度
    for(int x=0;x<d;++x) {
      Qi[tx * d + x] = Q[row * d + x];
    }
    __syncthreads();
    // 读取历史Q这个tile内行的最大值以及exp指数和到寄存器
    float tile_row_max_prev = m[row];
    float tile_row_sum_prev = l[row];
    // 遍历Q的行块对应的所有列块，计算tile内的行row最大值
    float tile_row_max = -FLT_MAX;
    for(int x=0;x<Bc;++x) {
      float qkt = 0.0f;
      for(int y=0;y<d;++y) {
        qkt += Qi[tx * d + y] * Kj[x * d + y];
      }
      qkt *= rsqrtf((float)d);
      S[tx * Bc + x] = qkt; // 存储行qkt值
      tile_row_max = fmaxf(qkt, tile_row_max);
    }
    // 根据tile_row_max计算当前tile内的行row的sum
    float tile_row_sum = 0.0f;
    for(int x = 0;x < Bc ;++x) {
      S[tx * Bc + x] = expf(S[tx * Bc + x] - tile_row_max);
      tile_row_sum += S[tx * Bc + x];
    }
    // 更新行row方向最大值和行方向的exp指数和，以及输出值
    float tile_row_max_new = fmaxf(tile_row_max, tile_row_max_prev);
    float tile_row_sum_new = expf(tile_row_max_prev - tile_row_max_new) * tile_row_sum_prev + 
      expf(tile_row_max - tile_row_max_new) * tile_row_sum;
    // 重新计算新的softmax(qkt/sqrt(d)) @ V
    for(int x=0;x<d;++x) {
      float pv = 0.0f;
      for(int y=0;y<Bc;++y) {
        pv += S[tx * Bc + y] * Vj[y * d + x]; 
      }
      int output_off = row * d + x;
      output[output_off] =  
        (
          tile_row_sum_prev * expf(tile_row_max_prev - tile_row_max_new) * output[output_off] + 
          expf(tile_row_max - tile_row_max_new) * pv
        ) / tile_row_sum_new;
    }
    // 寄存器值写入gMem, 各个列块是迭代更新行的最值的
    m[row] = tile_row_max_new;
    l[row] = tile_row_sum_new;
    __syncthreads();
  }
}

TORCH_BINDING_SELF_ATTN_V1_IMPL(fp32_v1, torch::kFloat32, float, 1)