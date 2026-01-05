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

// softmax以行为单位, 对[M,N]矩阵逐行进行
// tx是代表tile内的一列，ty是tile内的一行, 一个block里Bc个线程
// -> tx[Bc个]
// \|/
// ty[Br个]
//
template <int Bc, int Br>
__global__ void self_attn_fp32_v1_kernel(
  const float *Q, const float *K, const float *V,
  float *output,
  int M, int N, int d,
  float *l, float *m
) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int Tr = (M + Br - 1) / Br;
  int Tc = (N + Bc - 1) / Bc;
  // sram, Kj[Bc x d], Vj [Bc x d]
  // sram mapping Br x d + Bc x d + Bc x d [Qi, Kj, Vj]
  int Q_off = Br * d;
  int K_off = Bc * d;
  int V_off = Bc * d;
  int S_off = Br * Bc;
  extern __shared__ float sMem[];
  float *Qi = sMem;
  float *Kj = &sMem[Q_off];
  float *Vj = &sMem[Q_off + K_off];
  float *Sij = &sMem[Q_off + K_off + V_off];
  for(int j=0;j<Tc;++j) {
    // Load Kj, Vj
    for(int i=0;i<d;++i) {
      Kj[tx * d + i] = K[K_off * j + tx * d + i];
      Vj[tx * d + i] = V[K_off * j + tx * d + i];
    }
    __syncthreads(); // Bc内同步
    for(int i=0;i<Tr;++i) {
      // Load Qi
      for(int x=0;x<d;++x) {
        Qi[ty * d + x] = Q[i * Q_off + ty * d + x];
      }
      __syncthreads();
      // calculate Sij
      // Br x d 
      // Bc x d
      float row_max_prev = m[i * Br + ty];
      float row_sum_prev = l[i * Br + ty];
      for(int i1=0;i1<Br;++i1) {
        float row_max = -FLT_MAX;
        for(int j1=0;j1<Bc;++j1) {
          float sum = 0.0f;
          for(int k1=0;k1<d;++k) {
            sum += Qi[i1 * d + k1] * Kj[j1 * d + k1];
          }
          sum /= sqrtf((float)d);
          Sij[i1 * Bc + j1] = sum;
          row_max = fmax(row_max, sum);
        }
        // Pi1j1 = Si1j1 - row_max
        // update Sij use exp
        float row_sum = 0.0f;
        for(int j1=0;j1<Bc;++j1) {
          Sij[i1 * Bc + j1] = exp(Sij[i1 * Bc + j1] - row_max);
          row_sum += Sij[i1 * Bc + j1];
        }
        // update new m and l(one row)
        float m_new = fmax(row_max, row_max_prev);
        float l_new = exp(row_max_prev - m_new) * row_sum_prev + exp(m_new - row_max_prev) * row_sum_prev;
        m[i * Br + ty] = m_new;
        l[i * Br + ty] = l_new;
        // update output
        for(j1=0;j1<d;++j1) {
          float SV = 0.0f;
          for(k1=0;k1<Bc;++k1) {
            SV += Sij[i1 * Bc + k1] * Vj[k1 * d + j1];
          }
          output[i * Q_off + ty * d + j1] = 
            1.0f / l_new * row_sum_prev * exp(row_max_prev - m_new) * output[i * Q_off + ty * d + j1] + 
            exp(row_max - m_new) * SV;
        }
        __syncthreads();
      }
      __syncthreads();
    }
  }
}

TORCH_BINDING_SELF_ATTN_V1_IMPL(fp32_v1, torch::kFloat32, float, 1)