#include <add2/add2.hpp>

__global__ void add2_fp32_1d_kernel(const float *v1, const float *v2, float* output, int N) {
  int off = threadIdx.x + blockIdx.x * blockDim.x;
  if(off < N) {
    output[off] = v1[off] + v2[off];
  }
}

TORCH_BINDING_IMPL_ADD2_1D(fp32_1d, torch::kFloat32, float, 1)
