import torch, os, sys, time
from typing import Optional

# torch.set_grad_enabled(False)
# os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'

extra_cuda_cflags=[
  "-O3",
  "-U__CUDA_NO_HALF_OPERATORS__",
  "-U__CUDA_NO_HALF_CONVERSIONS__",
  "-U__CUDA_NO_HALF2_OPERATORS__",
  "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
  "--expt-relaxed-constexpr",
  "--expt-extended-lambda",
]

extra_cflags=["-std=c++17", "-j32"]

src_dir = os.getenv("CUDA_KERNEL_SRC_DIR")
src_path = os.getcwd() + "/src/" if src_dir is None else src_dir
extra_include_paths = [os.getcwd() + "/include/"]