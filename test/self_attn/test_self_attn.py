import math
from typing import Optional
from common.torch_env import extra_cuda_cflags, extra_cflags, src_path, extra_include_paths

import torch, os, sys, time, random
from torch.utils.cpp_extension import load
torch.set_grad_enabled(False)

epoch = 200
def validate() -> bool:
    d = random.randint(1, 128)
    # little wide <=32
    M = random.randint(1, 32)
    N = random.randint(1, 32)
    Q = torch.randn((M, d)).cuda().float().contiguous()
    K = torch.randn((N, d)).cuda().float().contiguous()
    V = torch.randn((N, d)).cuda().float().contiguous()
    scores = Q @ K.T / math.sqrt(d)     # [M, N]
    attn   = torch.softmax(scores, dim=1)
    ref    = attn @ V                   # [M, d]
    ref = ref.cpu()
    res = torch.ops.CudaLab.self_attn_fp32_v1(Q, K, V)
    torch.cuda.synchronize()
    res = res.cpu()
    # float单步误差1e-7,绝对误差最高1e-4，相对误差最高1e-5
    try:
        torch.testing.assert_close(
          res,
          ref,
          rtol=1e-4,
          atol=1e-5
        )
    except:
        print(f"validate self attn_fp32 failed at N:{N},ref:{ref}, res:{res}")
        return False
    # medium wide <=1024
    M = random.randint(33, 1024)
    N = random.randint(33, 1024)
    Q = torch.randn((M, d)).cuda().contiguous()
    K = torch.randn((N, d)).cuda().contiguous()
    V = torch.randn((N, d)).cuda().contiguous()
    scores = Q @ K.T / math.sqrt(d)     # [M, N]
    attn   = torch.softmax(scores, dim=1)
    ref    = attn @ V                   # [M, d]
    ref = ref.cpu()
    res = torch.ops.CudaLab.self_attn_fp32_v1(Q, K, V).cpu()
    torch.cuda.synchronize()
    # float单步误差1e-7,绝对误差最高1e-4，相对误差最高1e-5
    try:
        torch.testing.assert_close(
          res,
          ref,
          rtol=1e-4,
          atol=1e-5
        )
    except:
        print(f"validate self attn_fp32 failed at N:{N},ref:{ref}, res:{res}")
        return False    
    # large wide > 1024
    M = random.randint(1025, 2048)
    N = random.randint(1025, 2048)
    Q = torch.randn((M, d)).cuda().contiguous()
    K = torch.randn((N, d)).cuda().contiguous()
    V = torch.randn((N, d)).cuda().contiguous()
    scores = Q @ K.T / math.sqrt(d)     # [M, N]
    attn   = torch.softmax(scores, dim=1)
    ref    = attn @ V                   # [M, d]
    ref = ref.cpu()
    res = torch.ops.CudaLab.self_attn_fp32_v1(Q, K, V).cpu()
    torch.cuda.synchronize()
    # float单步误差1e-7,绝对误差最高1e-4，相对误差最高1e-5
    try:
        torch.testing.assert_close(
          res,
          ref,
          rtol=1e-4,
          atol=1e-5
        )
    except:
        print(f"validate self attn_fp32 failed at N:{N},ref:{ref}, res:{res}")
        return False
    return True

def validate_self_attn_fp32_v1():
    for i in range(epoch):
        if validate() is False:
            return False
    print("validate self attn_fp32_1d done")
    return True

def run_benchmark(
    perf_func: callable,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
  if out is not None:
      out.fill_(0)
  # warmup
  if out is not None:
      for i in range(warmup):
          out = perf_func(Q, K, V)
  else:
      for i in range(warmup):
          _ = perf_func(Q, K, V)
  torch.cuda.synchronize()

  start = time.time()
  # iters
  if out is not None:
      for i in range(iters):
          out = perf_func(Q, K, V)
  else:
      for i in range(iters):
          out = perf_func(Q, K, V)
  torch.cuda.synchronize()
  end = time.time()
  total_time = (end - start) * 1000  # ms
  mean_time = total_time / iters
  out_info = f"self_attn_{tag}"
  out_val = out.detach().cpu().numpy().tolist()
  print(f"{out_info:>18}:time:{mean_time:.8f}ms")
  if show_all:
      print(out)
  return out, mean_time

def run_benchmark_self_attn_fp32_v1():
    M = 10
    N = 10
    d = 12
    Q = torch.randn((M, d)).cuda().float().contiguous()
    K = torch.randn((N, d)).cuda().float().contiguous()
    V = torch.randn((N, d)).cuda().float().contiguous()   
    out = torch.randn((M, d)).cuda().float().contiguous() 
    run_benchmark(
      torch.ops.CudaLab.self_attn_fp32_v1,
      Q, K, V,
      "fp32_1d",
      out
    )

def test_self_attn():
    validate_self_attn_fp32_v1()
    run_benchmark_self_attn_fp32_v1()