import math
from typing import Optional
from common.torch_env import extra_cuda_cflags, extra_cflags, src_path, extra_include_paths

import torch, os, sys, time, random
from torch.utils.cpp_extension import load
torch.set_grad_enabled(False)

epoch = 200
def validate() -> bool:
    d = 256
    nh = 64
    B = 8
    # little wide <=32
    N = random.randint(1, 32)
    Q = torch.randn((B, N, d)).cuda().float().contiguous()
    K = torch.randn((B, N, d)).cuda().float().contiguous()
    V = torch.randn((B, N, d)).cuda().float().contiguous()
    multihead_attn = torch.nn.MultiheadAttention(d, nh, batch_first=True, device=torch.device('cuda'))
    ref, _ = multihead_attn.forward(Q, K, V)
    ref = ref.cpu()
    res = torch.ops.CudaLab.flash_attn_fp32_v1(Q, K, V)
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
        print(f"validate flash_attn_fp32_v1 failed at N:{N},ref:{ref}, res:{res}")
        return False
    # medium wide <=1024
    N = random.randint(33, 1024)
    Q = torch.randn((B, N, d)).cuda().contiguous()
    K = torch.randn((B, N, d)).cuda().contiguous()
    V = torch.randn((B, N, d)).cuda().contiguous()
    multihead_attn = torch.nn.MultiheadAttention(d, nh, batch_first=True, device=torch.device('cuda'))
    ref, _ = multihead_attn.forward(Q, K, V)
    ref = ref.cpu()
    res = torch.ops.CudaLab.flash_attn_fp32_v1(Q, K, V).cpu()
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
        print(f"validate flash_attn_fp32_v1 failed at N:{N},ref:{ref}, res:{res}")
        return False    
    # large wide > 1024
    N = random.randint(1025, 2048)
    Q = torch.randn((B, N, d)).cuda().contiguous()
    K = torch.randn((B, N, d)).cuda().contiguous()
    V = torch.randn((B, N, d)).cuda().contiguous()
    multihead_attn = torch.nn.MultiheadAttention(d, nh, batch_first=True, device=torch.device('cuda'))
    ref, _ = multihead_attn.forward(Q, K, V)
    ref = ref.cpu()
    res = torch.ops.CudaLab.flash_attn_fp32_v1(Q, K, V).cpu()
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
        print(f"validate flash_attn_fp32_v1 failed at N:{N},ref:{ref}, res:{res}")
        return False
    return True

def validate_flash_attn_fp32_v1():
    for i in range(epoch):
        if validate() is False:
            return False
    print("validate flash_attn_fp32_v1_1d done")
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
  out_info = f"flash_attn_{tag}"
  out_val = out.detach().cpu().numpy().tolist()
  print(f"{out_info}: time:{mean_time:.8f}ms")
  if show_all:
      print(out)
  return out, mean_time

def run_benchmark_torch(
    perf_func: callable,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
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
  out_info = "flash_attn_torch"
  out_val = out.detach().cpu().numpy().tolist()
  print(f"{out_info}: time:{mean_time:.8f}ms")
  if show_all:
      print(out)
  return out, mean_time

def run_benchmark_flash_attn_fp32_v1():
    N = 256
    d = 128
    nh = 64
    B = 8
    Q = torch.randn((B, N, d)).cuda().float().contiguous()
    K = torch.randn((B, N, d)).cuda().float().contiguous()
    V = torch.randn((B, N, d)).cuda().float().contiguous()   
    out = torch.randn((B, N, d)).cuda().float().contiguous() 
    run_benchmark(
      torch.ops.CudaLab.flash_attn_fp32_v1,
      Q, K, V,
      "fp32_1d",
      out
    )
    def torch_attn(Q, K, V):
      multihead_attn = multihead_attn = torch.nn.MultiheadAttention(d, nh, batch_first=True, device=torch.device('cuda'))
      ref, _ = multihead_attn.forward(Q, K, V)        
      return ref
    run_benchmark_torch(torch_attn, Q, K, V, out)

def test_flash_attn():
    validate_flash_attn_fp32_v1()
    run_benchmark_flash_attn_fp32_v1()