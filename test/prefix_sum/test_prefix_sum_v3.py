from typing import Optional
from common.torch_env import extra_cuda_cflags, extra_cflags, src_path, extra_include_paths

import torch, os, sys, time, random
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

torch.backends.cudnn.allow_tf32 = False
# torch.backends.cuda.matmul.allow_tf32 = False

epoch = 10000
# 100000000数据规模，迭代100次，算子时间4267us
def validate() -> bool:
    N = random.randint(1, 100)
    x = torch.randn((N)).cuda().contiguous()
    ref = torch.cumsum(x, 0).cpu()
    torch.cuda.synchronize()
    res = torch.ops.CudaLab.prefix_sum_v3_fp32_1d(x).cpu()
    torch.cuda.synchronize()
    # float单步误差1e-7,绝对误差最高1e-5，相对误差最高1e-6
    try:
        torch.testing.assert_close(
          res,
          ref,
          rtol=1e-4,
          atol=1e-5
        )
    except:
        print(f"validate prefix_sum_v3_fp32_1d failed at N:{N}, x:{x}, ref:{ref}, res:{res}")
        return False
    
    return True
    
def validate_prefix_sum_v3_fp32_1d():
    for i in range(epoch):
        if validate() is False:
            return False
    print("validate prefix_sum_v3_fp32_1d done")
    return True

def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 100,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    # warmup
    if out is not None:
        for i in range(warmup):
            out = perf_func(x)
    else:
        for i in range(warmup):
            _ = perf_func(x)
    torch.cuda.synchronize()

    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            out = perf_func(x)
    else:
        for i in range(iters):
            out = perf_func(x)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"prefix_sum_v3_{tag}"
    out_val = out.detach().cpu().numpy().tolist()
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>18}: time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time

def run_benchmark_prefix_sum_v3_fp32_1d():
    N = 10000000
    x = torch.randn((N)).cuda().float().contiguous()  
    out = torch.randn((N)).cuda().float().contiguous() 
    run_benchmark(
      torch.ops.CudaLab.prefix_sum_v3_fp32_1d,
      x,
      "fp32_1d",
      out
    )

def test_prefix_sum_v3():
    validate_prefix_sum_v3_fp32_1d()
    run_benchmark_prefix_sum_v3_fp32_1d()
