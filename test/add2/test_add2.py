from typing import Optional
from common.torch_env import extra_cuda_cflags, extra_cflags, src_path, extra_include_paths

import torch, os, sys, time, random
from torch.utils.cpp_extension import load
torch.set_grad_enabled(False)
# os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'

epoch = 10000

def validate() -> bool:
    N = random.randint(1, 1000)
    x = torch.randn((N)).cuda().float().contiguous()
    y = torch.randn((N)).cuda().float().contiguous()
    ref = torch.add(x, y).flatten().cpu()
    # res = lib.add2_fp32_1d(x, y)
    res = torch.ops.CudaLab.add2_fp32_1d(x, y)
    torch.cuda.synchronize()
    if not torch.equal(ref, res.cpu()):
        print(f"validate add2_fp32_1d failed at N:{N}, x:{x}, y:{y}")
        torch.cuda.synchronize()
        return False
    return True
    
def validate_add2_fp32_1d():
    for i in range(epoch):
        if validate() is False:
            return False
    print("validate add2_fp32_1d done")
    return True

def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    y: torch.Tensor,
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
            out = perf_func(x, y)
    else:
        for i in range(warmup):
            _ = perf_func(x, y)
    torch.cuda.synchronize()

    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            out = perf_func(x, y)
    else:
        for i in range(iters):
            out = perf_func(x, y)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"add2_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time

def run_benchmark_add2_fp32_1d():
    N = 5
    x = torch.randn((N)).cuda().float().contiguous()
    y = torch.randn((N)).cuda().float().contiguous()   
    out = torch.randn((N)).cuda().float().contiguous() 
    run_benchmark(
        torch.ops.CudaLab.add2_fp32_1d,
        x,
        y,
        "fp32_1d",
        out
    )

def test_add2():
    validate_add2_fp32_1d()
    run_benchmark_add2_fp32_1d()
