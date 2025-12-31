from typing import Optional
from common.torch_env import extra_cuda_cflags, extra_cflags, src_path, extra_include_paths

import torch, os, sys, time, random
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'

epoch = 10000

def validate() -> bool:
    N = random.randint(1, 100)
    x = torch.randn((N)).cuda().to(torch.float32).contiguous()
    ref = torch.sum(x, dtype=torch.float32).flatten().to(torch.float32).cpu()
    torch.cuda.synchronize()
    # res = lib.add2_fp32_1d(x, y)
    res = torch.ops.CudaLab.reduce_fp32_1d(x).flatten().cpu()
    torch.cuda.synchronize()
    # float单步误差1e-7,绝对误差最高1e-5，相对误差最高1e-6
    try:
        torch.testing.assert_close(
            res,
            ref,
            rtol=1e-5,
            atol=1e-6
        )
    except:
        print(f"validate reduce_fp32_1d failed at N:{N}, x:{x}, ref:{ref[0]:.9f}, res:{res.flatten().to(torch.float32)[0]:.9f}")
        return False
    
    return True
    
def validate_reduce_fp32_1d():
    for i in range(epoch):
        if validate() is False:
            return False
    print("validate reduce_fp32_1d done")
    return True

def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
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
    out_info = f"reduce_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time

def run_benchmark_reduce_fp32_1d():
    N = 5
    x = torch.randn((N)).cuda().float().contiguous()  
    out = torch.randn((N)).cuda().float().contiguous() 
    run_benchmark(
        torch.ops.CudaLab.reduce_fp32_1d,
        x,
        "fp32_1d",
        out
    )

def test_reduce():
    validate_reduce_fp32_1d()
    run_benchmark_reduce_fp32_1d()