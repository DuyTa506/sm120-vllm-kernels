"""Shared helpers for the DeepGEMM-SM120 study: CUDA-graph timing, stdout capture of DG_PRINT_CONFIGS,
synthetic FP8 operands with exact UE8M0 (power-of-two) scales, JSONL result writer."""
import json
import os
import statistics
import subprocess
import sys
import tempfile

import torch

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def capture_fd1(fn):
    """Run fn() and return (result, text written to fd 1, e.g. DG_PRINT_CONFIGS output from C++)."""
    sys.stdout.flush()
    tmp = tempfile.TemporaryFile(mode="w+b")
    saved = os.dup(1)
    os.dup2(tmp.fileno(), 1)
    try:
        res = fn()
        torch.cuda.synchronize()
    finally:
        os.dup2(saved, 1)
        os.close(saved)
    tmp.seek(0)
    return res, tmp.read().decode(errors="replace")


def bench_graph(fns, iters=30, warmup=3):
    """Capture all callables (in order) into one CUDA graph and time replays.
    Returns (median us per callable, min us, max us)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / len(fns))
    del g
    return statistics.median(ts), min(ts), max(ts)


def calc_diff(x, y):
    x, y = x.double(), y.double()
    den = (x * x + y * y).sum()
    if den == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / den)


def fp8_rand(shape, scale=1.0, device="cuda"):
    return (torch.randn(shape, device=device, dtype=torch.float32) * scale).to(torch.float8_e4m3fn)


def pow2_sf(shape, lo, hi, device="cuda"):
    """Exact UE8M0-representable float scales 2^randint(lo, hi)."""
    return torch.exp2(torch.randint(lo, hi, shape, device=device).float())


def make_act(M, K, seed=0, device="cuda"):
    """bf16 activations with LLM-like per-channel outlier structure (same recipe as kernels/dense-fp8-gemm)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ch = torch.exp(torch.randn(K, generator=g) * 0.6)
    idx = torch.randperm(K, generator=g)[: max(1, K // 256)]
    ch[idx] *= 20.0
    return (torch.randn(M, K, generator=g) * ch).to(torch.bfloat16).to(device)


def gpu_state():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu,memory.used", "--format=csv,noheader"],
            timeout=10).decode().strip()
        return out
    except Exception as e:  # noqa: BLE001
        return f"n/a ({type(e).__name__})"


class Out:
    def __init__(self, name):
        os.makedirs(RESULTS, exist_ok=True)
        self.path = os.path.join(RESULTS, name)
        self.f = open(self.path, "a")

    def write(self, **kw):
        self.f.write(json.dumps(kw) + "\n")
        self.f.flush()


def kernel_names(fn):
    """Names of the CUDA kernels launched by fn() (from torch.profiler/CUPTI). The JIT'd DeepGEMM kernel name carries
    its template arguments (BLOCK_M/BLOCK_N/stages/...), which is the ground truth for the picked tile."""
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]


def gemm_kernel(fn):
    try:
        names = kernel_names(fn)
    except Exception as e:  # noqa: BLE001
        return f"profiler failed: {type(e).__name__}"
    g = [n for n in names if "gemm" in n.lower() or "sm120" in n.lower()]
    return (g[0] if g else (names[0] if names else ""))[:400]
