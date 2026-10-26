import glob, json, os
import torch
from safetensors import safe_open

SNAP = glob.glob("/hf/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/*/")[0]
_WMAP = json.load(open(SNAP + "model.safetensors.index.json"))["weight_map"]
L = "model.language_model.layers."


def real(name, dev="cuda"):
    with safe_open(SNAP + _WMAP[name], "pt", device=dev) as f:
        return f.get_tensor(name)


def normed_input(rows, ln_w, gen=None):
    """RMSNorm-ed residual stream with outlier channels (same recipe as decode-roofline/verify_skinny.py)."""
    h = torch.randn(rows, 2048, device="cuda", generator=gen)
    h[:, torch.randint(0, 2048, (12,), device="cuda", generator=gen)] *= 20
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    return (h * (1 + ln_w.float())).bfloat16()


def time_graph(fn, reps=30, warm=3):
    """Median per-replay time (us) of a CUDA graph capturing fn()."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]
