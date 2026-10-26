"""Does TORCHINDUCTOR_ENABLE_PDL=1 shrink a chain of small inductor-generated Triton kernels in a CUDA graph?
Each layer: residual add + rmsnorm + per-token amax + fp8 quant/dequant (inductor fuses to 1-3 Triton kernels,
like vLLM's triton_red_fused_* kernels) followed by a bf16 cuBLAS linear (a kernel that cannot use PDL).
usage (inside the vLLM image):  TORCHINDUCTOR_ENABLE_PDL=0|1 python3 pdl_inductor.py [L] [M]
"""
import os, statistics, sys, time
import torch

L = int(sys.argv[1]) if len(sys.argv) > 1 else 12
M = int(sys.argv[2]) if len(sys.argv) > 2 else 4
H = 2048
dev = "cuda"
torch.manual_seed(0)
pdl = os.environ.get("TORCHINDUCTOR_ENABLE_PDL", "0")
Ws = [(torch.randn(H, H, device=dev, dtype=torch.bfloat16) * 0.02) for _ in range(L)]   # 8 MB each, 12 -> 96 MB (< L2 128 MB)
Ws += [(torch.randn(H, H, device=dev, dtype=torch.bfloat16) * 0.02) for _ in range(L)]  # use 2L matrices so the set exceeds L2
wn = [torch.rand(H, device=dev, dtype=torch.bfloat16) + 0.5 for _ in range(L)]


def model(x, r):
    for i in range(L):
        r = x + r
        rf = r.float()
        y = rf * torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + 1e-6) * wn[i].float()
        s = y.abs().amax(-1, keepdim=True) / 448.0 + 1e-8
        q = (y / s).to(torch.float8_e4m3fn)
        z = (q.to(torch.float32) * s).to(torch.bfloat16)
        x = torch.nn.functional.linear(z, Ws[i if (i % 2 == 0) else i + L])
    return x, r


cm = torch.compile(model, mode=None, fullgraph=True, dynamic=False)
x0 = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
r0 = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
t0 = time.time()
with torch.no_grad():
    ref = model(x0, r0)
    out = cm(x0, r0)
    torch.cuda.synchronize()
print(f"PDL={pdl} compile+first run {time.time()-t0:.1f}s; max|compiled-eager| {(out[0].float()-ref[0].float()).abs().max().item():.3e}", flush=True)

g = torch.cuda.CUDAGraph()
with torch.no_grad():
    for _ in range(3):
        cm(x0, r0)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        go = cm(x0, r0)
for _ in range(5):
    g.replay()
torch.cuda.synchronize()
nk = None
try:
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        g.replay(); torch.cuda.synchronize()
    ks = sorted([e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA], key=lambda e: e.time_range.start)
    nk = len(ks)
    ov = sum(1 for a, b in zip(ks, ks[1:]) if b.time_range.start < a.time_range.end)
    gaps = [b.time_range.start - a.time_range.end for a, b in zip(ks, ks[1:]) if b.time_range.start >= a.time_range.end]
    print(f"PDL={pdl}: consecutive kernel pairs {nk-1}, overlapping {ov}, mean non-overlapped gap {sum(gaps)/max(len(gaps),1):.2f} us")
    for a in ks[:12]:
        print(f"   {a.time_range.start:12.2f} +{a.time_range.elapsed_us():6.2f} {a.name[:90]}")
except Exception as e:
    print("profiler failed", e)
ts = []
for _ in range(15):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(200):
        g.replay()
    e1.record(); torch.cuda.synchronize()
    ts.append(e0.elapsed_time(e1) * 1000 / 200)
print(f"PDL={pdl} L={L} M={M}: graph replay median {statistics.median(ts):.1f} us (min {min(ts):.1f}), kernels in graph {nk}, "
      f"{statistics.median(ts)/(nk or 1):.2f} us/kernel", flush=True)
