"""Per-kernel device time of our FlashInfer CUTLASS NVFP4 MoE call (layer 19, real routing), weights rotated through 6
copies so they are cold, eager calls under torch.profiler (CUPTI). Splits the call into the two grouped GEMMs and the
glue kernels (prefix sums, expand, activation, quant, finalize), so the GEMM part can be compared with DeepGEMM's
grouped GEMM kernels, which have no glue."""
import glob
import sys

import torch

sys.dont_write_bytecode = True
sys.path.insert(0, "/moe")
import moe_common  # noqa: E402
moe_common._SNAP = glob.glob("/hf/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/*/")[0]
from moe_common import E, H, load_hidden, load_layer_raw, prep_cutlass, quant_input, route, run_cutlass  # noqa: E402
from flashinfer.autotuner import autotune  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402
from common import Out  # noqa: E402

L, R = 19, 6
raw = load_layer_raw(L)
prep = prep_cutlass(raw)
copies = [prep] + [{**prep, **{k: prep[k].clone() for k in ("w13", "s13", "w2", "s2")}} for _ in range(R - 1)]
xall = load_hidden(L)
out = Out("ours_kernels.jsonl")
for T, seed in ((3, 0), (12, 0), (48, 0), (192, 0), (8192, 0)):
    if T == 8192:
        x = xall.contiguous()
    else:
        g = torch.Generator().manual_seed(seed)
        x = xall[torch.randperm(xall.shape[0], generator=g)[:T].cuda()].contiguous()
    wts, ids = route(x, raw["gate"])
    xq = quant_input(x, prep["a1_gscale"])
    o = torch.empty(T, H, device="cuda", dtype=torch.bfloat16)
    with autotune(True):
        run_cutlass(prep, x, ids, wts, out=o, xq=xq)
    torch.cuda.synchronize()
    n = 18
    for i in range(6):
        run_cutlass(copies[i], x, ids, wts, out=o, xq=xq)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(n):
            run_cutlass(copies[i % R], x, ids, wts, out=o, xq=xq)
        torch.cuda.synchronize()
    agg = {}
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            a = agg.setdefault(e.name[:140], [0.0, 0])
            a[0] += e.device_time
            a[1] += 1
    evs = sorted([e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA], key=lambda e: e.time_range.start)
    K = len(evs) // n
    seq = [(evs[j].name[:60], sum(evs[i * K + j].device_time for i in range(n)) / n) for j in range(K)]
    print("   call sequence (us):", "; ".join(f"{nm.split('(')[0].split('::')[-1][:28]}={t:.1f}" for nm, t in seq), flush=True)
    rows = sorted(((v[0] / n, v[1] / n, k) for k, v in agg.items()), reverse=True)
    print(f"T={T} per-call kernel time (us), cold rotation:", flush=True)
    tot = 0.0
    for t, c, k in rows[:14]:
        print(f"   {t:8.1f} us x{c:.1f}  {k}", flush=True)
    tot = sum(r[0] for r in rows)
    out.write(T=T, total_kernel_us=tot, seq=seq, kernels=[dict(us=t, n=c, name=k) for t, c, k in rows])
    print(f"   total kernel time {tot:.1f} us", flush=True)
