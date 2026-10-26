"""Our actual MoE path (FlashInfer CUTLASS NVFP4 fused MoE as vLLM v0.32.0 runs it), layer 19, real hidden states and
real routing, timed (a) like kernels/moe-nvfp4/bench_tactics.py (same weights every call -> small batches stay in the
128 MB L2) and (b) with the weights rotated through 6 distinct copies (cold). Reuses kernels/moe-nvfp4/moe_common.py
read-only. Whole fused-MoE call (quantized input precomputed), CUDA graph, median us."""
import sys

import torch

sys.dont_write_bytecode = True
sys.path.insert(0, "/moe")
from common import bench_graph, gpu_state, Out  # noqa: E402
import glob  # noqa: E402
import moe_common  # noqa: E402
moe_common._SNAP = glob.glob("/hf/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/*/")[0]  # container user cannot read /root
from moe_common import E, H, load_hidden, load_layer_raw, prep_cutlass, quant_input, route, run_cutlass  # noqa: E402
from flashinfer.autotuner import autotune  # noqa: E402

L = 19
raw = load_layer_raw(L)
prep = prep_cutlass(raw)
R = 6
copies = [prep] + [{**prep, **{k: prep[k].clone() for k in ("w13", "s13", "w2", "s2")}} for _ in range(R - 1)]
xall = load_hidden(L)
out = Out("ours_cold.jsonl")
print("GPU", gpu_state(), flush=True)
for T, seeds in ((3, (0, 1, 2)), (12, (0, 1, 2)), (48, (0, 1, 2)), (192, (0, 1, 2)), (8192, (0,))):
    for seed in seeds:
        if T == 8192:
            x = xall.contiguous()
        else:
            g = torch.Generator().manual_seed(seed)
            x = xall[torch.randperm(xall.shape[0], generator=g)[:T].cuda()].contiguous()
        wts, ids = route(x, raw["gate"])
        xq = quant_input(x, prep["a1_gscale"])
        o = torch.empty(T, H, device="cuda", dtype=torch.bfloat16)
        cnt = torch.bincount(ids.view(-1).long(), minlength=E)
        with autotune(True):
            run_cutlass(prep, x, ids, wts, out=o, xq=xq)
        torch.cuda.synchronize()
        f = lambda c: run_cutlass(c, x, ids, wts, out=o, xq=xq)
        warm = bench_graph([lambda: f(copies[0])] * 5, iters=30)
        cold = bench_graph([lambda i=i: f(copies[i]) for i in range(R)], iters=30)
        rec = dict(T=T, seed=seed, active=int((cnt > 0).sum()), max_count=int(cnt.max()), warm_us=warm[0], cold_us=cold[0],
                   cold_min=cold[1], cold_max=cold[2], gpu=gpu_state())
        out.write(**rec)
        print(rec, flush=True)
