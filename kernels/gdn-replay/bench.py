"""Microbenchmark in a CUDA graph over many distinct layers/states (defeats the L2).
usage: bench.py [B ...]  -> JSON lines on stdout, a table at the end. Needs the GPU lock."""
import json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import vllm  # noqa: F401
from vllm import _custom_ops as ops
from common import *

ext = build()
dev = "cuda"
params = real_params()
gen = torch.Generator(device=dev); gen.manual_seed(0)
STATE_BYTES = HV * V * K * 4          # 2 MiB per request per slot
BUDGET = float(os.environ.get("BUDGET_GB", "3.2")) * 2**30


def timed(fn, iters=30, warm=5):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(g, stream=s):
        fn()
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b) * 1000)  # us
    return statistics.median(ts), min(ts)


def ptrs(ts):
    return torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)


def run_B(B, n_dist=(0.2, 0.15, 0.65)):
    S = 1 + B * W
    per_layer = S * STATE_BYTES + B * W * (HV * V + 2 * HV * K + HV) * 4
    L = int(max(2, min(256, BUDGET // per_layer)))
    touched = L * B * STATE_BYTES
    qlens = [3] * B
    inp = Inputs(qlens, gen)
    si = window_indices(B)
    acc = torch.full((B,), 3, dtype=torch.int32, device=dev)
    pn = torch.tensor(n_dist, device=dev)
    n_new = (torch.multinomial(pn, B, replacement=True, generator=gen) + 1).to(torch.int32)
    states = [torch.randn(S, HV, V, K, generator=gen, device=dev) * 0.1 for _ in range(L)]
    reps = [(torch.zeros(B, W, HV, V, device=dev), torch.zeros(B, W, H, K, device=dev), torch.zeros(B, W, HV, device=dev)) for _ in range(L)]
    repd = [(torch.zeros(B, W, HV, V, device=dev), torch.zeros(B, W, HV, K, device=dev), torch.zeros(B, W, HV, device=dev)) for _ in range(L)]
    markers = [torch.zeros(B, dtype=torch.int32, device=dev) for _ in range(L)]
    epoch = torch.zeros(1, dtype=torch.int32, device=dev)
    strides = torch.full((L,), states[0].stride(0), dtype=torch.int64, device=dev)
    A, dt, nw = params
    out = inp.out
    margs = lambda st: (inp.mixed_qkv, inp.a, inp.b, A, dt, si, inp.cu, acc, st, inp.gate, nw, out)
    fold = torch.full((B,), 3, dtype=torch.int32, device=dev)
    ones = torch.ones(B, dtype=torch.int32, device=dev)
    sp, dp, kp, ep, mp = ptrs(states), ptrs([r[0] for r in reps]), ptrs([r[1] for r in reps]), ptrs([r[2] for r in reps]), ptrs(markers)
    res = {"B": B, "layers": L, "touched_MB": touched / 2**20}

    def base():
        for st in states:
            ext.base_decode(*margs(st))

    def prod():
        for st in states:
            ops.fused_gdn_decode_post_conv_mtp(inp.mixed_qkv, inp.a, inp.b, A, dt, si, inp.cu, acc, st, inp.gate, nw, out, SCALE, EPS, "silu")

    def verify():
        for st, r, mk in zip(states, reps, markers):
            ext.verify(*margs(st), rep=r, marker=mk, epoch=epoch)

    def commit():
        ext.commit(sp, dp, kp, ep, strides, si, inp.cu, acc, n_new, k_heads=H, marker_ptrs=mp, epoch=epoch)

    def deferred():
        for st, r in zip(states, repd):
            ext.deferred(*(margs(st)[:7] + (ones,) + margs(st)[8:]), rep=r, fold_n=fold)

    for name, fn, per in (("prod", prod, L), ("base", base, L), ("verify", verify, L), ("commit", commit, L), ("deferred", deferred, L)):
        med, mn = timed(fn)
        res[name + "_us"] = med / per
        res[name + "_min_us"] = mn / per
    res["verify+commit_us"] = res["verify_us"] + res["commit_us"]
    # bytes: base 1R+3W, verify 1R, commit 1R+1W, deferred 1R+1W   (state bytes only)
    mb = B * STATE_BYTES / 1e6
    res["prod_TBs"] = 4 * mb / res["prod_us"]
    res["base_TBs"] = 4 * mb / res["base_us"]
    res["verify_TBs"] = 1 * mb / res["verify_us"]
    res["commit_TBs"] = 2 * mb / res["commit_us"]
    res["deferred_TBs"] = 2 * mb / res["deferred_us"]
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    return res


def bandwidth_ref():
    x = torch.empty(2**29, dtype=torch.float32, device=dev)  # 2 GiB
    x.normal_()
    y = torch.empty_like(x)
    r = timed(lambda: x.sum(), iters=10)[0]
    c = timed(lambda: y.copy_(x), iters=10)[0]
    res = {"read_TBs": x.numel() * 4 / r / 1e6, "copy_TBs(r+w)": 2 * x.numel() * 4 / c / 1e6}
    print(json.dumps({k: round(v, 3) for k, v in res.items()}), flush=True)
    del x, y


if __name__ == "__main__":
    print(torch.cuda.get_device_name(), flush=True)
    bandwidth_ref()
    Bs = [int(a) for a in sys.argv[1:]] or [1, 2, 4, 8, 16, 32, 64]
    rows = [run_B(B) for B in Bs]
    print("\n us per layer; saved = production op - (verify+commit) / - deferred; x30 = ms per 30-layer step")
    print(" B | L | prod | base copy | verify | commit | v+c | saved | x30 ms | deferred | saved | x30 ms | TB/s prod, verify, commit")
    for r in rows:
        print(f"{r['B']:3d}|{r['layers']:4d}| {r['prod_us']:7.1f} | {r['base_us']:7.1f} | {r['verify_us']:7.1f} | {r['commit_us']:7.1f} | {r['verify+commit_us']:7.1f} | "
              f"{r['prod_us']-r['verify+commit_us']:7.1f} | {30*(r['prod_us']-r['verify+commit_us'])/1000:6.2f} | {r['deferred_us']:7.1f} | {r['prod_us']-r['deferred_us']:7.1f} | {30*(r['prod_us']-r['deferred_us'])/1000:6.2f} | "
              f"{r['prod_TBs']:.2f}, {r['verify_TBs']:.2f}, {r['commit_TBs']:.2f}")
