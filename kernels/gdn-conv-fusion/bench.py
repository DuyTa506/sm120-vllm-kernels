"""Benchmark: fused conv1d+post-conv vs causal_conv1d_fn + fused_post_conv_prep.

Per-call GPU time with CUDA events (eager launches, no CUDA graph: the GDN core op
runs eagerly in vLLM's piecewise graphs), median of N runs after warmup.

Modes
  flush : 256 MB L2-flush write before every timed call (cold L2, like a real layer
          whose input GEMM output (T x 12288 bf16) does not fit in the 128 MB L2)
  chain : 30 back-to-back calls rotating over 4 independent input sets (one
          "request"), time / 30
"""

import argparse
import statistics

import torch

from common import H, K, conv_meta, layer_params, make_conv_states, make_inputs, reference
from fused_conv_post_conv import fused_conv1d_post_conv_prep

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--fused", default="f2g", help="variant used for the chain/headline")
args = ap.parse_args()
FUSED_DEFAULT = args.fused
dev = "cuda"
P = layer_params(0)
flush_buf = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)


def time_call(fn, flush=True):
    ts = []
    for i in range(args.warmup + args.reps):
        if flush:
            flush_buf.random_(0, 255) if i == 0 else flush_buf.add_(1)
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        if i >= args.warmup:
            ts.append(s.elapsed_time(e) * 1e3)
    return statistics.median(ts)


def time_chain(fns, n=30):
    ts = []
    for i in range(args.warmup + max(5, args.reps // 5)):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for j in range(n):
            fns[j % len(fns)]()
        e.record()
        torch.cuda.synchronize()
        if i >= args.warmup:
            ts.append(s.elapsed_time(e) * 1e3 / n)
    return statistics.median(ts)


def setup(lens, init, seed):
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import (
        fused_post_conv_prep,
    )

    T = sum(lens)
    x, a, b = make_inputs(T, seed=seed)
    qsl_cpu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
    qsl = qsl_cpu.to(dev)
    meta = conv_meta(qsl_cpu)
    has_init = torch.tensor(init, dtype=torch.bool, device=dev)
    _, cs = make_conv_states(len(lens) + 4)
    ci = torch.arange(1, len(lens) + 1, dtype=torch.int32, device=dev)
    conv_out = causal_conv1d_fn(x.T, P.conv_w, None, cs, qsl, ci, has_init, metadata=meta).T

    def base():
        reference(x, P.conv_w, cs, qsl, ci, has_init, a, b, P.A_log, P.dt_bias, meta=meta)

    def conv_only():
        causal_conv1d_fn(x.T, P.conv_w, None, cs, qsl, ci, has_init, metadata=meta)

    def post_only():
        fused_post_conv_prep(conv_out, a, b, P.A_log, P.dt_bias, H, K, K, True, True)

    def mk(sub, gf):
        def fused():
            fused_conv1d_post_conv_prep(x, P.conv_w, None, cs, qsl, ci, has_init, a, b, P.A_log,
                                        P.dt_bias, H, K, K, output_g_exp=True, metadata=meta,
                                        sub=sub, group_fastest=gf)
        return fused

    dst = torch.empty(T, 8192, dtype=x.dtype, device=dev)

    def copy_floor():  # strided read of x + contiguous write of the same bytes
        dst.copy_(x)

    d = dict(base=base, conv=conv_only, post=post_only, copy=copy_floor)
    for sub in (1, 2, 4):
        for gf in (True, False):
            d[f"f{sub}{'g' if gf else 'c'}"] = mk(sub, gf)
    d["fused"] = d[FUSED_DEFAULT]
    return d, T


CASES = [
    ("8k single", [8192], [False]),
    ("4k single", [4096], [False]),
    ("16k single", [16384], [False]),
    ("varlen 8k", [3000, 2000, 1500, 1000, 500, 192], [False, True, False, True, True, False]),
    ("varlen 64x128", [128] * 64, [True] * 64),
]
print(f"{'case':16s} {'T':>6s} {'conv':>8s} {'post':>8s} {'base':>8s} {'fused':>8s} {'saved':>8s}"
      f" {'GB/s fused':>10s} {'copy':>7s}   chain(base/fused)")
for name, lens, init in CASES:
    fns, T = setup(lens, init, seed=1)
    r = {k: time_call(f) for k, f in fns.items()}
    print("   variants: " + " ".join(f"{k}={v:.1f}" for k, v in r.items() if k.startswith("f") and k != "fused"))
    sets = [setup(lens, init, seed=s)[0] for s in range(1, 5)]
    cb = time_chain([s["base"] for s in sets])
    cf = time_chain([s["fused"] for s in sets])
    floor_bytes = T * 8192 * 2 * 2 + T * 32 * (2 * 2 + 2 * 4)
    print(f"{name:16s} {T:6d} {r['conv']:8.1f} {r['post']:8.1f} {r['base']:8.1f} {r['fused']:8.1f}"
          f" {r['base'] - r['fused']:8.1f} {floor_bytes / r['fused'] / 1e3:10.0f} {r['copy']:7.1f}   {cb:.1f}/{cf:.1f} us")
    del sets, fns
    torch.cuda.empty_cache()
print(f"times in us per layer-call (median); fused variant for the headline/chain: {FUSED_DEFAULT}; "
      "floor bytes = read x + write q,k,v (+a,b,g,beta); copy = x.copy_ of the same bytes (practical floor)")
