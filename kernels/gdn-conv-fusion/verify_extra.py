"""Verifier's extra checks for the fused conv1d + post-conv kernel (not the claimant's code).

1. Correctness on layouts the claimant's test does not use but the server does:
   - a / b as strided column halves of a [T, 64] `ba` buffer (server: mixed_ba.chunk(2))
   - contiguous x (server spec-decode mixed path: index_select) with 1-token non-spec rows
   - a large conv cache (production-sized num_cache_lines, high line indices)
   - group_fastest=False grid order (used when tot > 65535) with sub 1/2/4
   - 32k-token single sequence (above the max_num_batched_tokens=16384 the server uses)
   - mixed decode+prefill (raw rows) with strided a/b
   - determinism: two fused runs bitwise identical
2. JIT specialization count over a sweep of runtime shapes (compile hazard).
3. Independent timing: torch.profiler device-kernel durations (no launch/event overhead)
   and CUDA-event chains with a 512 MB L2 scrub, server-like strided inputs.
"""

import os
import statistics
import sys
import time

import torch

from common import DIM, H, HV, K, QKVZ, conv_meta, layer_params, make_inputs, reference
from fused_conv_post_conv import _fused_conv_post_conv_kernel, fused_conv1d_post_conv_prep

dev = "cuda"
FAIL = []


def cmp(name, got, ref):
    d = (got.float() - ref.float()).abs()
    mx = d.max().item() if d.numel() else 0.0
    rel = (d.max() / ref.float().abs().max().clamp_min(1e-30)).item() if d.numel() else 0.0
    nd = int((d > 0).sum().item()) if d.numel() else 0
    ok = mx == 0.0
    print(f"    {name:10s} max_abs={mx:.3e} max_rel={rel:.3e} ndiff={nd} {'OK' if ok else 'DIFF'}")
    if not ok:
        FAIL.append(name)


def server_inputs(T, layer=0, seed=0, strided_ab=True, contiguous_x=False):
    x, a, b = make_inputs(T, layer=layer, seed=seed)
    if contiguous_x:
        x = x.contiguous()
    if strided_ab:
        ba = torch.empty(T, 2 * HV, dtype=a.dtype, device=dev)
        ba[:, :HV] = b
        ba[:, HV:] = a
        b, a = ba.chunk(2, dim=-1)  # same as prepare_gdn_attention_core_inputs
        assert a.stride(0) == 2 * HV
    return x, a, b


def run(title, lens, init, *, num_lines=None, strided_ab=True, contiguous_x=False, raw_seqs=0,
        layer=0, gf=True, subs=(1, 2, 4), high_lines=False):
    print(f"[{title}] nseq={len(lens)} T={sum(lens)} gf={gf} strided_ab={strided_ab} "
          f"contig_x={contiguous_x} raw={raw_seqs}")
    P = layer_params(layer)
    T = sum(lens)
    x, a, b = server_inputs(T, layer, seed=T + 7, strided_ab=strided_ab, contiguous_x=contiguous_x)
    qsl_cpu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
    qsl = qsl_cpu.to(dev)
    meta = conv_meta(qsl_cpu)
    has_init = torch.tensor(init, dtype=torch.bool, device=dev)
    nseq = len(lens)
    num_lines = num_lines or (4 * nseq + 8)
    # SD layout cache [lines, 3, dim] filled with realistic rows (scaled randn of the activation scale)
    base = (torch.randn(num_lines, 3, DIM, device=dev) * x.float().std()).to(torch.bfloat16)
    if high_lines:
        ci = (num_lines - 1 - torch.randperm(min(num_lines - 1, 4 * nseq), device=dev)[:nseq]).to(torch.int32)
    else:
        ci = (torch.randperm(num_lines - 1, device=dev)[:nseq] + 1).to(torch.int32)
    raw_tokens = sum(lens[:raw_seqs])
    ref_base = base.clone()
    ref = reference(x, P.conv_w, ref_base.transpose(-1, -2), qsl, ci, has_init, a, b, P.A_log,
                    P.dt_bias, meta=meta, output_g_exp=True, num_raw_tokens=raw_tokens)
    outs = []
    for sub in subs:
        fb = base.clone()
        got = fused_conv1d_post_conv_prep(
            x, P.conv_w, None, fb.transpose(-1, -2), qsl, ci, has_init, a, b, P.A_log, P.dt_bias,
            num_k_heads=H, head_k_dim=K, head_v_dim=K, output_g_exp=True, metadata=meta,
            num_raw_seqs=raw_seqs, num_raw_tokens=raw_tokens, sub=sub, group_fastest=gf)
        torch.cuda.synchronize()
        print(f"  sub={sub}")
        for n, g_, r_ in zip(["q", "k", "v", "g", "beta"], got[:5], ref[:5]):
            cmp(n, g_, r_)
        if raw_seqs:
            cmp("raw", got[5], ref[5])
        cmp("conv_cache", fb, ref_base)
        outs.append((got, fb))
    # determinism: rerun sub=2 and compare bitwise with first sub=2 run
    if 2 in subs:
        fb = base.clone()
        got2 = fused_conv1d_post_conv_prep(
            x, P.conv_w, None, fb.transpose(-1, -2), qsl, ci, has_init, a, b, P.A_log, P.dt_bias,
            num_k_heads=H, head_k_dim=K, head_v_dim=K, output_g_exp=True, metadata=meta,
            num_raw_seqs=raw_seqs, num_raw_tokens=raw_tokens, sub=2, group_fastest=gf)
        first = outs[list(subs).index(2)][0]
        same = all(torch.equal(u, w) for u, w in zip(got2[:5], first[:5]))
        print(f"  determinism (rerun sub=2 bitwise equal): {same}")
        if not same:
            FAIL.append("determinism")


MODE = os.environ.get("MODE", "all")

if MODE in ("all", "correct"):
    run("server layout 8k varlen", [3000, 2000, 1500, 1000, 500, 192],
        [False, True, False, True, True, False])
    run("server layout 8k single init", [8192], [True])
    run("spec-mixed: contiguous x, 1-token non-spec rows, no raw", [1, 1, 700, 1, 333, 2, 5],
        [True, True, False, True, True, True, False], contiguous_x=True)
    run("mixed decode+prefill raw, strided a/b", [1, 1, 1, 1, 1, 1, 4000, 97],
        [True] * 6 + [False, True], raw_seqs=6)
    run("big cache, high line idx", [1500, 900, 64, 3], [True, True, False, True],
        num_lines=3000, high_lines=True)
    run("group_fastest=False", [3001, 1500, 7, 2], [False, True, True, True], gf=False)
    run("32k single", [32768], [True], subs=(2,))
    run("layer 36 (last GDN-ish layer)", [2048, 511, 1], [True, False, True], layer=36, subs=(2,))
    print("CORRECTNESS:", "FAIL " + ",".join(sorted(set(FAIL))) if FAIL else "ALL BIT-EXACT")

if MODE in ("all", "jit"):
    # How many specializations does a realistic stream of prefill shapes create?
    P = layer_params(0)
    lines = 512
    base = torch.zeros(lines, 3, DIM, dtype=torch.bfloat16, device=dev)
    cs = base.transpose(-1, -2)
    shapes = []
    g = torch.Generator().manual_seed(3)
    for i in range(40):
        n = int(torch.randint(1, 5, (1,), generator=g))
        lens = [int(v) for v in torch.randint(1, 3000, (n,), generator=g)]
        if i % 5 == 0:
            lens = [1] * 3 + lens  # decode rows in front
        shapes.append(lens)
    shapes += [[16], [17], [1, 16], [1, 1, 32], [8192], [8191], [1], [2], [3]]

    def nspec():
        c = getattr(_fused_conv_post_conv_kernel, "device_caches", None)
        if c is None:
            return len(getattr(_fused_conv_post_conv_kernel, "cache", {}).get(0, {}))
        return sum(len(v[0]) for v in c.values())

    x_all, a_all, b_all = server_inputs(16384, seed=5)
    t_first = []
    for lens in shapes:
        T = sum(lens)
        raw_seqs = 3 if lens[:3] == [1, 1, 1] and len(lens) > 3 else 0
        qsl_cpu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
        meta = conv_meta(qsl_cpu)
        ci = torch.arange(1, len(lens) + 1, dtype=torch.int32, device=dev)
        hi = torch.ones(len(lens), dtype=torch.bool, device=dev)
        before = nspec()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fused_conv1d_post_conv_prep(
            x_all[:T], P.conv_w, None, cs, qsl_cpu.to(dev), ci, hi, a_all[:T], b_all[:T], P.A_log,
            P.dt_bias, H, K, K, output_g_exp=True, metadata=meta, num_raw_seqs=raw_seqs,
            num_raw_tokens=sum(lens[:raw_seqs]))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if nspec() > before:
            t_first.append(dt)
    print(f"JIT: {len(shapes)} shapes -> {nspec()} specializations; compile-call times (s): "
          + " ".join(f"{t:.2f}" for t in t_first))

if MODE in ("all", "time"):
    from torch.profiler import ProfilerActivity, profile

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import (
        fused_post_conv_prep,
    )

    P = layer_params(0)
    scrub = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=dev)

    def mk(lens, init, seed):
        T = sum(lens)
        x, a, b = server_inputs(T, seed=seed)
        qsl_cpu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
        qsl = qsl_cpu.to(dev)
        meta = conv_meta(qsl_cpu)
        hi = torch.tensor(init, dtype=torch.bool, device=dev)
        base = torch.zeros(len(lens) + 4, 3, DIM, dtype=torch.bfloat16, device=dev)
        cs = base.transpose(-1, -2)
        ci = torch.arange(1, len(lens) + 1, dtype=torch.int32, device=dev)

        def base_fn():  # exactly the server's two calls
            out = causal_conv1d_fn(x.transpose(0, 1), P.conv_w, None, activation="silu",
                                   conv_states=cs, has_initial_state=hi, cache_indices=ci,
                                   query_start_loc=qsl, metadata=meta).transpose(0, 1)
            fused_post_conv_prep(conv_output=out, a=a, b=b, A_log=P.A_log, dt_bias=P.dt_bias,
                                 num_k_heads=H, head_k_dim=K, head_v_dim=K, apply_l2norm=True,
                                 output_g_exp=True)

        def fused_fn():
            fused_conv1d_post_conv_prep(x, P.conv_w, None, cs, qsl, ci, hi, a, b, P.A_log,
                                        P.dt_bias, num_k_heads=H, head_k_dim=K, head_v_dim=K,
                                        apply_l2norm=True, output_g_exp=True, metadata=meta)
        return base_fn, fused_fn

    def chain(fns, n=30, reps=25, warm=5):
        ts = []
        for i in range(warm + reps):
            scrub.fill_(i & 0xFF)
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for j in range(n):
                fns[j % len(fns)]()
            e.record()
            torch.cuda.synchronize()
            if i >= warm:
                ts.append(s.elapsed_time(e) * 1e3 / n)
        return statistics.median(ts), min(ts), max(ts)

    def prof_kernels(fn, iters=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(iters):
                scrub.fill_(1)
                fn()
            torch.cuda.synchronize()
        agg = {}
        for ev in p.events():
            if ev.device_type.name == "CUDA" and "fill" not in ev.name.lower():
                agg.setdefault(ev.name, []).append(ev.time_range.elapsed_us())
        return {k: (statistics.median(v), len(v)) for k, v in agg.items()}

    for name, lens, init in [
        ("8k single", [8192], [False]),
        ("8k varlen", [3000, 2000, 1500, 1000, 500, 192], [False, True, False, True, True, False]),
        ("4k single", [4096], [False]),
        ("16k single", [16384], [False]),
        ("2k single", [2048], [False]),
        ("512 single", [512], [False]),
    ]:
        sets = [mk(lens, init, s) for s in range(4)]
        bk = prof_kernels(sets[0][0])
        fk = prof_kernels(sets[0][1])
        print(f"[{name}] profiler device time per kernel (median us, count over 20 iters):")
        for k_, (v, c) in list(bk.items()) + list(fk.items()):
            print(f"    {k_[:60]:60s} {v:8.1f} x{c}")
        cb = chain([s[0] for s in sets])
        cf = chain([s[1] for s in sets])
        print(f"    chain30 (rotating 4 input sets, warm L2 between calls): base {cb[0]:.1f} "
              f"[{cb[1]:.1f}-{cb[2]:.1f}]  fused {cf[0]:.1f} [{cf[1]:.1f}-{cf[2]:.1f}] us/call")
        del sets
        torch.cuda.empty_cache()

sys.exit(1 if FAIL else 0)
