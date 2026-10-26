"""Independent verification (verifier, not the author) of the SM120 skinny FP8 GEMM patch.

Runs the *shipped* op (torch.ops.vllm.sm120_fp8_scaled_mm from patch/b, bind-mounted over the image)
against the stock path (vllm._custom_ops.cutlass_scaled_mm, which is what stock apply_scaled_mm calls).

Part A  correctness: real weights of layers NOT used by the author's test (4,5,11,22,31,35 + lm_head),
        every M in 1..64 plus a few fallbacks, fresh seeds; bitwise-equal fraction, max diff vs vLLM,
        max abs err vs fp32 reference for both kernels; batch invariance; edge cases (M=0, huge
        activations, non-contiguous A, misaligned A).
Part B  timing: a full decode-step chain of all 80 dense layer GEMMs (distinct real weights of all
        40 layers, in model order, 1.28 GB, so weights are cold) plus the 3 lm_head calls of an MTP-2
        step, each GEMM preceded by its per-token quant kernel (same on both sides). One CUDA graph per
        side, warmup 5, median of 50 replays. Baseline and new are measured alternately (ABAB) to
        cancel clock/thermal drift. Also: the author's per-shape rotation methodology for a few shapes.
Usage (patch files mounted, VLLM_SM120_SKINNY_FP8=1): see VERIFY.md.
"""
import os
import statistics
import sys

import torch

assert os.environ.get("VLLM_SM120_SKINNY_FP8") == "1"
sys.path.insert(0, "/work/kernels/dense-fp8-gemm")
from common import SHAPES, load_weight, make_act, quant_per_token  # noqa: E402

import vllm.model_executor.kernels.linear.scaled_mm.sm120_skinny_fp8 as sk  # noqa: E402
from vllm import _custom_ops as ops  # noqa: E402

op = torch.ops.vllm.sm120_fp8_scaled_mm
PART = sys.argv[1] if len(sys.argv) > 1 else "AB"
torch.manual_seed(0)


def base_mm(a, b, sa, sb):
    return ops.cutlass_scaled_mm(a, b, sa, sb, torch.bfloat16)


# ----------------------------------------------------------------------------------------------- A
if "A" in PART:
    print("=== Part A: correctness of shipped op vs stock cutlass_scaled_mm ===", flush=True)
    worst = dict(bitEq=100.0, maxd=0.0, rel_new=0.0, rel_vllm=0.0, new_worse=0)
    for sn in SHAPES:
        layers = [None] if sn == "lm_head" else ([4, 5, 22, 33] if sn.startswith("gdn") else [11, 35])
        for ly in layers:
            w, s = load_weight(sn, ly)
            s = s.view(-1, 1)
            wt = w.t()
            N, K = w.shape
            line = []
            for M in list(range(1, 65)) + [65, 96, 192]:
                x = make_act(M, K, seed=777 + 31 * M + (ly or 0))
                a, a_s = quant_per_token(x)
                b0 = base_mm(a, wt, a_s, s)
                n0 = op(a, wt, a_s, s)
                ref = (a.float() @ w.float().t()) * a_s.view(-1, 1) * s.view(1, -1)
                eq = (n0.view(torch.int16) == b0.view(torch.int16)).float().mean().item() * 100
                md = (n0.float() - b0.float()).abs().max().item()
                rm = ref.abs().max().item()
                en = (n0.float() - ref).abs().max().item() / rm
                ev = (b0.float() - ref).abs().max().item() / rm
                # mean abs error vs reference (catches systematic bias that max-err would hide)
                mn = (n0.float() - ref).abs().mean().item()
                mv = (b0.float() - ref).abs().mean().item()
                worst["bitEq"] = min(worst["bitEq"], eq)
                worst["maxd"] = max(worst["maxd"], md)
                worst["rel_new"] = max(worst["rel_new"], en)
                worst["rel_vllm"] = max(worst["rel_vllm"], ev)
                if mn > mv * 1.01:
                    worst["new_worse"] += 1
                if eq < 99.9 or M in (1, 3, 4, 16, 48, 64, 65, 192):
                    line.append(f"M={M}:eq{eq:.2f}%,md{md:.3g},mae {mn:.3g}/{mv:.3g}")
                if M > 64 or sk._pick(M, N, K) is None:
                    b1 = base_mm(a, wt, a_s, s)
                    self_eq = (b1.view(torch.int16) == b0.view(torch.int16)).float().mean().item() * 100
                    if not (torch.equal(n0, b0) and torch.equal(b1, b0)):
                        line.append(f"FALLBACK M={M}: op-vs-stock eq{eq:.3f}% stock-vs-stock eq{self_eq:.3f}%")
            print(f"{sn:9s} L{ly}: " + " | ".join(line), flush=True)
            # batch invariance: row r of an M-batch equals the same row computed alone (M=1)
            x = make_act(48, K, seed=4242)
            a, a_s = quant_per_token(x)
            full = op(a, wt, a_s, s)
            inv = all(torch.equal(op(a[r:r + 1].contiguous(), wt, a_s[r:r + 1].contiguous(), s)[0], full[r])
                      for r in (0, 7, 8, 47))
            inv_s = all(torch.equal(base_mm(a[r:r + 1].contiguous(), wt, a_s[r:r + 1].contiguous(), s)[0],
                                    base_mm(a, wt, a_s, s)[r]) for r in (0, 7, 8, 47))
            inv16 = all(torch.equal(op(a[r:r + 1].contiguous(), wt, a_s[r:r + 1].contiguous(), s)[0],
                                    op(a[:16].contiguous(), wt, a_s[:16].contiguous(), s)[r]) for r in (0, 7, 8, 15))
            # run-to-run determinism of both
            det_n = all(torch.equal(op(a, wt, a_s, s), full) for _ in range(5))
            ref48 = base_mm(a, wt, a_s, s)
            det_s = all(torch.equal(base_mm(a, wt, a_s, s), ref48) for _ in range(5))
            print(f"   batch-invariant rows M=48 vs M=1: patched {inv}, stock {inv_s}; patched M=16 vs M=1: {inv16}"
                  f" | deterministic x5 at M=48: patched {det_n}, stock {det_s}", flush=True)
            del w
    print("WORST", worst, flush=True)

    # edge cases on one shape
    w, s = load_weight("gdn_out", 0)
    s = s.view(-1, 1)
    wt = w.t()
    K = w.shape[1]
    # M = 0
    a0 = torch.empty(0, K, dtype=torch.float8_e4m3fn, device="cuda")
    s0 = torch.empty(0, 1, dtype=torch.float32, device="cuda")
    for name, fn in (("stock", base_mm), ("patched", op)):
        try:
            o = fn(a0, wt, s0, s)
            torch.cuda.synchronize()
            print(f"edge M=0 {name}: ok shape {tuple(o.shape)}")
        except Exception as e:  # noqa: BLE001
            print(f"edge M=0 {name}: RAISES {type(e).__name__}: {str(e)[:160]}")
    # saturated activations: every element at +-448 with alternating sign pattern
    a = (torch.randint(0, 2, (16, K), device="cuda").float() * 2 - 1).mul(448).to(torch.float8_e4m3fn)
    sa = torch.full((16, 1), 3.0, device="cuda")
    d = (op(a, wt, sa, s).float() - base_mm(a, wt, sa, s).float()).abs().max().item()
    print(f"edge saturated A: maxdiff vs stock {d:.3g}")
    # non-contiguous A (column slice of a wider buffer) -> must fall back, not crash or read garbage
    big = torch.randn(4, K + 128, device="cuda").to(torch.float8_e4m3fn)
    an = big[:, :K]
    san = torch.ones(4, 1, device="cuda")
    try:
        d = (op(an, wt, san, s).float() - base_mm(an, wt, san, s).float()).abs().max().item()
        print(f"edge non-contiguous A: maxdiff {d}")
    except Exception as e:  # noqa: BLE001
        print(f"edge non-contiguous A: RAISES {type(e).__name__}: {str(e)[:160]}")
    # misaligned A start (8-byte offset): NOT run. Stock cutlass_scaled_mm itself fails to build the TMA
    # descriptor and then hits an illegal instruction (seen in the first verifier run); the patched op
    # falls back to stock for data_ptr % 16 != 0, so it behaves exactly like stock (both crash).
    del w

# ----------------------------------------------------------------------------------------------- B


def graph_time(fns, iters=50, warmup=5):
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
    return g


def replay_ms(g):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    g.replay()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1)


if "B" in PART:
    print("=== Part B: decode-step chain, all 40 layers' real weights, CUDA graph ===", flush=True)
    W = {}
    order = []
    for ly in range(40):
        if ly % 4 == 3:
            order += [("attn_qkv", ly), ("attn_o", ly)]
        else:
            order += [("gdn_qkvz", ly), ("gdn_out", ly)]
    for key in order:
        w, s = load_weight(*key)
        W[key] = (w.t(), s.view(-1, 1))
    wl, sl = load_weight("lm_head", None)
    W["lm"] = (wl.t(), sl.view(-1, 1))
    print("weights loaded", flush=True)

    def chain(mm, M_layer, M_lm, with_gemm=True, lm=True):
        fns = []
        for key in order:
            b, sb = W[key]
            x = make_act(M_layer, b.shape[0], seed=7 * key[1] + len(key[0]))

            def f(x=x, b=b, sb=sb):
                a, sa = quant_per_token(x)
                if with_gemm:
                    mm(a, b, sa, sb)
            fns.append(f)
        if lm:
            b, sb = W["lm"]
            for Mi in M_lm:
                x = make_act(Mi, 2048, seed=Mi)

                def f(x=x, b=b, sb=sb):
                    a, sa = quant_per_token(x)
                    if with_gemm:
                        mm(a, b, sa, sb)
                fns.append(f)
        return fns

    cases = [("c=1", 4, [3, 1, 1]), ("c=4", 16, [12, 4, 4]), ("c=16", 48, [48, 16, 16]),
             ("c=8", 32, [24, 8, 8]), ("c=64 (fallback)", 192, [192, 64, 64])]
    for name, Ml, Mlm in cases:
        gq = graph_time(chain(base_mm, Ml, Mlm, with_gemm=False))
        gb = graph_time(chain(base_mm, Ml, Mlm))
        gn = graph_time(chain(op, Ml, Mlm))
        gbl = graph_time(chain(base_mm, Ml, Mlm, lm=False))
        gnl = graph_time(chain(op, Ml, Mlm, lm=False))
        tq, tb, tn, tbl, tnl = [], [], [], [], []
        for _ in range(50):  # interleaved ABAB
            tb.append(replay_ms(gb)); tn.append(replay_ms(gn))
            tbl.append(replay_ms(gbl)); tnl.append(replay_ms(gnl))
            tq.append(replay_ms(gq))
        q, b, n, bl, nl = map(statistics.median, (tq, tb, tn, tbl, tnl))
        print(f"{name:16s} layerM={Ml:3d} lmM={Mlm}: quant-only {q*1e3:7.1f} us | stock {b*1e3:8.1f} us | "
              f"patched {n*1e3:8.1f} us | saved {1e3*(b-n):6.1f} us/step "
              f"(layers only: {1e3*(bl-nl):6.1f}, lm_head x3: {1e3*((b-bl)-(n-nl)):6.1f}) "
              f"| stdev stock {1e3*statistics.pstdev(tb):.1f} patched {1e3*statistics.pstdev(tn):.1f}",
              flush=True)
        del gq, gb, gn, gbl, gnl
        torch.cuda.empty_cache()

    # The author's methodology (per-shape back-to-back rotation of weight copies) for spot checks
    print("--- per-shape, author-style rotation (cold weights, back-to-back) ---", flush=True)
    for sn, M in [("gdn_qkvz", 4), ("gdn_out", 4), ("attn_qkv", 4), ("gdn_qkvz", 48), ("attn_qkv", 48),
                  ("lm_head", 3), ("lm_head", 1), ("lm_head", 16)]:
        N, K, _ = SHAPES[sn]
        key = ("lm" if sn == "lm_head" else (sn, 0 if sn.startswith("gdn") else 3))
        b0, sb = W[key]
        R = max(1, min(96, -(-(640 << 20) // (N * K))))
        bs = [b0] + [b0.t().clone().t() for _ in range(R - 1)]
        a, sa = quant_per_token(make_act(M, K, seed=M))
        reps = max(1, 8 // R)
        gb = graph_time([lambda b=b: base_mm(a, b, sa, sb) for b in bs] * reps)
        gn = graph_time([lambda b=b: op(a, b, sa, sb) for b in bs] * reps)
        tb, tn = [], []
        for _ in range(30):
            tb.append(replay_ms(gb)); tn.append(replay_ms(gn))
        cnt = R * reps
        print(f"{sn:9s} M={M:3d}: stock {1e3*statistics.median(tb)/cnt:7.1f} us  patched "
              f"{1e3*statistics.median(tn)/cnt:7.1f} us  (R={R})", flush=True)
        del gb, gn, bs
        torch.cuda.empty_cache()
print("DONE")
