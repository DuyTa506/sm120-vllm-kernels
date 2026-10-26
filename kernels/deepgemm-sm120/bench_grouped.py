"""M-grouped GEMM (MoE expert GEMMs) with DeepGEMM nv_dev SM120 kernels on the REAL routing of Qwen3.6-35B-A3B.

Expert shapes (config.json): hidden 2048, moe_intermediate 512, 256 experts, top-8.
  GEMM1 gate/up: N = 2*512 = 1024, K = 2048     GEMM2 down: N = 2048, K = 512
Prefill: m-grouped CONTIGUOUS layout (labels or psum), 8192 tokens -> 65536 routed rows (layer 19 trace: hottest
expert 6007 rows, 243 active). Decode: m-grouped MASKED layout and contiguous with 32-row alignment (BLOCK_M=32).
FP8 is the scheduler/tile proxy (UE8M0 1x128 scales, NOT our NVFP4 contract); an MXF4 (block 32) run is included
only as a second proxy for the FP4 MMA path. Baseline: loop of cuBLASLt rowwise torch._scaled_mm per active expert.
"""
import argparse
import math
import re

import torch

import deep_gemm
from common import Out, bench_graph, calc_diff, capture_fd1, fp8_rand, gemm_kernel, gpu_state, pow2_sf

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="prefill,decode")
ap.add_argument("--out", default="grouped.jsonl")
ap.add_argument("--warm-only", action="store_true")
ap.add_argument("--slice", default="0/1")
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--routings", default="L19,L0,uniform")
ap.add_argument("--aligns", default="32,64,128")
ap.add_argument("--no-fp4", action="store_true")
ap.add_argument("--fmts", default="fp8,fp4")
ap.add_argument("--no-ref", action="store_true")
args = ap.parse_args()

G = 256
GEMMS = {1: (1024, 2048), 2: (2048, 512)}
si, sn = map(int, args.slice.split("/"))
out = Out(args.out) if not args.warm_only else None
_case = [-1]


def mine():
    _case[0] += 1
    return _case[0] % sn == si


def rup(x, a):
    return (x + a - 1) // a * a


# ---------------------------------------------------------------- routing
def load_routing():
    d = torch.load("/moe/data/moe_inputs.pt", map_location="cpu", weights_only=False)
    ids = {}
    for L in (0, 19):
        x = d["hidden"][L].cuda()
        gw = d["gate"][L].cuda()
        logits = torch.nn.functional.linear(x, gw).float()
        w, i = torch.softmax(logits, -1).topk(8, -1)
        ids[L] = i.to(torch.int32).cpu()
    return ids


ids = load_routing()
cnt = {f"L{L}": torch.bincount(ids[L].view(-1).long(), minlength=G) for L in ids}
cnt["uniform"] = torch.full((G,), 256, dtype=torch.long)
for k, v in cnt.items():
    print(f"routing {k}: total {int(v.sum())} max {int(v.max())} active {int((v > 0).sum())}", flush=True)


def decode_counts(T, seed):
    g = torch.Generator().manual_seed(seed)
    sel = torch.randperm(ids[19].shape[0], generator=g)[:T]
    return torch.bincount(ids[19][sel].view(-1).long(), minlength=G)


# ---------------------------------------------------------------- operands
def make_weights(N, K, fmt, R):
    if fmt == "fp8":
        b = torch.empty(G, N, K, dtype=torch.float8_e4m3fn, device="cuda")
        for s in range(0, G, 16):
            b[s:s + 16] = fp8_rand((16, N, K))
        sf = pow2_sf((G, N, K // 128), -9, -6)
    else:
        b = torch.randint(-128, 128, (G, N, K // 2), dtype=torch.int8, device="cuda")
        sf = pow2_sf((G, N, K // 32), -4, -1)
    sfp = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)
    copies = [(b, sfp)] + [(b.clone(), sfp.clone()) for _ in range(R - 1)]
    return copies, sf


def make_rows(rows, K, fmt, lead=()):
    shape = (*lead, rows, K)
    if fmt == "fp8":
        a = fp8_rand(shape)
        sf = pow2_sf((*lead, rows, K // 128), -3, 1)
    else:
        a = torch.randint(-128, 128, (*lead, rows, K // 2), dtype=torch.int8, device="cuda")
        sf = pow2_sf((*lead, rows, K // 32), -3, 1)
    return a, sf


def contig_layout(counts, al):
    starts, ends, pos = [], [], 0
    for c in counts.tolist():
        pos = rup(pos, al)
        starts.append(pos)
        pos += c
        ends.append(pos)
    m_total = rup(pos, al)
    labels = torch.full((m_total,), -1, dtype=torch.int32)
    for g, (s, e) in enumerate(zip(starts, ends)):
        labels[s:e] = g
    return starts, ends, m_total, labels.cuda(), torch.tensor(ends, dtype=torch.int32).cuda()


CFG_RE = re.compile(r"SM120GemmConfig\(layout=Layout\(swap_ab=(\d), block_m=(\d+), block_n=(\d+), block_k=(\d+).*?num_stages=(\d+)")


def parse_cfg(txt):
    m = CFG_RE.search(txt)
    if not m:
        return {"cfg_raw": txt[-200:]}
    return dict(zip(("layout_swap", "block_m", "block_n", "block_k", "stages"), map(int, m.groups())))


def ref_contig(a, sfa, b_sf_pair, starts, ends, counts, N, K, fmt, w):
    """fp32 reference rows for a few active experts. w = (b [G,N,K] fp8, sf float [G,N,K/128])."""
    b, sf = w
    worst = 0.0
    act = [g for g in range(G) if counts[g] > 0]
    return act


def timeit(fns, reps):
    return bench_graph(fns * reps, iters=args.iters)


def emit(**kw):
    kw["gpu"] = gpu_state()
    out.write(**kw)
    print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in kw.items() if k not in ("gpu", "cfg_raw")}, flush=True)


# ---------------------------------------------------------------- prefill
def run_prefill():
    als = [int(x) for x in args.aligns.split(",")]
    for fmt in args.fmts.split(","):
        if fmt == "fp4" and args.no_fp4:
            continue
        gran = 128 if fmt == "fp8" else 32
        for gi, (N, K) in GEMMS.items():
            # (routing, align, layout) cases decided first so unused weights are not generated in warm slices
            cases = []
            for rname in args.routings.split(","):
                for al in (als if fmt == "fp8" else [a for a in als if a >= 64]):
                    for lm in ("labels", "psum"):
                        if fmt == "fp4" and rname == "L0":
                            continue
                        cases.append((rname, al, lm))
            cases = [c for c in cases if mine()]
            if not cases:
                continue
            R = 1 if args.warm_only else 2
            wc, wsf = make_weights(N, K, fmt, R)
            wch = pow2_sf((G, N), -9, -6) if fmt == "fp8" else None
            bw_per_expert = N * K * (1 if fmt == "fp8" else 0.5)
            for rname, al, lm in cases:
                counts = cnt[rname]
                deep_gemm.set_mk_alignment_for_contiguous_layout(al)
                starts, ends, m_total, labels, psum = contig_layout(counts, al)
                a, sfa = make_rows(m_total, K, fmt)
                sfa_p = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sfa, psum if lm == "psum" else None)
                d = torch.empty(m_total, N, dtype=torch.bfloat16, device="cuda")
                exp_m = int(counts.float().mean().item())
                meta = labels if lm == "labels" else psum

                def call(i, d=d):
                    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                        (a, sfa_p), wc[i], d, meta, recipe_a=(1, gran), recipe_b=(1, gran), use_psum_layout=(lm == "psum"),
                        ensure_zero_padding=False, **({"expected_m_for_psum_layout": exp_m} if lm == "psum" else {}))

                tag = f"{fmt} gemm{gi} {rname} align={al} {lm}"
                print(f"CASE prefill {tag} m_total={m_total}", flush=True)
                _, txt = capture_fd1(lambda: call(0))
                cfg = parse_cfg(txt)
                if args.warm_only:
                    continue
                cfg['kernel'] = gemm_kernel(lambda: call(0))
                valid = int(counts.sum())
                diff = None
                if fmt == "fp8" and not args.no_ref:
                    worst = 0.0
                    act = [g for g in range(G) if counts[g] > 0]
                    for g in act[:: max(1, len(act) // 6)]:
                        s, e = starts[g], ends[g]
                        a_deq = a[s:e].float() * sfa[s:e].repeat_interleave(128, 1)
                        b_deq = wc[0][0][g].float() * wsf[g].repeat_interleave(128, 1)
                        worst = max(worst, calc_diff(d[s:e].float(), a_deq @ b_deq.T))
                    diff = worst
                else:
                    diff = None
                us, lo, hi = timeit([lambda i=i: call(i) for i in range(R)], 2)
                act_n = int((counts > 0).sum())
                fl_u, fl_p = 2 * valid * N * K, 2 * m_total * N * K
                wbytes = act_n * bw_per_expert
                emit(kind="contiguous", fmt=fmt, routing=rname, gemm=gi, N=N, K=K, layout=lm, align=al, m_total=m_total,
                     rows_valid=valid, active=act_n, max_count=int(counts.max()), us=us, us_min=lo, us_max=hi,
                     tflops_useful=fl_u / us / 1e6, tflops_padded=fl_p / us / 1e6, w_TBps=wbytes / us / 1e6,
                     pad_overhead=m_total / valid, calc_diff=diff, **cfg)
                if lm == "labels" and al == 128 and fmt == "fp8":
                    # baseline: loop of cuBLASLt rowwise _scaled_mm per active expert (rows padded to 16)
                    act = [g for g in range(G) if counts[g] > 0]
                    r16 = [rup(int(counts[g]), 16) for g in act]
                    offs = [0]
                    for r in r16:
                        offs.append(offs[-1] + r)
                    ab = fp8_rand((offs[-1], K))
                    asc = pow2_sf((offs[-1], 1), -3, 1)

                    def loop(i):
                        for j, g in enumerate(act):
                            torch._scaled_mm(ab[offs[j]:offs[j + 1]], wc[i][0][g].t(), scale_a=asc[offs[j]:offs[j + 1]],
                                             scale_b=wch[g].view(1, -1), out_dtype=torch.bfloat16)
                    try:
                        loop(0)
                        us, lo, hi = timeit([lambda i=i: loop(i) for i in range(R)], 1)
                        emit(kind="loop_scaled_mm", fmt="fp8", routing=rname, gemm=gi, N=N, K=K, layout="loop", align=16,
                             m_total=offs[-1], rows_valid=valid, active=act_n, max_count=int(counts.max()), us=us, us_min=lo,
                             us_max=hi, tflops_useful=fl_u / us / 1e6, w_TBps=wbytes / us / 1e6)
                    except Exception as e:  # noqa: BLE001
                        print("  loop FAILED", type(e).__name__, str(e)[:120], flush=True)
                    del ab, asc
                del a, sfa, sfa_p, d, labels, psum
                torch.cuda.empty_cache()
            del wc
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- decode
def run_decode():
    Ts = (3, 12, 48, 192)
    for fmt in args.fmts.split(","):
        if fmt == "fp4" and args.no_fp4:
            continue
        gran = 128 if fmt == "fp8" else 32
        for gi, (N, K) in GEMMS.items():
            cases = []
            for T in Ts:
                for seed in (0, 1, 2):
                    if fmt == "fp4" and (seed > 0):
                        continue
                    for lm in ("masked", "labels32", "psum32"):
                        cases.append((T, seed, lm))
            cases = [c for c in cases if mine()]
            if not cases:
                continue
            R = 1 if args.warm_only else 6
            wc, wsf = make_weights(N, K, fmt, R)
            wch = pow2_sf((G, N), -9, -6) if fmt == "fp8" else None
            bw_per_expert = N * K * (1 if fmt == "fp8" else 0.5)
            for T, seed, lm in cases:
                counts = decode_counts(T, seed)
                act = [g for g in range(G) if counts[g] > 0]
                act_n, valid, cmax = len(act), int(counts.sum()), int(counts.max())
                mean = valid / G
                if lm == "masked":
                    em_mean = max(1, math.ceil(1.2 * mean))
                    th = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout
                    variants = sorted({(th(em_mean), em_mean), (th(cmax), cmax), (64, em_mean), (128, em_mean)})
                    for al_m, exp_m in variants:
                        deep_gemm.set_mk_alignment_for_contiguous_layout(al_m)
                        max_m = T
                        a, sfa = make_rows(max_m, K, fmt, lead=(G,))
                        sfa_p = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sfa)
                        mm = counts.to(torch.int32).cuda()
                        d = torch.empty(G, max_m, N, dtype=torch.bfloat16, device="cuda")

                        def call(i, d=d, a=a, sfa_p=sfa_p, mm=mm, exp_m=exp_m):
                            deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked((a, sfa_p), wc[i], d, mm, exp_m,
                                                                      recipe_a=(1, gran), recipe_b=(1, gran))

                        tag = f"decode {fmt} gemm{gi} T={T} seed={seed} masked exp_m={exp_m} align={al_m}"
                        print("CASE", tag, flush=True)
                        _, txt = capture_fd1(lambda: call(0))
                        cfg = parse_cfg(txt)
                        if args.warm_only:
                            continue
                        cfg['kernel'] = gemm_kernel(lambda: call(0))
                        diff = None
                        if fmt == "fp8" and not args.no_ref:
                            worst = 0.0
                            for g in act[:4]:
                                c = int(counts[g])
                                a_deq = a[g, :c].float() * sfa[g, :c].repeat_interleave(128, 1)
                                b_deq = wc[0][0][g].float() * wsf[g].repeat_interleave(128, 1)
                                worst = max(worst, calc_diff(d[g, :c].float(), a_deq @ b_deq.T))
                            diff = worst
                        us, lo, hi = timeit([lambda i=i: call(i) for i in range(R)], 1)
                        fl_u = 2 * valid * N * K
                        emit(kind="masked", fmt=fmt, T=T, seed=seed, gemm=gi, N=N, K=K, layout="masked", align=al_m, expected_m=exp_m,
                             max_m=max_m, rows_valid=valid, active=act_n, max_count=cmax, us=us, us_min=lo, us_max=hi,
                             tflops_useful=fl_u / us / 1e6, w_TBps=act_n * bw_per_expert / us / 1e6, calc_diff=diff, **cfg)
                        del a, sfa, sfa_p, d
                else:
                    al = 32
                    deep_gemm.set_mk_alignment_for_contiguous_layout(al)
                    use_psum = lm == "psum32"
                    starts, ends, m_total, labels, psum = contig_layout(counts, al)
                    a, sfa = make_rows(m_total, K, fmt)
                    sfa_p = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sfa, psum if use_psum else None)
                    d = torch.empty(m_total, N, dtype=torch.bfloat16, device="cuda")
                    meta = psum if use_psum else labels
                    exp_m = max(1, math.ceil(1.2 * mean))

                    def call(i, d=d, a=a, sfa_p=sfa_p):
                        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                            (a, sfa_p), wc[i], d, meta, recipe_a=(1, gran), recipe_b=(1, gran), use_psum_layout=use_psum,
                            ensure_zero_padding=False, **({"expected_m_for_psum_layout": exp_m} if use_psum else {}))

                    print("CASE", f"decode {fmt} gemm{gi} T={T} seed={seed} {lm} m_total={m_total}", flush=True)
                    _, txt = capture_fd1(lambda: call(0))
                    cfg = parse_cfg(txt)
                    if args.warm_only:
                        continue
                    cfg['kernel'] = gemm_kernel(lambda: call(0))
                    diff = None
                    if fmt == "fp8" and not args.no_ref:
                        worst = 0.0
                        for g in act[:4]:
                            s, e = starts[g], ends[g]
                            a_deq = a[s:e].float() * sfa[s:e].repeat_interleave(128, 1)
                            b_deq = wc[0][0][g].float() * wsf[g].repeat_interleave(128, 1)
                            worst = max(worst, calc_diff(d[s:e].float(), a_deq @ b_deq.T))
                        diff = worst
                    us, lo, hi = timeit([lambda i=i: call(i) for i in range(R)], 1)
                    emit(kind="contiguous", fmt=fmt, T=T, seed=seed, gemm=gi, N=N, K=K, layout=lm, align=al, m_total=m_total,
                         rows_valid=valid, active=act_n, max_count=cmax, us=us, us_min=lo, us_max=hi,
                         tflops_useful=2 * valid * N * K / us / 1e6, w_TBps=act_n * bw_per_expert / us / 1e6, calc_diff=diff, **cfg)
                    del a, sfa, sfa_p, d, labels, psum
                if lm == "masked" and fmt == "fp8" and seed in (0, 1, 2) and not args.warm_only:
                    r16 = [rup(int(counts[g]), 16) for g in act]
                    offs = [0]
                    for r in r16:
                        offs.append(offs[-1] + r)
                    ab = fp8_rand((offs[-1], K))
                    asc = pow2_sf((offs[-1], 1), -3, 1)

                    def loop(i):
                        for j, g in enumerate(act):
                            torch._scaled_mm(ab[offs[j]:offs[j + 1]], wc[i][0][g].t(), scale_a=asc[offs[j]:offs[j + 1]],
                                             scale_b=wch[g].view(1, -1), out_dtype=torch.bfloat16)
                    try:
                        loop(0)
                        us, lo, hi = timeit([lambda i=i: loop(i) for i in range(R)], 1)
                        emit(kind="loop_scaled_mm", fmt="fp8", T=T, seed=seed, gemm=gi, N=N, K=K, layout="loop", rows_valid=valid,
                             active=act_n, max_count=cmax, us=us, us_min=lo, us_max=hi,
                             tflops_useful=2 * valid * N * K / us / 1e6, w_TBps=act_n * bw_per_expert / us / 1e6)
                    except Exception as e:  # noqa: BLE001
                        print("  loop FAILED", type(e).__name__, str(e)[:120], flush=True)
                    del ab, asc
                torch.cuda.empty_cache()
            del wc
            torch.cuda.empty_cache()


print("GPU", torch.cuda.get_device_name(0), "SMs", deep_gemm.get_num_sms(), "pdl", deep_gemm.get_pdl(), gpu_state(), flush=True)
for m in args.mode.split(","):
    {"prefill": run_prefill, "decode": run_decode}[m]()
print("done", gpu_state(), flush=True)
