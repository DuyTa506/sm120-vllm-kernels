"""CUDA-graph safety of DeepGEMM nv_dev SM120 m-grouped kernels (issue deepseek-ai/DeepGEMM#444).

Capture a graph ONCE with routing A (static buffers: A/SFA/B/SFB/D and the routing metadata tensor), then replay it
N times, each time overwriting the metadata IN PLACE with a different per-expert token distribution (real L19 routing,
uniform, one/few hot experts, empty experts, different totals, all empty, random Dirichlet skew), D pre-filled with NaN.
After every replay: synchronize (an illegal memory access shows up here), compare the valid rows bitwise with an
eager call on the same inputs (separate output buffer), and every 50th replay also with an fp32 reference of 3 experts.
"""
import argparse
import sys
import time

import torch

import deep_gemm
from common import Out, calc_diff, fp8_rand, gpu_state, pow2_sf

ap = argparse.ArgumentParser()
ap.add_argument("--replays", type=int, default=1000)
ap.add_argument("--out", default="graph_safety.jsonl")
ap.add_argument("--modes", default="fp8_labels128,fp8_labels32,fp8_psum128,fp8_masked,fp8_labels128_zp,fp8_labels128_pdl,fp4_labels128,fp8_labels128_G288")
args = ap.parse_args()
out = Out(args.out)


def rup(x, a):
    return (x + a - 1) // a * a


def routing_pool(G, T_rows_total):
    """list of (name, counts[G]) for a prefill of T tokens x top8 = T_rows_total routed rows; each expert <= T tokens."""
    T = T_rows_total // 8
    g = torch.Generator().manual_seed(123)
    pool = []
    base = torch.bincount(torch.load("/work/data/ids_L19.pt").view(-1).long(), minlength=256)
    if G != 256 or T_rows_total != 65536:
        base = base.repeat(2)[:G].clone()
        base = (base.float() * (T_rows_total / base.sum())).long().clamp(max=T)
    pool.append(("real_L19", base))
    pool.append(("uniform", torch.full((G,), T_rows_total // G, dtype=torch.long)))
    hot = torch.zeros(G, dtype=torch.long)
    hot[:8] = T
    pool.append(("8_hot_experts_only", hot))
    one = torch.zeros(G, dtype=torch.long)
    one[G - 1] = T
    one[0] = T_rows_total - T if T_rows_total - T <= T else T
    pool.append(("two_experts_last_first", one))
    half = torch.zeros(G, dtype=torch.long)
    half[::2] = T_rows_total // (G // 2)
    pool.append(("even_experts_only", half))
    small = torch.zeros(G, dtype=torch.long)
    small[3] = 1
    small[200 % G] = 2
    pool.append(("tiny_total_3_rows", small))
    pool.append(("all_empty", torch.zeros(G, dtype=torch.long)))
    pool.append(("decode_like_12tok", torch.bincount(torch.randint(0, G, (96,), generator=g), minlength=G)))
    for i in range(40):
        p = torch.distributions.Dirichlet(torch.full((G,), 0.15 + 0.1 * (i % 5))).sample()
        c = torch.multinomial(p, T_rows_total, replacement=True, generator=g)
        c = torch.bincount(c, minlength=G).clamp(max=T)
        pool.append((f"dirichlet{i}", c))
    return pool


def contig_meta(counts, al, m_cap, psum):
    pos, labels, ends = 0, torch.full((m_cap,), -1, dtype=torch.int32), []
    for g, c in enumerate(counts.tolist()):
        pos = rup(pos, al)
        labels[pos:pos + c] = g
        pos += c
        ends.append(pos)
    assert rup(pos, al) <= m_cap, (pos, m_cap)
    valid = labels >= 0
    return (torch.tensor(ends, dtype=torch.int32) if psum else labels), valid, labels


def run_mode(mode):
    fmt = "fp4" if mode.startswith("fp4") else "fp8"
    gran = 128 if fmt == "fp8" else 32
    G = 288 if "G288" in mode else 256
    if "w2" in mode:
        N, K = (4096, 256) if G == 288 else (2048, 512)
    else:
        N, K = (512, 4096) if G == 288 else (1024, 2048)
    masked = "masked" in mode
    psum = "psum" in mode
    al = 32 if ("labels32" in mode or "masked" in mode) else 128
    zp = mode.endswith("_zp")
    pdl = mode.endswith("_pdl")
    rows_total = 65536 if not masked else 8 * 192  # masked: decode-sized batches, max_m = 192 tokens
    T = rows_total // 8
    deep_gemm.set_pdl(pdl)
    deep_gemm.set_mk_alignment_for_contiguous_layout(al)
    pool = routing_pool(G, rows_total)
    if masked:
        max_m = T
        pool = [(n, c.clamp(max=max_m)) for n, c in pool]
    m_cap = (rows_total + G * al + al) if not masked else max_m
    m_cap = rup(m_cap, 128)
    # static buffers
    if fmt == "fp8":
        b = torch.empty(G, N, K, dtype=torch.float8_e4m3fn, device="cuda")
        for s in range(0, G, 16):
            b[s:s + 16] = fp8_rand((min(16, G - s), N, K))
        bsf = pow2_sf((G, N, K // 128), -9, -6)
        a = fp8_rand((G, max_m, K) if masked else (m_cap, K))
        asf = pow2_sf(((G, max_m) if masked else (m_cap,)) + (K // 128,), -3, 1)
    else:
        b = torch.randint(-128, 128, (G, N, K // 2), dtype=torch.int8, device="cuda")
        bsf = pow2_sf((G, N, K // 32), -4, -1)
        a = torch.randint(-128, 128, (m_cap, K // 2), dtype=torch.int8, device="cuda")
        asf = pow2_sf((m_cap, K // 32), -3, 1)
    bsf_p = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(bsf)
    meta0, valid0, _ = (None, None, None)
    if masked:
        meta = torch.zeros(G, dtype=torch.int32, device="cuda")
        asf_p = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(asf)
        shape_d = (G, max_m, N)
    else:
        m0, _, _ = contig_meta(pool[0][1], al, m_cap, psum)
        meta = m0.cuda()
        asf_p = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(asf, meta if psum else None)
        shape_d = (m_cap, N)
    d_g = torch.empty(shape_d, dtype=torch.bfloat16, device="cuda")
    d_e = torch.empty(shape_d, dtype=torch.bfloat16, device="cuda")
    exp_m = 256 if not masked else 8
    mean_m = 256

    def call(d):
        if masked:
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked((a, asf_p), (b, bsf_p), d, meta, exp_m, recipe_a=(1, gran), recipe_b=(1, gran))
        else:
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                (a, asf_p), (b, bsf_p), d, meta, recipe_a=(1, gran), recipe_b=(1, gran), use_psum_layout=psum,
                ensure_zero_padding=zp, **({"expected_m_for_psum_layout": mean_m} if psum else {}))

    def set_routing(counts):
        if masked:
            meta.copy_(counts.to(torch.int32))
            valid = torch.arange(max_m)[None, :] < counts[:, None]
            return valid.cuda(), counts
        mt, valid, labels = contig_meta(counts, al, m_cap, psum)
        meta.copy_(mt.cuda())
        if psum:  # production: the activation quant kernel re-emits SFA for the current routing every step
            asf_p.copy_(deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(asf, meta))
        return valid.cuda(), labels

    # eager warmup + capture with routing A (pool[0]); same stream discipline as PyTorch docs
    set_routing(pool[0][1])
    call(d_e)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call(d_g)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        call(d_g)
    n_ok = n_bit_bad = n_ref_bad = 0
    t0 = time.time()
    err = None
    names = []
    worst_ref = 0.0
    for it in range(args.replays):
        name, counts = pool[it % len(pool)] if it >= len(pool) else pool[(it * 7) % len(pool)]
        valid, labels = set_routing(counts)
        d_g.fill_(float("nan"))
        d_e.fill_(float("nan"))
        try:
            graph.replay()
            torch.cuda.synchronize()
            call(d_e)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:200]} at replay {it} routing={name}"
            break
        vg, ve = d_g[valid], d_e[valid]
        nv = int(valid.sum())
        if nv and (not torch.equal(vg.view(torch.int16), ve.view(torch.int16)) or not bool(torch.isfinite(vg.float()).all())):
            n_bit_bad += 1
            names.append(name)
        if it % 50 == 0 and fmt == "fp8" and nv:
            g_list = (counts > 0).nonzero().view(-1)[:3].tolist()
            for g in g_list:
                if masked:
                    c = int(counts[g])
                    ad = a[g, :c].float() * asf[g, :c].repeat_interleave(128, 1)
                    got = d_g[g, :c].float()
                else:
                    idx = (labels == g).nonzero().view(-1)
                    s0, e0 = int(idx[0]), int(idx[-1]) + 1
                    ad = a[s0:e0].float() * asf[s0:e0].repeat_interleave(128, 1)
                    got = d_g[s0:e0].float()
                bd = b[g].float() * bsf[g].repeat_interleave(128, 1)
                df = calc_diff(got, ad @ bd.T)
                worst_ref = max(worst_ref, df)
                if df > 1e-3:
                    n_ref_bad += 1
        n_ok += 1
    el = time.time() - t0
    rec = dict(mode=mode, G=G, N=N, K=K, replays_done=n_ok, replays_requested=args.replays, routings=len(pool),
               bitwise_mismatch_vs_eager=n_bit_bad, ref_mismatch=n_ref_bad, worst_calc_diff_vs_fp32=worst_ref,
               error=err, bad_routings=sorted(set(names))[:10], seconds=el, gpu=gpu_state())
    out.write(**rec)
    print(rec, flush=True)
    del graph
    deep_gemm.set_pdl(False)
    if err:
        print("FATAL context poisoned after error; stopping", flush=True)
        sys.exit(3)


for m in args.modes.split(","):
    print("MODE", m, flush=True)
    run_mode(m)
    torch.cuda.empty_cache()
