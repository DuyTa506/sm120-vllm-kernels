"""Per-kernel device time (CUPTI) of the DeepGEMM grouped GEMMs on the same routing as bench_ours_kernels.py, weights
rotated through 6 copies (2 for prefill). Includes any helper kernel the API launches (e.g. padding clear, SF copies)."""
import math
import os
import sys

import torch

import deep_gemm
from common import Out, fp8_rand, pow2_sf

sys.dont_write_bytecode = True
from torch.profiler import ProfilerActivity, profile  # noqa: E402

G = 256
GEMMS = {1: (1024, 2048), 2: (2048, 512)}
d = torch.load("/moe/data/moe_inputs.pt", map_location="cpu", weights_only=False)
x = d["hidden"][19].cuda()
_, ids = torch.softmax(torch.nn.functional.linear(x, d["gate"][19].cuda()).float(), -1).topk(8, -1)
ids = ids.int().cpu()
out = Out("dg_kernels.jsonl")


def rup(x, a):
    return (x + a - 1) // a * a


def layout(counts, al):
    starts, ends, pos = [], [], 0
    for c in counts.tolist():
        pos = rup(pos, al)
        starts.append(pos)
        pos += c
        ends.append(pos)
    m = rup(pos, al)
    lab = torch.full((m,), -1, dtype=torch.int32)
    for g, (s, e) in enumerate(zip(starts, ends)):
        lab[s:e] = g
    return m, lab.cuda(), torch.tensor(ends, dtype=torch.int32).cuda()


def weights(N, K, fmt, R):
    if fmt == "fp8":
        b = torch.empty(G, N, K, dtype=torch.float8_e4m3fn, device="cuda")
        for s in range(0, G, 16):
            b[s:s + 16] = fp8_rand((16, N, K))
        sf = pow2_sf((G, N, K // 128), -9, -6)
    else:
        b = torch.randint(-128, 128, (G, N, K // 2), dtype=torch.int8, device="cuda")
        sf = pow2_sf((G, N, K // 32), -4, -1)
    p = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)
    return [(b, p)] + [(b.clone(), p.clone()) for _ in range(R - 1)]


def run(tag, fmt, gi, counts, lay, al, R, n=12):
    N, K = GEMMS[gi]
    gran = 128 if fmt == "fp8" else 32
    wc = weights(N, K, fmt, R)
    deep_gemm.set_mk_alignment_for_contiguous_layout(al)
    m, lab, ps = layout(counts, al)
    if fmt == "fp8":
        a = fp8_rand((m, K)); sfa = pow2_sf((m, K // 128), -3, 1)
    else:
        a = torch.randint(-128, 128, (m, K // 2), dtype=torch.int8, device="cuda"); sfa = pow2_sf((m, K // 32), -3, 1)
    sfp = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sfa, ps if lay == "psum" else None)
    dd = torch.empty(m, N, dtype=torch.bfloat16, device="cuda")
    em = max(1, int(counts.float().mean().item()))

    def call(i):
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous((a, sfp), wc[i], dd, ps if lay == "psum" else lab, recipe_a=(1, gran), recipe_b=(1, gran),
                                                      use_psum_layout=(lay == "psum"), ensure_zero_padding=False,
                                                      **({"expected_m_for_psum_layout": em} if lay == "psum" else {}))
    for i in range(R):
        call(i)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(n):
            call(i % R)
        torch.cuda.synchronize()
    agg = {}
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            agg[e.name[:90]] = agg.get(e.name[:90], 0.0) + e.device_time
    tot = sum(agg.values()) / n
    gem = sum(v for k, v in agg.items() if "gemm" in k.lower()) / n
    out.write(bm32_bn=os.environ.get("DG_FORCE_BM32_BN", "64"), tag=tag, fmt=fmt, gemm=gi, layout=lay, align=al, m_total=m, kernel_total_us=tot, gemm_kernel_us=gem)
    return gem, tot


for T, seed in (((3, 0), (12, 0), (48, 0), (192, 0)) if "--decode-only" in sys.argv else ((3, 0), (12, 0), (48, 0), (192, 0), (8192, 0))):
    if T == 8192:
        counts = torch.bincount(ids.view(-1).long(), minlength=G)
        cases = [("fp4", "psum", 128), ("fp4", "psum", 64), ("fp4", "labels", 128), ("fp8", "psum", 128)]
        R = 2
    else:
        g = torch.Generator().manual_seed(seed)
        sel = torch.randperm(ids.shape[0], generator=g)[:T]
        counts = torch.bincount(ids[sel].view(-1).long(), minlength=G)
        cases = [("fp4", "labels", 32), ("fp4", "psum", 32), ("fp8", "labels", 32)] if "--decode-only" not in sys.argv else [("fp4", "labels", 32), ("fp8", "labels", 32)]
        R = 6
    for fmt, lay, al in cases:
        r = [run(f"T{T}", fmt, gi, counts, lay, al, R) for gi in (1, 2)]
        print(f"T={T} {fmt} {lay} al={al}: gemm kernels g1 {r[0][0]:.1f} + g2 {r[1][0]:.1f} = {r[0][0]+r[1][0]:.1f} us (all kernels {r[0][1]+r[1][1]:.1f})", flush=True)
