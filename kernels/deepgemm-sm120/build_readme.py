"""Assemble README.md: narrative below + tables generated from results/*.jsonl by summarize.py."""
import contextlib
import io
import json

import summarize as S

R = S.R


def cap(fn):
    b = io.StringIO()
    with contextlib.redirect_stdout(b):
        fn()
    return b.getvalue().strip()


def graph_table():
    rows = S.load("graph_safety.jsonl") + S.load("graph_safety2.jsonl")
    out = ["| mode | experts G | N x K | replays | bitwise mismatch vs eager | fp32-ref mismatch | error |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['mode']} | {r['G']} | {r['N']}x{r['K']} | {r['replays_done']} | {r['bitwise_mismatch_vs_eager']} | {r['ref_mismatch']} | {r['error']} |")
    return "\n".join(out)


def kernel_table():
    ours = {r["T"]: r["seq"] for r in S.load("ours_kernels.jsonl")[-5:]}
    dg = {}
    for r in S.load("dg_kernels.jsonl"):
        if r.get("bm32_bn", "64") in ("64",) or "bm32_bn" not in r:
            dg[(r["tag"], r["fmt"], r["layout"], r["align"], r["gemm"])] = r["gemm_kernel_us"]
    def og(T):  # ours: the two cutlass device_kernel entries of the call sequence = GEMM1, GEMM2; the rest is glue
        s = ours[T]
        g = [t for n, t in s if "device_kernel" in n]
        return g[0], g[1], sum(t for n, t in s if "device_kernel" not in n)
    rows = ["| T tokens | ours GEMM1 / GEMM2 us (NVFP4 CUTLASS) | ours glue us | DG FP4 (MXF4) labels BM32 GEMM1 / GEMM2 us | DG vs ours (GEMMs) | DG FP8 labels BM32 g1 / g2 us |", "|---|---|---|---|---|---|"]
    for T in (3, 12, 48, 192):
        a, b, glue = og(T)
        d1, d2 = dg[(f"T{T}", "fp4", "labels", 32, 1)], dg[(f"T{T}", "fp4", "labels", 32, 2)]
        f1, f2 = dg[(f"T{T}", "fp8", "labels", 32, 1)], dg[(f"T{T}", "fp8", "labels", 32, 2)]
        rows.append(f"| {T} | {a:.1f} / {b:.1f} = {a+b:.1f} | {glue:.1f} | {d1:.1f} / {d2:.1f} = {d1+d2:.1f} | {100*((d1+d2)/(a+b)-1):+.0f}% | {f1:.1f} / {f2:.1f} |")
    a, b, glue = og(8192)
    d = lambda al, lay: (dg[("T8192", "fp4", lay, al, 1)], dg[("T8192", "fp4", lay, al, 2)])
    for al, lay in ((128, "psum"), (64, "psum"), (128, "labels")):
        d1, d2 = d(al, lay)
        rows.append(f"| 8192 (L19) | {a:.1f} / {b:.1f} = {a+b:.1f} | {glue:.1f} | {lay} al{al}: {d1:.1f} / {d2:.1f} = {d1+d2:.1f} | {100*((d1+d2)/(a+b)-1):+.0f}% | psum al128: {dg[('T8192','fp8','psum',128,1)]:.1f} / {dg[('T8192','fp8','psum',128,2)]:.1f} |")
    return "\n".join(rows)


def bn_table():
    rows = [r for r in S.load("dg_kernels.jsonl") if "bm32_bn" in r and r["fmt"] == "fp4" and r["layout"] == "labels" and r["tag"] != "T8192"]
    out = ["| T | BLOCK_N=64 (upstream) g1+g2 us | BLOCK_N=128 g1+g2 us |", "|---|---|---|"]
    for T in (3, 12, 48, 192):
        v = {}
        for bn in ("64", "128"):
            v[bn] = sum(r["gemm_kernel_us"] for r in rows if r["tag"] == f"T{T}" and r["bm32_bn"] == bn and r["gemm"] in (1, 2))
        out.append(f"| {T} | {v['64']:.1f} | {v['128']:.1f} |")
    return "\n".join(out)


tpl = open(R + "../README.tmpl.md").read()
tables = {"DENSE": cap(S.dense), "PREFILL": cap(S.prefill_summary), "DECODE": cap(S.decode_summary), "OURS": cap(S.ours_vs_dg),
          "GRAPH": graph_table(), "KERNEL": kernel_table(), "BN": bn_table()}
for k, v in tables.items():
    tpl = tpl.replace("{{" + k + "}}", v)
open(R + "../README.md", "w").write(tpl)
print("README.md written", len(tpl))
