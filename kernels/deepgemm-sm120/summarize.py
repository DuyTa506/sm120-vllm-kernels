"""Turn results/*.jsonl into markdown tables (printed to stdout)."""
import json
import re
import statistics
import sys
from collections import defaultdict

R = "/home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/deepgemm-sm120/results/"


def load(n):
    try:
        return [json.loads(l) for l in open(R + n)]
    except FileNotFoundError:
        return []


def tile(r):
    k = r.get("kernel", "")
    m = re.search(r"impl<([^>]*)>", k)
    if m:
        t = [x.strip().rstrip("u") for x in m.group(1).split(",")]
        try:
            return f"{t[6]}x{t[7]}x{t[8]} st{t[12]}" + (" swapAB" if r.get("api_swap_ab") else "")
        except IndexError:
            pass
    if "block_m" in r:
        return f"{r['block_m']}x{r['block_n']}x{r['block_k']} st{r['stages']}" + (" swap" if r.get("api_swap_ab") else "")
    return "?"


def dense():
    rows = load("dense.jsonl") + load("dense2.jsonl")
    by = defaultdict(lambda: defaultdict(list))
    cfg = {}
    for r in rows:
        by[(r["shape"], r["M"])][r["impl"]].append(r["us"])
        if r["impl"] == "dg":
            cfg[(r["shape"], r["M"])] = r
    order = ["gdn_qkvz", "gdn_out", "attn_qkv", "attn_o", "lm_head"]
    print("| shape (N x K) | M | DeepGEMM us | DG+PDL us | vLLM CUTLASS us | cuBLASLt us | our skinny us | DG / vLLM | DG TFLOPS | vLLM TFLOPS | DG GB/s | DG tile (BMxBNxBK, stages) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    dims = {"gdn_qkvz": (12288, 2048), "gdn_out": (2048, 4096), "attn_qkv": (9216, 2048), "attn_o": (2048, 4096), "lm_head": (248320, 2048)}
    for (sh, M), d in sorted(by.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1])):
        N, K = dims[sh]
        m = {k: statistics.mean(v) for k, v in d.items()}
        f = lambda k: f"{m[k]:.1f}" if k in m else "-"
        fl = 2 * M * N * K
        tf = lambda k: f"{fl / m[k] / 1e6:.0f}" if k in m else "-"
        gb = (N * K + M * K + 2 * M * N) / m["dg"] / 1e3
        print(f"| {sh} ({N}x{K}) | {M} | {f('dg')} | {f('dg_pdl')} | {f('vllm')} | {f('cublaslt')} | {f('skinny')} | {m['dg']/m['vllm']:.2f} | {tf('dg')} | {tf('vllm')} | {gb:.0f} | {tile(cfg[(sh, M)])} |")


def grouped():
    rows = load("grouped.jsonl")
    print("\n### prefill (contiguous), us per GEMM")
    pre = [r for r in rows if "routing" in r]
    keyf = lambda r: (r["fmt"], r["routing"], r["gemm"], r["kind"], r["layout"], r["align"])
    for r in sorted(pre, key=keyf):
        print(f"{r['fmt']:3s} {r['routing']:8s} g{r['gemm']} {r['kind']:15s} {r['layout']:6s} al={r['align']:3d} m_total={r['m_total']:6d} pad x{r['m_total']/r['rows_valid']:.2f} "
              f"{r['us']:8.1f} us {r['tflops_useful']:6.0f} TFLOPS(useful) wTB/s={r['w_TBps']:.2f} {tile(r)}")
    print("\n### decode")
    dec = [r for r in rows if "T" in r]
    for r in sorted(dec, key=lambda r: (r["fmt"], r["gemm"], r["T"], r["kind"], r["layout"], r.get("align", 0), r["seed"])):
        print(f"{r['fmt']:3s} g{r['gemm']} T={r['T']:3d} seed{r['seed']} active={r['active']:3d} max={r['max_count']:3d} {r['kind']:15s} {r['layout']:7s} al={r.get('align','-')} em={r.get('expected_m','-')} "
              f"{r['us']:8.1f} us wTB/s={r['w_TBps']:.2f} {tile(r)}")


if __name__ == "__main__" and sys.argv[1] in ("dense", "grouped"):
    {"dense": dense, "grouped": grouped}[sys.argv[1]]()


def decode_summary():
    rows = [r for r in load("grouped.jsonl") + load("grouped_fp4.jsonl") if "T" in r]
    def avg(sel):
        v = [r["us"] for r in rows if sel(r)]
        return statistics.mean(v) if v else float("nan")
    print("| T tokens (x top-8) | active experts (seed 0/1/2) | variant | GEMM1 us | GEMM2 us | sum us | weight TB/s (g1) |")
    print("|---|---|---|---|---|---|---|")
    for T in (3, 12, 48, 192):
        act = "/".join(str(next(r["active"] for r in rows if r["T"] == T and r["seed"] == s and r["gemm"] == 1)) for s in (0, 1, 2))
        variants = [("fp8", "labels32", "FP8 contiguous labels, align 32 (BM=32)"), ("fp8", "psum32", "FP8 contiguous psum, align 32 (BM=32)"),
                    ("fp8", "loop", "FP8 loop of cuBLASLt _scaled_mm per active expert"),
                    ("fp4", "labels32", "FP4 (MXF4) contiguous labels, align 32 (BM=32)"), ("fp4", "psum32", "FP4 (MXF4) contiguous psum, align 32 (BM=32)")]
        for fmt, lay, name in variants:
            g1 = avg(lambda r: r["T"] == T and r["fmt"] == fmt and r["layout"] == lay and r["gemm"] == 1)
            g2 = avg(lambda r: r["T"] == T and r["fmt"] == fmt and r["layout"] == lay and r["gemm"] == 2)
            tb = statistics.mean([r["w_TBps"] for r in rows if r["T"] == T and r["fmt"] == fmt and r["layout"] == lay and r["gemm"] == 1])
            print(f"| {T} | {act} | {name} | {g1:.1f} | {g2:.1f} | {g1+g2:.1f} | {tb:.2f} |")
        for fmt, al in (("fp8", 32), ("fp8", 64), ("fp8", 128), ("fp4", 32), ("fp4", 64), ("fp4", 128)):
            sel = lambda r, g: r["T"] == T and r["fmt"] == fmt and r["layout"] == "masked" and r["align"] == al and r["gemm"] == g and r["expected_m"] <= 8 * 1 + 200
            # take the variant with the smallest expected_m per (align) to avoid mixing forced-large-tile cases
            def best(g):
                v = [r for r in rows if r["T"] == T and r["fmt"] == fmt and r["layout"] == "masked" and r["align"] == al and r["gemm"] == g]
                if not v:
                    return float("nan"), float("nan")
                em = min(r["expected_m"] for r in v)
                v = [r for r in v if r["expected_m"] == em]
                return statistics.mean(r["us"] for r in v), statistics.mean(r["w_TBps"] for r in v)
            (g1, tb), (g2, _) = best(1), best(2)
            if g1 == g1:
                print(f"| {T} | {act} | {fmt.upper()} masked, align {al}{' (BM=32)' if al == 32 else ''} | {g1:.1f} | {g2:.1f} | {g1+g2:.1f} | {tb:.2f} |")


def prefill_summary():
    rows = [r for r in load("grouped.jsonl") + load("grouped_fp4.jsonl") if "routing" in r]
    print("| routing | variant | GEMM1 us | GEMM2 us | sum us | padded rows | useful TFLOPS g1 / g2 | g1 tile | g2 tile |")
    print("|---|---|---|---|---|---|---|---|---|")
    for rn in ("L19", "L0", "uniform"):
        for fmt, kind, lay, al in [("fp8", "contiguous", "labels", 128), ("fp8", "contiguous", "psum", 128), ("fp8", "contiguous", "labels", 64),
                                   ("fp8", "contiguous", "psum", 64), ("fp8", "contiguous", "labels", 32), ("fp8", "contiguous", "psum", 32),
                                   ("fp4", "contiguous", "labels", 128), ("fp4", "contiguous", "psum", 128), ("fp4", "contiguous", "psum", 64),
                                   ("fp8", "loop_scaled_mm", "loop", 16)]:
            g = {}
            for gi in (1, 2):
                v = [r for r in rows if r["routing"] == rn and r["fmt"] == fmt and r["kind"] == kind and r["layout"] == lay and r["align"] == al and r["gemm"] == gi]
                if v:
                    g[gi] = v[0]
            if len(g) == 2:
                name = {"loop_scaled_mm": "FP8 cuBLASLt loop (per active expert)"}.get(kind, f"{fmt.upper()}{' (MXF4 proxy)' if fmt=='fp4' else ''} {lay} align {al}")
                print(f"| {rn} | {name} | {g[1]['us']:.0f} | {g[2]['us']:.0f} | {g[1]['us']+g[2]['us']:.0f} | x{g[1]['m_total']/g[1]['rows_valid']:.2f} | "
                      f"{g[1]['tflops_useful']:.0f} / {g[2]['tflops_useful']:.0f} | {tile(g[1]) if kind!='loop_scaled_mm' else '-'} | {tile(g[2]) if kind!='loop_scaled_mm' else '-'} |")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("decode_summary", "prefill_summary"):
    {"decode_summary": decode_summary, "prefill_summary": prefill_summary}[sys.argv[1]]()


def ours_vs_dg():
    ours = load("ours_cold.jsonl")
    rows = [r for r in load("grouped.jsonl") + load("grouped_fp4.jsonl") if "T" in r]
    pre = [r for r in load("grouped.jsonl") + load("grouped_fp4.jsonl") if "routing" in r]
    def dg(T, fmt, lay, al=None):
        out = []
        for g in (1, 2):
            v = [r for r in rows if r["T"] == T and r["fmt"] == fmt and r["layout"] == lay and r["gemm"] == g and (al is None or r["align"] == al)]
            if lay == "masked" and v:
                em = min(r["expected_m"] for r in v)
                v = [r for r in v if r["expected_m"] == em]
            if not v:
                return None
            out.append(statistics.mean(r["us"] for r in v))
        return sum(out)
    print("| T | active experts | ours warm (bench_tactics style) | ours cold (6 weight copies) | DG FP4 masked BM32 | DG FP4 contiguous labels BM32 | DG FP4 contiguous psum BM32 | DG FP8 contiguous labels BM32 | DG FP8 masked BM32 |")
    print("|---|---|---|---|---|---|---|---|---|")
    f = lambda x: f"{x:.1f}" if x else "-"
    for T in (3, 12, 48, 192):
        o = [r for r in ours if r["T"] == T]
        act = "/".join(str(r["active"]) for r in o)
        print(f"| {T} | {act} | {statistics.mean(r['warm_us'] for r in o):.1f} | {statistics.mean(r['cold_us'] for r in o):.1f} | {f(dg(T,'fp4','masked',32))} | {f(dg(T,'fp4','labels32'))} | {f(dg(T,'fp4','psum32'))} | {f(dg(T,'fp8','labels32'))} | {f(dg(T,'fp8','masked',32))} |")
    o = [r for r in ours if r["T"] == 8192][0]
    def pf(fmt, lay, al):
        s = 0
        for g in (1, 2):
            v = [r for r in pre if r["routing"] == "L19" and r["fmt"] == fmt and r["kind"] == "contiguous" and r["layout"] == lay and r["align"] == al and r["gemm"] == g]
            if not v:
                return None
            s += v[0]["us"]
        return s
    print(f"| 8192 | {o['active']} | {o['warm_us']:.1f} | {o['cold_us']:.1f} | n/a | FP4 labels al128: {f(pf('fp4','labels',128))} / psum al128: {f(pf('fp4','psum',128))} / psum al64: {f(pf('fp4','psum',64))} | | FP8 psum al128: {f(pf('fp8','psum',128))} | |")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "ours_vs_dg":
    ours_vs_dg()
