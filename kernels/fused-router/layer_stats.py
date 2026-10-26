"""Per-MoE-layer timing from a node-level nsys trace (kernels inside the CUDA graph, c=1 verify step, M=3):
router GEMM -> topkGating on the main stream, and the shared-expert side branch (aux stream).
usage: python3 -I layer_stats.py <node_c1.sqlite>
Durations from --cuda-graph-trace=node are inflated by ~0.5-1 us per short kernel; gaps are graph-edge latencies.
"""
import collections, sqlite3, sys, statistics as st
db = sqlite3.connect(sys.argv[1])
S = dict(db.execute("SELECT id, value FROM StringIds"))
rows = db.execute("SELECT correlationId, demangledName, gridX, gridY, streamId, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
by = collections.defaultdict(list)
for r in rows:
    by[r[0]].append(r)
sizes = collections.Counter(len(v) for v in by.values())
tn = max(n for n, c in sizes.items() if c >= 20)
steps = [sorted(v, key=lambda r: r[5]) for v in by.values() if len(v) == tn]
print("steps", len(steps), "kernels/step", tn)
acc = collections.defaultdict(list)
for s in steps[20:-20]:
    names = [S[r[1]] for r in s]
    for i, n in enumerate(names):
        if "topkGating" not in n:
            continue
        j = i - 1
        while j >= 0 and not ("wmma" in names[j] and s[j][2] == 8): j -= 1
        rg, tk = s[j], s[i]
        us = lambda a, b: (b - a) / 1e3
        acc["router GEMM (cuBLAS wmma) dur"].append(us(rg[5], rg[6]))
        acc["topkGating dur"].append(us(tk[5], tk[6]))
        acc["gap router end -> topk start"].append(us(rg[6], tk[5]))
        acc["router start -> topk end (span)"].append(us(rg[5], tk[6]))
        # kernels on the main stream between topk and GEMM1: quant, buildExpertMaps, expandInputRows, computeStrides
        k = i + 1
        while "GroupProblemShape" not in names[k]: k += 1
        acc["topk end -> grouped GEMM1 start"].append(us(tk[6], s[k][5]))
        k2 = k + 1
        while "GroupProblemShape" not in names[k2]: k2 += 1
        gemm2_end = s[k2][6]
        # shared-expert side branch: gemvx (shared_expert_gate) .. the sigmoid*out elementwise kernels after it
        g = i - 1
        while g >= 0 and "gemvx" not in names[g]: g -= 1
        e = g + 1
        side = [x for x in s[g:i + 8] if x[4] != tk[4]]
        side_end = max(x[6] for x in side if "elementwise" in S[x[1]] or "gemvx" in S[x[1]])
        acc["shared gate gemvx dur"].append(us(s[g][5], s[g][6]))
        acc["shared branch end -> MoE GEMM2 end (slack)"].append(us(side_end, gemm2_end))
for k, v in acc.items():
    v = sorted(v)
    print(f"{k:46s} n={len(v):5d} median {st.median(v):7.2f} us  p10 {v[len(v)//10]:6.2f}  p90 {v[9*len(v)//10]:6.2f}")
