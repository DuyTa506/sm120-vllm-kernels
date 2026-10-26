"""Same-stream consecutive kernel pairs in the target decode graph: how many overlap in time (negative gap),
which is what programmatic dependent launch (PDL) looks like in an nsys node-level trace.
usage: python3 -I overlap_pairs.py <node_trace.sqlite>"""
import collections, re, sqlite3, statistics, sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
db = sqlite3.connect(sys.argv[1]); S = dict(db.execute("SELECT id, value FROM StringIds"))
by = collections.defaultdict(list)
for s, e, c, n, st in db.execute("SELECT start,end,correlationId,demangledName,streamId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphNodeId IS NOT NULL"):
    by[c].append((s, e, S[n], st))
sizes = collections.Counter(len(v) for v in by.values()); tn = max(n for n, c in sizes.items() if c >= 20)
L = [sorted(v) for v in by.values() if len(v) == tn]
def short(n):
    for pat, lab in [(r"GroupProblemShape|cutlass::device_kernel", "cutlass_grouped_gemm"), (r"enable_sm120_family.*GemmUniversal", "fp8_gemm"),
        (r"wmma_tensorop|splitKreduce", "bf16_wmma_gemm"), (r"gemvx", "gemv"), (r"kernel_mha", "xqa_attn"), (r"gdn_decode_post_conv", "gdn_decode"),
        (r"topkGating", "topkGating"), (r"BuildExpertMaps", "BuildExpertMaps"), (r"expandInputRows", "expandInputRows"), (r"doActivation", "doActivation"),
        (r"computeStrides", "computeStrides"), (r"cvt_fp16_to_fp4", "fp4_quant"), (r"causal_conv1d", "conv1d"), (r"rmsnorm_rope|reshape_and_cache", "rope_kvwrite"),
        (r"triton_", "triton"), (r"elementwise", "elementwise")]:
        if re.search(pat, n): return lab
    return n[:30]
agg = collections.defaultdict(lambda: [0, 0, 0, 0])  # n, n_overlap, sum_overlap_ns, sum_gap_ns
for ks in L:
    by_s = collections.defaultdict(list)
    for k in ks: by_s[k[3]].append(k)
    for v in by_s.values():
        v.sort()
        for a, b in zip(v, v[1:]):
            r = agg[(short(a[2]), short(b[2]))]; r[0] += 1
            if b[0] < a[1]: r[1] += 1; r[2] += a[1] - b[0]
            else: r[3] += b[0] - a[1]
nl = len(L); print(f"{nl} launches of the {tn}-kernel graph; same-stream consecutive pairs by (first, second)")
print("  pairs/launch  overlapped%  mean overlap ns  mean gap ns (non-overlapped)")
for (a, b), (n, no, so, sg) in sorted(agg.items(), key=lambda x: -x[1][0])[:30]:
    print(f"  {n/nl:6.1f}   {100*no/n:6.1f}%   {so/max(no,1):8.0f}   {sg/max(n-no,1):8.0f}   {a} -> {b}")
tot = sum(v[0] for v in agg.values()); tov = sum(v[1] for v in agg.values())
print(f"total same-stream pairs/launch {tot/nl:.0f}, overlapped {tov/nl:.0f} ({100*tov/tot:.0f}%)")
