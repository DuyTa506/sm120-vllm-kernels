"""PDL needs BOTH ends: the consumer launched with the programmatic attribute AND the producer calling
launch_dependents early. Count, per decode step (target graph of a node-level trace), the same-stream kernel
boundaries whose two kernels are both PDL-capable under each scenario, and how many of them do not overlap today.
usage: python3 -I pdl_edges.py <node_trace.sqlite>"""
import collections, re, sqlite3, sys
db = sqlite3.connect(sys.argv[1]); S = dict(db.execute("SELECT id, value FROM StringIds"))
by = collections.defaultdict(list)
for s, e, c, n, st in db.execute("SELECT start,end,correlationId,demangledName,streamId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphNodeId IS NOT NULL"):
    by[c].append((s, e, S[n], st))
for tbl in ("CUPTI_ACTIVITY_KIND_MEMSET", "CUPTI_ACTIVITY_KIND_MEMCPY"):
    for s, e, c, st in db.execute(f"SELECT start,end,correlationId,streamId FROM {tbl} WHERE graphNodeId IS NOT NULL"):
        by[c].append((s, e, "MEMNODE", st))
sizes = collections.Counter(len(v) for v in by.values()); tn = max(n for n, c in sizes.items() if c >= 20)
L = [sorted(v) for v in by.values() if len(v) == tn]
FI = r"kernel_mha|topkGating|BuildExpertMaps|expandInputRows|computeStrides|doActivation|finalizeMoe|PrefixSum|cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute::tuple|GroupProblemShape|cvt_fp16_to_fp4|causal_conv1d"
IND = r"triton_"
OURS = r"enable_sm120_family|wmma_tensorop|splitKreduce"   # dense FP8 + BF16 skinny GEMMs (we ship replacements)
OTHER = r"gdn_decode_post_conv|rmsnorm_rope|reshape_and_cache|gemvx|elementwise|reduce_kernel"
scen = {
    "today (FlashInfer/vLLM PDL kernels only)": [FI],
    "+ TORCHINDUCTOR_ENABLE_PDL=1": [FI, IND],
    "+ our skinny FP8/BF16 GEMMs PDL-aware": [FI, OURS],
    "+ both": [FI, IND, OURS],
    "+ both + GDN decode, rope/kv-write, gemv, ATen (everything)": [FI, IND, OURS, OTHER],
}
pairs = []
for ks in L:
    by_s = collections.defaultdict(list)
    for k in ks: by_s[k[3]].append(k)
    for v in by_s.values():
        v.sort()
        pairs += [(a, b) for a, b in zip(v, v[1:])]
nl = len(L)
print(f"{nl} launches; {len(pairs)/nl:.0f} same-stream boundaries/step; currently overlapped {sum(b[0] < a[1] for a, b in pairs)/nl:.0f}")
print(f"{'scenario':64s} {'PDL edges/step':>14s} {'of which not overlapped today':>30s}")
base = None
for name, pats in scen.items():
    cap = lambda n: n != "MEMNODE" and any(re.search(p, n) for p in pats)
    edges = [(a, b) for a, b in pairs if cap(a[2]) and cap(b[2])]
    new = [(a, b) for a, b in edges if not b[0] < a[1]]
    print(f"{name:64s} {len(edges)/nl:14.0f} {len(new)/nl:30.0f}")
