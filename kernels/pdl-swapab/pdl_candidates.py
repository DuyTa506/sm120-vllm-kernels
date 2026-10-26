"""Who could take part in PDL? For every kernel boundary (consumer kernel and its same-stream predecessor, graph
memset nodes included) in the target decode graph of a node-level nsys trace, group by the CONSUMER kernel family
and by who controls its launch, and report how many boundaries already overlap (= PDL already active).
usage: python3 -I pdl_candidates.py <node_trace.sqlite>"""
import collections, re, sqlite3, statistics, sys
db = sqlite3.connect(sys.argv[1]); S = dict(db.execute("SELECT id, value FROM StringIds"))
by = collections.defaultdict(list)
for s, e, c, n, st in db.execute("SELECT start,end,correlationId,demangledName,streamId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphNodeId IS NOT NULL"):
    by[c].append((s, e, S[n], st))
for tbl in ("CUPTI_ACTIVITY_KIND_MEMSET", "CUPTI_ACTIVITY_KIND_MEMCPY"):
    for s, e, c, st in db.execute(f"SELECT start,end,correlationId,streamId FROM {tbl} WHERE graphNodeId IS NOT NULL"):
        by[c].append((s, e, "MEMSET/MEMCPY node", st))
sizes = collections.Counter(len(v) for v in by.values()); tn = max(n for n, c in sizes.items() if c >= 20)
L = [sorted(v) for v in by.values() if len(v) == tn]
# (regex, family, owner): owner "FI" = FlashInfer/TRT-LLM (PDL attr set by library), "vLLM-PDL" = vLLM Triton with launch_pdl,
# "inductor" = torch.compile-generated Triton (TORCHINDUCTOR_ENABLE_PDL=0 by default), "cuBLAS/ATen", "vLLM-C++", "ours-candidate"
RULES = [
    (r"kernel_mha", "XQA decode attention", "FlashInfer (PDL on)"),
    (r"topkGating|BuildExpertMaps|expandInputRows|computeStrides|doActivation|finalizeMoe|PrefixSum", "MoE routing/permute glue", "FlashInfer (PDL on)"),
    (r"cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute::tuple|GroupProblemShape", "NVFP4 MoE/shared-expert GEMM (CUTLASS)", "FlashInfer (PDL on)"),
    (r"enable_sm120_family", "FP8 dense GEMM (CUTLASS, vLLM _C)", "vLLM C++ (patchable: skinny FP8 is ours)"),
    (r"wmma_tensorop|splitKreduce", "BF16 router/in_proj_ba GEMM (cuBLAS)", "cuBLAS (patchable: skinny BF16 is ours, Triton)"),
    (r"cvt_fp16_to_fp4", "NVFP4 act quant (vLLM _C)", "vLLM C++"),
    (r"causal_conv1d", "GDN conv1d update", "vLLM Triton (launch_pdl)"),
    (r"gdn_decode_post_conv|fused_recurrent|gdn", "GDN recurrent decode", "vLLM"),
    (r"rmsnorm_rope|reshape_and_cache", "rope / kv write", "vLLM/FlashInfer"),
    (r"triton_", "norm/quant/act fused Triton", "inductor (env switch exists)"),
    (r"gemvx", "gemv (shared_expert_gate)", "cuBLAS"),
    (r"elementwise|reduce_kernel", "ATen elementwise", "PyTorch"),
    (r"MEMSET", "memset node (MoE output zero)", "FlashInfer, not PDL-able"),
]
def fam(n):
    for p, f, o in RULES:
        if re.search(p, n): return f, o
    return "other", "?"
agg = collections.defaultdict(lambda: [0, 0, 0.0, 0.0, ""])
for ks in L:
    by_s = collections.defaultdict(list)
    for k in ks: by_s[k[3]].append(k)
    for v in by_s.values():
        v.sort()
        for a, b in zip(v, v[1:]):
            f, o = fam(b[2]); r = agg[f]; r[4] = o; r[0] += 1
            if b[0] < a[1]: r[1] += 1; r[2] += a[1] - b[0]
            else: r[3] += b[0] - a[1]
nl = len(L)
print(f"{nl} launches, {tn} nodes; boundaries by consumer family")
print(f"{'consumer family':44s} {'bound/step':>10s} {'already overlap':>15s} {'mean ovl ns':>11s} {'mean gap ns (rest)':>18s}  owner")
tot = nov = 0
for f, (n, no, so, sg, o) in sorted(agg.items(), key=lambda x: -x[1][0]):
    print(f"{f:44s} {n/nl:10.1f} {100*no/n:14.0f}% {so/max(no,1):11.0f} {sg/max(n-no,1):18.0f}  {o}")
    tot += n; nov += no
print(f"{'total':44s} {tot/nl:10.1f} {100*nov/tot:14.0f}%")
