"""Group the kernels of a node-level decode trace into per-step categories (us per decode step).
usage: python3 -I step_categories.py traces/node_c1.sqlite
Steps are counted as launches of the largest graph (the target verify forward). Kernel durations
come from a --cuda-graph-trace=node capture, which inflates short kernels by ~0.5-1 us each.
"""
import collections, re, sqlite3, sys

db = sqlite3.connect(sys.argv[1])
S = dict(db.execute("SELECT id, value FROM StringIds"))
rows = db.execute("SELECT correlationId, demangledName, gridX, gridY, end-start FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
by_corr = collections.defaultdict(list)
for c, n, gx, gy, d in rows:
    by_corr[c].append((S[n], gx, gy, d))
sizes = collections.Counter(len(v) for v in by_corr.values())
target_n = max(n for n, cnt in sizes.items() if cnt >= 20)
draft_ns = sorted((n for n, cnt in sizes.items() if cnt >= 0.5 * sizes[target_n] and 20 <= n < target_n), reverse=True)
steps = sizes[target_n]

RULES = [
    ("FP8 dense GEMM (GDN qkvz/out, attn qkv/o)", r"enable_sm120_family.*GemmUniversal"),
    ("NVFP4 MoE grouped GEMM", r"GroupProblemShape"),
    ("NVFP4 shared-expert GEMM", r"cutlass::gemm::kernel::GemmUniversal<cute::tuple<int, int, int, int>, cutlass::gemm::collect"),
    ("BF16 small GEMM (router, in_proj_ba) wmma/skinny", r"wmma_tensorop|_skinny_bf16_gemm|splitKreduce"),
    ("gemv (shared_expert_gate)", r"gemvx"),
    ("decode attention (XQA kernel_mha)", r"kernel_mha"),
    ("GDN recurrent decode (fp32 state)", r"gdn_decode_post_conv"),
    ("MoE routing/permute glue", r"topkGating|BuildExpertMaps|expandInputRows|doActivation|computeStrides|finalizeMoeRouting|cvt_fp16_to_fp4|blockExpertPrefixSum"),
    ("BF16 MoE (MTP experts)", r"fused_moe::|MoeFCGemm"),
    ("norm/quant/act fused (triton)", r"triton_"),
    ("conv1d update", r"causal_conv1d"),
    ("rope/qk-norm/kv-cache write", r"rmsnorm_rope|reshape_and_cache"),
]


def cat(n):
    for c, p in RULES:
        if re.search(p, n):
            return c
    return "other (elementwise, memset, misc)"


for label, ns in (("target verify graph", [target_n]), ("MTP drafter graphs", draft_ns)):
    agg = collections.defaultdict(lambda: [0, 0])
    for v in by_corr.values():
        if len(v) in ns:
            for n, gx, gy, d in v:
                a = agg[cat(n)]
                a[0] += d; a[1] += 1
    tot = sum(a[0] for a in agg.values())
    print(f"\n== {label}: {tot/steps/1e3:.0f} us/step kernel time, {sum(a[1] for a in agg.values())/steps:.0f} kernels/step")
    for c, (t, k) in sorted(agg.items(), key=lambda x: -x[1][0]):
        print(f"  {t/steps/1e3:8.1f} us  x{k/steps:6.1f}  {c}")
