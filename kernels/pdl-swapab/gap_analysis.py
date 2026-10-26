"""Idle gap between consecutive kernels inside one decode CUDA graph launch (nsys node-level trace).

usage: python3 -I gap_analysis.py <node_trace.sqlite> [--pairs N]

Kernels of one cudaGraphLaunch share a correlationId. For the target verify graph (largest kernel
count seen >= 20 times) and the MTP drafter graphs, order the kernels by start time and report the
gap end(i) -> start(i+1). Overlapping kernels (parallel graph branches) give a negative gap, which is
counted as 0 and reported separately. Stats are medians over graph launches.
"""
import collections, re, sqlite3, statistics, sys

path = sys.argv[1]
MINC = int(sys.argv[sys.argv.index("--min") + 1]) if "--min" in sys.argv else 20
NP = int(sys.argv[sys.argv.index("--pairs") + 1]) if "--pairs" in sys.argv else 25
db = sqlite3.connect(path)
S = dict(db.execute("SELECT id, value FROM StringIds"))
by = collections.defaultdict(list)
for s, e, c, n, st in db.execute("SELECT start,end,correlationId,demangledName,streamId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphNodeId IS NOT NULL"):
    by[c].append((s, e, S[n], st))
for tbl, lab in (("CUPTI_ACTIVITY_KIND_MEMSET", "memset"), ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy")):  # graph memset/memcpy nodes
    for s, e, c, st in db.execute(f"SELECT start,end,correlationId,streamId FROM {tbl} WHERE graphNodeId IS NOT NULL"):
        by[c].append((s, e, lab, st))
sizes = collections.Counter(len(v) for v in by.values())
target_n = max(n for n, cnt in sizes.items() if cnt >= MINC)
draft_ns = sorted((n for n, cnt in sizes.items() if cnt >= 0.5 * sizes[target_n] and MINC <= n < target_n), reverse=True)
print("graph sizes (kernels: launches):", dict(sorted(sizes.items(), key=lambda x: -x[1])[:8]))


def short(n):
    n = re.sub(r"\(.*", "", n) if not n.startswith("void") else n
    for pat, lab in [
        (r"GroupProblemShape", "moe_grouped_gemm"), (r"enable_sm120_family.*GemmUniversal", "fp8_gemm"),
        (r"cutlass::gemm::kernel::GemmUniversal<cute::tuple<int, int, int, int>", "nvfp4_shared_gemm"),
        (r"wmma_tensorop|splitKreduce", "bf16_wmma_gemm"), (r"gemvx", "gemv"), (r"kernel_mha", "xqa_attn"),
        (r"gdn_decode_post_conv", "gdn_decode"), (r"topkGating", "topkGating"), (r"BuildExpertMaps", "BuildExpertMaps"),
        (r"expandInputRows", "expandInputRows"), (r"doActivation", "doActivation"), (r"computeStrides", "computeStrides"),
        (r"cvt_fp16_to_fp4|scaled_fp4_quant", "fp4_quant"), (r"blockExpertPrefixSum|globalExpertPrefixSum|mergeExpertPrefixSum", "expertPrefixSum"),
        (r"finalizeMoeRouting", "finalizeMoe"), (r"causal_conv1d", "conv1d"), (r"rmsnorm_rope|reshape_and_cache", "rope_kvwrite"),
        (r"triton_(\w+?)_\d", None), (r"triton_", "triton_other"), (r"vectorized_elementwise|elementwise_kernel", "elementwise"),
        (r"memset|Memset", "memset")]:
        m = re.search(pat, n)
        if m:
            return lab or ("triton:" + re.sub(r"_\d+$", "", m.group(0).replace("triton_", ""))[:28])
    return re.sub(r"\s+", " ", n)[:40]


def analyze(label, ns):
    launches = [sorted(v) for v in by.values() if len(v) in ns]
    if not launches:
        return
    nl = len(launches)
    spans, sums, gaps_tot, ngap, lt2, lt1, lt05, neg, nstreams = [], [], [], [], [], [], [], [], []
    pair = collections.defaultdict(list)
    allgaps = []
    for ks in launches:
        spans.append(ks[-1][1] - ks[0][0] if True else 0)
        spans[-1] = max(k[1] for k in ks) - ks[0][0]
        sums.append(sum(k[1] - k[0] for k in ks))
        g_tot = 0; cnt = 0; l2 = l1 = l05 = ng = 0
        run_end = ks[0][1]
        for a, b in zip(ks, ks[1:]):
            gap = b[0] - run_end
            if gap < 0:
                ng += 1; gap = 0
            g_tot += gap; cnt += 1
            l2 += gap < 2000; l1 += gap < 1000; l05 += gap < 500
            pair[(short(a[2]), short(b[2]))].append(gap)
            allgaps.append(gap)
            run_end = max(run_end, b[1])
        gaps_tot.append(g_tot); ngap.append(cnt); lt2.append(l2); lt1.append(l1); lt05.append(l05); neg.append(ng)
        nstreams.append(len({k[3] for k in ks}))
    m = statistics.median
    print(f"\n=== {label}: {len(launches[0])} kernels/launch, {nl} launches, streams {m(nstreams):.0f}")
    print(f"  node-trace span {m(spans)/1e3:8.1f} us | sum of kernel durations {m(sums)/1e3:8.1f} us | gap total {m(gaps_tot)/1e3:8.1f} us "
          f"({100*m(gaps_tot)/m(spans):.1f}% of span) | mean gap {m(gaps_tot)/max(1,m(ngap)):.0f} ns")
    print(f"  gaps: {m(ngap):.0f} per launch; <2us {m(lt2):.0f}, <1us {m(lt1):.0f}, <0.5us {m(lt05):.0f}; overlapped(neg) {m(neg):.0f}")
    allgaps.sort()
    n = len(allgaps)
    print(f"  gap percentiles (ns): p10 {allgaps[n//10]}  p50 {allgaps[n//2]}  p90 {allgaps[9*n//10]}  p99 {allgaps[99*n//100]}  max {allgaps[-1]}")
    print(f"  top neighbour pairs by total gap (us per launch / pairs per launch / mean gap ns / mean dur of 2nd kernel):")
    rows = sorted(pair.items(), key=lambda x: -sum(x[1]))[:NP]
    for (a, b), v in rows:
        print(f"    {sum(v)/nl/1e3:7.1f} us x{len(v)/nl:5.1f}  {statistics.mean(v):6.0f} ns   {a} -> {b}")
    return pair, nl


analyze("target verify graph", [target_n])
for dn in draft_ns[:2]:
    analyze(f"MTP drafter graph ({dn} kernels)", [dn])
