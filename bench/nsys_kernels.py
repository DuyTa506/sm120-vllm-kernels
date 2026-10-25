import sqlite3, sys, re, collections
db = sqlite3.connect(sys.argv[1]); NREQ = int(sys.argv[2]) if len(sys.argv) > 2 else 4
TOP = int(sys.argv[3]) if len(sys.argv) > 3 else 40
rows = db.execute("""SELECT k.start,k.end,s.value,k.graphNodeId IS NOT NULL
                     FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName""").fetchall()
rows.sort()
# keep the busy window(s): split on >3 ms idle, keep groups with >=100 kernels
groups, cur, le = [], [rows[0]], rows[0][1]
for r in rows[1:]:
    if r[0] - le > 3_000_000: groups.append(cur); cur = [r]
    else: cur.append(r)
    le = max(le, r[1])
groups.append(cur)
rows = [r for g in groups if len(g) >= 100 for r in g]
span = sum(max(x[1] for x in g) - g[0][0] for g in groups if len(g) >= 100)
tot = sum(e - s for s, e, _, _ in rows)
print(f"{len(rows)} kernels, kernel-time sum {tot/1e6/NREQ:.1f} ms/req, span {span/1e6/NREQ:.1f} ms/req")

def cat(n):
    l = n.lower()
    rules = [
        ('fa/attn', r'flash|fmha|attn|attention|flashinfer|prefill_kernel|batchprefill'),
        ('gdn/fla', r'chunk_|fla|gated_delta|recurrent|gdn|wy_fast|solve_tril|cumsum|fwd_h|fwd_o|scaled_dot_kkt'),
        ('conv1d', r'causal_conv|conv1d'),
        ('moe-gemm', r'moe|grouped|cutlass.*group|fused_experts|marlin|cutlass_fp4_group'),
        ('moe-route', r'topk|router|routing|expert_ids|sort|align|permute|moe_align|scatter|gather'),
        ('nvfp4-quant', r'quant|fp4|nvfp4|scaled_fp4'),
        ('gemm', r'gemm|cutlass|cublas|nvjet|sm\d+_xmma|ampere_|sm90|sm100|sm120'),
        ('norm/act', r'rms|norm|silu|swiglu|act_and_mul|gelu|sigmoid'),
        ('rope/kv', r'rotary|rope|reshape_and_cache|cache|kv'),
        ('elementwise', r'elementwise|vectorized|copy|cat|index|fill|mul|add'),
    ]
    for c, p in rules:
        if re.search(p, l): return c
    return 'other'

by = collections.defaultdict(lambda: [0, 0]); cats = collections.defaultdict(lambda: [0, 0])
for s, e, n, g in rows:
    by[n][0] += e - s; by[n][1] += 1
    c = cats[cat(n)]; c[0] += e - s; c[1] += 1
print("\n== by category (ms/req, % of kernel time, launches/req)")
for c, (t, k) in sorted(cats.items(), key=lambda x: -x[1][0]):
    print(f"  {c:12s} {t/1e6/NREQ:7.2f} ms  {100*t/tot:5.1f}%  {k/NREQ:7.0f}")
print(f"\n== top {TOP} kernels")
for n, (t, k) in sorted(by.items(), key=lambda x: -x[1][0])[:TOP]:
    print(f"  {t/1e6/NREQ:7.2f} ms {100*t/tot:5.1f}% x{k/NREQ:6.0f}  avg {t/k/1e3:7.1f} us [{cat(n)}] {n[:110]}")
