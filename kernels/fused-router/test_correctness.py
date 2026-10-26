"""Correctness of the fused router against vLLM's unfused path (torch.mm / F.linear + ops.topk_softmax|topk_sigmoid).
Run on GPU through ./run_gpu.sh test_correctness.py  (writes results/correctness.txt via tee in the caller).

 1. selection/normalisation arithmetic, isolated: logits are injected exactly (W = identity block, x = desired bf16
    logits), so the fused kernel sees the same bf16 logits as vLLM's topk kernel. Random + heavy ties. Expect bitwise.
 2. real router weights (40 layers + MTP) with RMSNorm-style activations, M = 1..32: fused vs (cuBLAS logits -> vLLM topk).
 3. parameter sweep: E, topk, scoring, bias, scale, renorm, shared gate rows, strided x, M>32 (several token blocks).
 4. kernel configs (nt, nw, split) all give the same answer; determinism; CUDA graph replay; counters re-armed;
    run while another stream saturates the GPU (no co-residency assumption).
"""
import itertools, sys
import torch
import torch.nn.functional as F
import vllm._custom_ops as ops
from common import real, normed_input, L
import fused_router as fr

dev = "cuda"
torch.manual_seed(0)
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


def vllm_route(logits, k, renorm, kind="softmax", bias=None, scale=1.0, pad=None):
    M = logits.shape[0]
    w = torch.empty(M, k, dtype=torch.float32, device=dev)
    i = torch.empty(M, k, dtype=torch.int32, device=dev)
    t = torch.empty(M, k, dtype=torch.int32, device=dev)
    if kind == "softmax":
        ops.topk_softmax(w, i, t, logits, renorm, bias, is_padding=pad)
    else:
        ops.topk_sigmoid(w, i, t, logits, renorm, bias, scale, is_padding=pad)
    return w, i


def inject(logits, K=2048, sh_logit=None):
    """x, W such that bf16(x @ W^T) == logits exactly: W[e, e] = 1 (and W[E, K-1] = 1 for a shared gate row)."""
    M, E = logits.shape
    x = torch.zeros(M, K, dtype=torch.bfloat16, device=dev)
    x[:, :E] = logits
    R = E + (0 if sh_logit is None else sh_logit.shape[1])
    W = torch.zeros(R, K, dtype=torch.bfloat16, device=dev)
    W[torch.arange(E), torch.arange(E)] = 1
    if sh_logit is not None:
        for j in range(sh_logit.shape[1]):
            x[:, K - 1 - j] = sh_logit[:, j]
            W[E + j, K - 1 - j] = 1
    return x, W


# ------------------------------------------------------------------------------------------------
print("== 1. scoring / top-k / normalisation on injected logits: ids must be identical to vLLM; weights within a few fp32 ulp")
for E, k in ((256, 8), (128, 8), (512, 8), (64, 6), (256, 4), (256, 1)):
    for tie_mode in ("random", "ties"):
        for M in (1, 3, 8, 24):
            if tie_mode == "random":
                lg = (torch.randn(M, E, device=dev) * 2).bfloat16()
            else:   # only a few distinct values -> many exact ties inside and across the top-k boundary
                lg = torch.randint(-3, 3, (M, E), device=dev).float().bfloat16() * 0.5
            x, W = inject(lg)
            for renorm in (True, False):
                w0, i0 = vllm_route(lg, k, renorm)
                w1, i1, _, _ = fr.fused_router(x, W, k, scoring="softmax", renormalize=renorm)
                ok_i = bool((i0 == i1).all())
                ok_w = bool((w0 == w1).all())
                rel = ((w0 - w1).abs() / w0.clamp_min(1e-30)).max().item()
                check(f"softmax E={E} k={k} {tie_mode} M={M} renorm={renorm}", ok_i and rel < 2e-6,
                      f"ids_equal={ok_i} weights_bitwise={ok_w} max_rel={rel:.2e}")
sys.stdout.flush()

# ------------------------------------------------------------------------------------------------
print("== 2. real router weights, RMSNorm-style activations. Reference = exact (fp64) logits rounded to bf16 -> vLLM topk;")
print("      also reported: vLLM's own unfused cuBLAS path against the same reference (cuBLAS is not exact at every M)")
agg = {}
worst_rel = 0.0
sh_maxabs = 0.0
n_same_rows = n_same_rows_ok = 0
for layer in list(range(40)) + ["mtp"]:
    if layer == "mtp":
        Wg = real("mtp.layers.0.mlp.gate.weight"); lnw = real("mtp.layers.0.post_attention_layernorm.weight")
        Wsh = real("mtp.layers.0.mlp.shared_expert_gate.weight")
    else:
        Wg = real(L + f"{layer}.mlp.gate.weight"); lnw = real(L + f"{layer}.post_attention_layernorm.weight")
        Wsh = real(L + f"{layer}.mlp.shared_expert_gate.weight")
    Wc = torch.cat([Wg, Wsh]).contiguous()
    pool = normed_input(96, lnw)
    for M in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        a = agg.setdefault(M, dict(rows=0, f_set=0, f_ids=0, c_set=0, c_ids=0, f_vs_c_set=0, logit_exact=0, logit_n=0,
                                   c_logit_exact=0, same_rows=0, same_rows_ids=0))
        for off in range(0, 96 - M + 1, max(M, 24)):
            xm = pool[off:off + M].contiguous()
            ref_lg = (xm.double() @ Wg.double().T).float().bfloat16()          # exact logits, one rounding
            wr, ir = vllm_route(ref_lg, 8, True)
            lg = F.linear(xm, Wg)                                               # vLLM tier-6 path (cuBLAS)
            wc, ic = vllm_route(lg, 8, True)
            wf, i_f, sh, lg1 = fr.fused_router(xm, Wc, 8, num_experts=256, scoring="softmax", renormalize=True, write_logits=True)
            a["rows"] += M
            a["f_ids"] += int((i_f == ir).all(-1).sum()); a["c_ids"] += int((ic == ir).all(-1).sum())
            a["f_set"] += int((i_f.sort(-1).values == ir.sort(-1).values).all(-1).sum())
            a["c_set"] += int((ic.sort(-1).values == ir.sort(-1).values).all(-1).sum())
            a["f_vs_c_set"] += int((i_f.sort(-1).values == ic.sort(-1).values).all(-1).sum())
            a["logit_exact"] += int((lg1 == ref_lg).sum()); a["c_logit_exact"] += int((lg == ref_lg).sum()); a["logit_n"] += lg.numel()
            # rows where the fused kernel's 256 bf16 logits are bit-identical to the exact ones: the routing arithmetic
            # (softmax, top-8 order incl. ties, renormalisation) must then be identical to vLLM's kernel
            same = (lg1 == ref_lg).all(-1)
            a["same_rows"] += int(same.sum())
            ok_rows = (i_f == ir).all(-1) & same
            a["same_rows_ids"] += int(ok_rows.sum())
            if same.any():
                worst_rel = max(worst_rel, ((wr - wf).abs() / wr)[same].max().item())
            sh_maxabs = max(sh_maxabs, (torch.sigmoid(F.linear(xm, Wsh)).float() - sh.float()).abs().max().item())
print("   M   rows | logits==exact: fused  cuBLAS | top-8 set==exact: fused  cuBLAS | rows with all 256 logits exact: n, ordered ids identical | fused set==cuBLAS set")
for M, a in agg.items():
    print(f"  {M:3d} {a['rows']:6d} |  {a['logit_exact']/a['logit_n']:.5f} {a['c_logit_exact']/a['logit_n']:.5f}        |  {a['f_set']/a['rows']:.5f} {a['c_set']/a['rows']:.5f}     | {a['same_rows']:6d} {a['same_rows_ids']/max(1,a['same_rows']):.5f}                          | {a['f_vs_c_set']/a['rows']:.5f}")
tot_rows = sum(a["rows"] for a in agg.values())
f_set = sum(a["f_set"] for a in agg.values()) / tot_rows
c_set = sum(a["c_set"] for a in agg.values()) / tot_rows
same_rows = sum(a["same_rows"] for a in agg.values()); same_ok = sum(a["same_rows_ids"] for a in agg.values())
print(f"   all M: fused top-8 set == exact on {f_set:.5f} of rows, cuBLAS+vLLM on {c_set:.5f}; rows with bit-identical logits {same_rows}/{tot_rows}:"
      f" ordered ids identical on {same_ok}/{same_rows}, max rel weight diff {worst_rel:.2e}; shared-gate sigmoid max abs diff vs unfused {sh_maxabs:.1e}")
check("real weights: fused top-8 set equals the exact-logit routing on >= 99.9% of rows", f_set >= 0.999)
check("real weights: fused is at least as close to exact routing as the unfused cuBLAS path", f_set >= c_set)
check("real weights: on rows with bit-identical logits, ordered ids identical and weights within 1e-6 rel", same_ok == same_rows and worst_rel < 1e-6)
check("shared-gate sigmoid within 1 bf16 ulp of unfused (was bitwise identical in the first run)", sh_maxabs <= 4e-3)
sys.stdout.flush()

# ------------------------------------------------------------------------------------------------
print("== 3. parameter sweep against vLLM ops (injected logits, so only the routing arithmetic differs)")
for E, k, kind, bias_on, renorm, scale in [
        (256, 8, "sigmoid", True, True, 2.5),      # DeepSeek-V3 style (without grouping)
        (256, 8, "sigmoid", False, False, 1.0),
        (384, 6, "sigmoid", True, True, 1.5),
        (128, 8, "softmax", False, True, 1.0),     # Qwen3-30B-A3B shape
        (64, 6, "softmax", False, False, 1.0),
        (256, 8, "softmax", True, True, 1.0)]:
    for M in (1, 3, 24, 70):
        lg = (torch.randn(M, E, device=dev) * 2).bfloat16()
        bias = torch.randn(E, device=dev) if bias_on else None
        x, W = inject(lg)
        w0, i0 = vllm_route(lg, k, renorm, kind=kind, bias=bias, scale=scale)
        w1, i1, _, _ = fr.fused_router(x, W, k, scoring=kind, renormalize=renorm, bias=bias, scale=scale)
        ok_i = bool((i0 == i1).all())
        rel = ((w0 - w1).abs() / w0.clamp_min(1e-30)).max().item()
        check(f"{kind} E={E} k={k} bias={bias_on} renorm={renorm} scale={scale} M={M}", ok_i and rel < 2e-6,
              f"ids_equal={ok_i} weights_bitwise={bool((w0 == w1).all())} max_rel={rel:.1e}")
# sqrtsoftplus (DeepSeek-V4 style) has no plain vLLM op without hash tables: compare with a torch reference
for M in (1, 3, 24):
    E, k = 256, 8
    lg = (torch.randn(M, E, device=dev) * 2).bfloat16()
    bias = torch.randn(E, device=dev)
    x, W = inject(lg)
    w1, i1, _, _ = fr.fused_router(x, W, k, scoring="sqrtsoftplus", renormalize=True, bias=bias, scale=1.5)
    sc = torch.sqrt(F.softplus(lg.float()))
    ir = torch.topk(sc + bias, k, -1).indices.int()
    wr = sc.gather(1, ir.long()); wr = wr / wr.sum(-1, keepdim=True) * 1.5
    check(f"sqrtsoftplus bias M={M} vs torch", bool((ir.sort(-1).values == i1.sort(-1).values).all()) and
          ((wr.sort(-1).values - w1.sort(-1).values).abs() / wr.sort(-1).values).max().item() < 1e-5)
# shared gate rows with exact injected values; strided x; non-multiple-of-16 rows (E+S=257 -> padded)
M, E = 5, 256
lg = (torch.randn(M, E, device=dev) * 2).bfloat16()
shl = (torch.randn(M, 1, device=dev) * 3).bfloat16()
x, W = inject(lg, sh_logit=shl)
big = torch.zeros(M, 2048 + 64, dtype=torch.bfloat16, device=dev); big[:, :2048] = x
_, _, sh, _ = fr.fused_router(big[:, :2048], W, 8, num_experts=E)
check("shared gate = bf16(sigmoid(bf16 logit)), strided x (ldx=2112)", bool((sh == torch.sigmoid(shl)).all()),
      f"max abs {(sh.float() - torch.sigmoid(shl).float()).abs().max().item():.1e}")
# 2 shared-gate rows
shl2 = (torch.randn(M, 2, device=dev) * 3).bfloat16()
x2, W2 = inject(lg, sh_logit=shl2)
_, _, sh2, _ = fr.fused_router(x2, W2, 8, num_experts=E)
check("two shared gate rows", bool((sh2 == torch.sigmoid(shl2)).all()))
sys.stdout.flush()

# fast path (softmax, no bias, bf16 logits: packed-key redux ranking) vs the general path (forced with a zero bias)
for E, k in ((256, 8), (128, 8), (512, 8), (64, 6)):
    for tie_mode in ("random", "ties"):
        for M in (1, 3, 24):
            lg = (torch.randn(M, E, device=dev) * 2).bfloat16() if tie_mode == "random" else torch.randint(-3, 3, (M, E), device=dev).float().bfloat16() * 0.5
            x, W = inject(lg)
            wf, i_f, _, _ = fr.fused_router(x, W, k)
            wg, i_g, _, _ = fr.fused_router(x, W, k, bias=torch.zeros(E, device=dev))
            check(f"fast path == general path E={E} k={k} {tie_mode} M={M}", bool((i_f == i_g).all()) and ((wf - wg).abs() / wg).max().item() < 3e-7,
                  f"max_rel={((wf - wg).abs() / wg).max().item():.1e}")
# negative / mixed-sign logits and all-equal rows through the packed key
for M in (3, 8):
    for lg in ((torch.randn(M, 256, device=dev) * 5 - 3).bfloat16(), torch.zeros(M, 256, device=dev, dtype=torch.bfloat16),
               -torch.rand(M, 256, device=dev).bfloat16() * 20):
        x, W = inject(lg)
        w0, i0 = vllm_route(lg, 8, True)
        w1, i1, _, _ = fr.fused_router(x, W, 8)
        check(f"negative/zero logits M={M}", bool((i0 == i1).all()) and ((w0 - w1).abs() / w0).max().item() < 3e-7)

# padded rows (VLLM_MOE_SKIP_PADDING): vLLM writes topk_ids = -1 for them
for M in (4, 24):
    lg = (torch.randn(M, 256, device=dev) * 2).bfloat16()
    x, W = inject(lg)
    pad = torch.zeros(M, dtype=torch.bool, device=dev); pad[M - 1] = True; pad[1] = True
    w0, i0 = vllm_route(lg, 8, True, pad=pad)
    w1, i1, _, _ = fr.fused_router(x, W, 8, pad=pad)
    check(f"padding rows -> ids -1 like vLLM, M={M}", bool((i0 == i1).all()) and bool((i1[pad] == -1).all()) and
          ((w0 - w1).abs() / w0).max().item() < 2e-6)
# routed_scaling_factor without renormalisation (sigmoid), scale != 1
lg = (torch.randn(8, 256, device=dev) * 2).bfloat16(); x, W = inject(lg)
w0, i0 = vllm_route(lg, 8, False, kind="sigmoid", scale=2.0)
w1, i1, _, _ = fr.fused_router(x, W, 8, scoring="sigmoid", renormalize=False, scale=2.0)
check("sigmoid, no renorm, scale=2", bool((i0 == i1).all()) and ((w0 - w1).abs() / w0).max().item() < 2e-6,
      f"max_rel={((w0 - w1).abs() / w0).max().item():.1e}")

# ------------------------------------------------------------------------------------------------
print("== 4. configs, determinism, CUDA graph, re-armed counters, concurrent load")
Wg = real(L + "0.mlp.gate.weight"); Wsh = real(L + "0.mlp.shared_expert_gate.weight"); Wc = torch.cat([Wg, Wsh]).contiguous()
lnw = real(L + "0.post_attention_layernorm.weight")
base = None
for M in (3, 24, 40, 100):
    x = normed_input(M, lnw)
    ref = fr.fused_router(x, Wc, 8, num_experts=256, cfg=(4, 32, 4))
    ok = True
    for nt, nw, split in itertools.product((1, 2, 3, 4), (4, 8, 16, 32), (1, 2, 4, 8, 11, 16)):
        if nt * 8 < min(M, 32) and M <= 32:
            continue
        out = fr.fused_router(x, Wc, 8, num_experts=256, cfg=(nt, nw, split))
        # different split => different (still fp32-exact-ish) summation order => logits may differ by 1 bf16 ulp
        same_ids = bool((out[1].sort(-1).values == ref[1].sort(-1).values).float().mean() > 0.995)
        ok &= same_ids
    check(f"all (nt,nw,split) configs agree on top-8 sets M={M}", ok)
x = normed_input(3, lnw)
a = fr.fused_router(x, Wc, 8, num_experts=256); b = fr.fused_router(x, Wc, 8, num_experts=256)
ra0 = fr.fused_router(x, Wc, 8, num_experts=256, relacq=0); ra1 = fr.fused_router(x, Wc, 8, num_experts=256, relacq=1)
check("relacq=0 and relacq=1 give identical results", bool((ra0[0] == ra1[0]).all() and (ra0[1] == ra1[1]).all()))
pfo = fr.fused_router(x, Wc, 8, num_experts=256, prefetch_next=Wc.clone())
check("L2 prefetch of the next weight does not change the result", bool((pfo[0] == a[0]).all() and (pfo[1] == a[1]).all()))
check("bitwise repeatable", bool((a[0] == b[0]).all() and (a[1] == b[1]).all() and (a[2] == b[2]).all()))
ws = fr.Workspace(dev)
fr.fused_router(x, Wc, 8, num_experts=256, ws=ws)
torch.cuda.synchronize()
check("ticket counters back to 0 after a launch", bool((ws.counters == 0).all()))
# CUDA graph replay with new data
xs = normed_input(3, lnw); out_w = torch.empty(3, 8, device=dev); out_i = torch.empty(3, 8, dtype=torch.int32, device=dev)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    fr.fused_router(xs, Wc, 8, num_experts=256, ws=ws, out=(out_w, out_i))
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    fr.fused_router(xs, Wc, 8, num_experts=256, ws=ws, out=(out_w, out_i))
okg = True
for _ in range(5):
    xs.copy_(normed_input(3, lnw)); g.replay(); torch.cuda.synchronize()
    ew, ei, _, _ = fr.fused_router(xs, Wc, 8, num_experts=256)
    okg &= bool((ew == out_w).all() and (ei == out_i).all())
check("CUDA graph replays with new inputs == eager", okg)
check("counters still 0 after graph replays", bool((ws.counters == 0).all()))
# concurrent hog on another stream (many CTAs, long): the fused kernel must still finish and be right
hog = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
s2 = torch.cuda.Stream()
okc = True
with torch.cuda.stream(s2):
    for _ in range(6):
        hog @ hog
for _ in range(20):
    xs = normed_input(24, lnw)
    r1 = fr.fused_router(xs, Wc, 8, num_experts=256)
    torch.cuda.synchronize()
    r2 = fr.fused_router(xs, Wc, 8, num_experts=256)
    okc &= bool((r1[1] == r2[1]).all() and (r1[0] == r2[0]).all())
torch.cuda.synchronize()
check("completes and is correct while another stream saturates the SMs", okc)

print("FAILED:" if FAILS else "ALL PASSED", FAILS)
sys.exit(1 if FAILS else 0)
