"""Cluster variant (thread block clusters + DSMEM) correctness. Separate script so a launch failure on a given
GPU cannot take the main correctness run down. Run through ./run_gpu.sh test_cluster.py."""
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
print("== 5. cluster variant (thread block clusters + DSMEM hand-off)")
Wg = real(L + "0.mlp.gate.weight"); Wsh = real(L + "0.mlp.shared_expert_gate.weight"); Wc = torch.cat([Wg, Wsh]).contiguous()
lnw = real(L + "0.post_attention_layernorm.weight")
CL = dict(mode="cluster")
for cs, nw in itertools.product((8, 16), (8, 16)):
    for M in (1, 3, 8, 16, 24, 32, 45):
        nt = min(4, (M + 7) // 8)
        for E, k in ((256, 8),):
            lg = (torch.randn(M, E, device=dev) * 2).bfloat16()
            x, W = inject(lg, sh_logit=(torch.randn(M, 1, device=dev) * 3).bfloat16())
            w0, i0 = vllm_route(lg, k, True)
            w1, i1, sh, _ = fr.fused_router(x, W, k, num_experts=E, cfg=(nt, nw, cs), **CL)
            ok = bool((i0 == i1).all()) and ((w0 - w1).abs() / w0).max().item() < 2e-6 and bool((sh == torch.sigmoid(x[:, 2047:2048].float()).bfloat16()).all())
            check(f"cluster cs={cs} nw={nw} M={M} injected logits (+ shared gate)", ok)
# E=128 (G=9 groups -> slots), ties, general path (sigmoid + bias + scale), padding rows
for E, k in ((128, 8), (64, 6)):
    lg = torch.randint(-3, 3, (24, E), device=dev).float().bfloat16() * 0.5
    x, W = inject(lg)
    w0, i0 = vllm_route(lg, k, True)
    w1, i1, _, _ = fr.fused_router(x, W, k, **CL)
    check(f"cluster E={E} k={k} ties M=24", bool((i0 == i1).all()) and ((w0 - w1).abs() / w0).max().item() < 2e-6)
lg = (torch.randn(5, 256, device=dev) * 2).bfloat16(); x, W = inject(lg); bias = torch.randn(256, device=dev)
w0, i0 = vllm_route(lg, 8, True, kind="sigmoid", bias=bias, scale=2.5)
w1, i1, _, _ = fr.fused_router(x, W, 8, scoring="sigmoid", bias=bias, scale=2.5, **CL)
check("cluster sigmoid + bias + scale", bool((i0 == i1).all()) and ((w0 - w1).abs() / w0).max().item() < 2e-6)
pad = torch.zeros(5, dtype=torch.bool, device=dev); pad[2] = True
w0, i0 = vllm_route(lg, 8, True, pad=pad); w1, i1, _, _ = fr.fused_router(x, W, 8, pad=pad, **CL)
check("cluster padding rows -> ids -1", bool((i0 == i1).all()))
# real weights vs exact logits (all 40 routers + MTP)
set_bad = 0; tot = 0; rows_exact = 0; ids_ok_on_exact = 0; same_as_splitk = 0; worst = 0.0
for layer in list(range(40)) + ["mtp"]:
    if layer == "mtp":
        Wg = real("mtp.layers.0.mlp.gate.weight"); lnw_ = real("mtp.layers.0.post_attention_layernorm.weight"); Wsh = real("mtp.layers.0.mlp.shared_expert_gate.weight")
    else:
        Wg = real(L + f"{layer}.mlp.gate.weight"); lnw_ = real(L + f"{layer}.post_attention_layernorm.weight"); Wsh = real(L + f"{layer}.mlp.shared_expert_gate.weight")
    Wc_ = torch.cat([Wg, Wsh]).contiguous()
    for M in (1, 3, 8, 16, 24):
        xm = normed_input(M, lnw_)
        ref_lg = (xm.double() @ Wg.double().T).float().bfloat16()
        wr, ir = vllm_route(ref_lg, 8, True)
        wc, ic, shc, lgc = fr.fused_router(xm, Wc_, 8, num_experts=256, write_logits=True, **CL)
        ws_, is_, shs, lgs_ = fr.fused_router(xm, Wc_, 8, num_experts=256, write_logits=True)
        tot += M
        set_bad += int((ic.sort(-1).values != ir.sort(-1).values).any(-1).sum())
        ex = (lgc == ref_lg).all(-1)
        rows_exact += int(ex.sum()); ids_ok_on_exact += int(((ic == ir).all(-1) & ex).sum())
        if ex.any():
            worst = max(worst, ((wc - wr).abs() / wr)[ex].max().item())
        same_as_splitk += int((lgc == lgs_).all(-1).sum())
print(f"   cluster: top-8 set differs from the exact-logit routing in {set_bad} of {tot} rows; rows with all 256 logits bit-exact {rows_exact}/{tot} "
      f"(ordered ids identical on {ids_ok_on_exact}/{rows_exact}, max rel weight diff {worst:.1e}); logits bitwise equal to the split-K kernel on {same_as_splitk}/{tot} rows")
check("cluster real weights: top-8 set == exact routing on >= 99.9% of rows", set_bad <= 0.001 * tot)
check("cluster real weights: bit-exact-logit rows route identically (ordered ids, weights < 1e-6 rel)", ids_ok_on_exact == rows_exact and worst < 1e-6)
# determinism, graph replay, concurrent load, counters not needed
xs = normed_input(3, lnw)
a1 = fr.fused_router(xs, Wc, 8, num_experts=256, **CL); a2 = fr.fused_router(xs, Wc, 8, num_experts=256, **CL)
check("cluster bitwise repeatable", bool((a1[0] == a2[0]).all() and (a1[1] == a2[1]).all()))
out_w = torch.empty(3, 8, device=dev); out_i = torch.empty(3, 8, dtype=torch.int32, device=dev)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    fr.fused_router(xs, Wc, 8, num_experts=256, out=(out_w, out_i), **CL)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    fr.fused_router(xs, Wc, 8, num_experts=256, out=(out_w, out_i), **CL)
okg = True
for _ in range(5):
    xs.copy_(normed_input(3, lnw)); g.replay(); torch.cuda.synchronize()
    ew, ei, _, _ = fr.fused_router(xs, Wc, 8, num_experts=256, **CL)
    okg &= bool((ew == out_w).all() and (ei == out_i).all())
check("cluster: CUDA graph replays with new inputs == eager", okg)
hog = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
s2 = torch.cuda.Stream(); okc = True
with torch.cuda.stream(s2):
    for _ in range(6):
        hog @ hog
for _ in range(20):
    xs = normed_input(16, lnw)
    r1 = fr.fused_router(xs, Wc, 8, num_experts=256, **CL); torch.cuda.synchronize()
    r2 = fr.fused_router(xs, Wc, 8, num_experts=256)
    okc &= bool((r1[1].sort(-1).values == r2[1].sort(-1).values).all())
torch.cuda.synchronize()
check("cluster completes and is correct while another stream saturates the SMs", okc)
print("FAILED:" if FAILS else "ALL PASSED", FAILS)
sys.exit(1 if FAILS else 0)
