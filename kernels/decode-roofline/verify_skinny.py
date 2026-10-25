"""Independent verification of the skinny BF16 GEMM (VLLM_SKINNY_BF16_GEMM).

Run inside the vLLM image with the repo at /work and the patched files bind-mounted over the image
(so torch.ops.vllm.skinny_bf16_linear is registered exactly as the patch does it):
  -v .../decode-roofline/vllm/model_executor/layers/utils.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/utils.py:ro
  -v .../decode-roofline/vllm/model_executor/layers/skinny_bf16_gemm.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/skinny_bf16_gemm.py:ro

Checks
 1. All 40 router gates, the MTP router and all 30 GDN in_proj_ba (b|a merged like vLLM) from the real
    checkpoint, through the custom op (with its own M/N gating), against F.linear (cuBLAS) and an fp32 ref,
    on RMSNorm-ed inputs that use the real layernorm weights. Router: top-8 expert-set agreement.
    in_proj_ba: downstream beta=sigmoid(b) and g=-exp(A_log)*softplus(a+dt_bias) error.
 2. Edge cases: M=0, M=17 (fallback), non-unit last-dim stride, row-strided weight view, bias, 3-D.
 3. Determinism (bitwise repeat), CUDA-graph capture + replay with new data, torch.compile with dynamic M.
 4. Timing with weights cycled through > L2 (128 MB) so calls read DRAM like in a real decode step,
    CUDA graph of 256 calls, medians of 30 replays, both orders.
"""
import glob, json, statistics
import torch
import torch.nn.functional as F
from safetensors import safe_open

import vllm.model_executor.layers.utils as U  # registers torch.ops.vllm.skinny_bf16_linear
from vllm.model_executor.layers.skinny_bf16_gemm import skinny_bf16_gemm

op = torch.ops.vllm.skinny_bf16_linear
torch.manual_seed(1234)
dev = "cuda"
snap = glob.glob("/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/*/")[0]
wmap = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]


def real(name):
    with safe_open(snap + wmap[name], "pt", device=dev) as f:
        return f.get_tensor(name)


L = "model.language_model.layers."


def normed_input(rows, ln_w):
    # residual stream with a few large outlier channels, then Qwen3-Next RMSNorm (x * (1 + w))
    h = torch.randn(rows, 2048, device=dev)
    h[:, torch.randint(0, 2048, (12,))] *= 20
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    return (h * (1 + ln_w.float())).bfloat16()


print("== 1. real weights, all layers, through torch.ops.vllm.skinny_bf16_linear")
routers = [(f"router L{i}", real(L + f"{i}.mlp.gate.weight"), real(L + f"{i}.post_attention_layernorm.weight"), None)
           for i in range(40)]
routers.append(("router MTP", real("mtp.layers.0.mlp.gate.weight"), real("mtp.layers.0.post_attention_layernorm.weight"), None))
bas = []
for i in range(40):
    k = L + f"{i}.linear_attn.in_proj_b.weight"
    if k in wmap:
        W = torch.cat([real(k), real(L + f"{i}.linear_attn.in_proj_a.weight")]).contiguous()
        bas.append((f"in_proj_ba L{i}", W, real(L + f"{i}.input_layernorm.weight"),
                    (real(L + f"{i}.linear_attn.A_log"), real(L + f"{i}.linear_attn.dt_bias"))))
print(f"{len(routers)} router weights, {len(bas)} in_proj_ba weights")

ROWS = 1536  # processed as many independent calls of M rows
worst = {}
topk_flip_sk = topk_flip_ref = topk_tot = 0
gerr = berr = 0.0
gerr_cub = berr_cub = 0.0
for M in (1, 2, 3, 4, 6, 9, 12, 15, 16):
    agg = dict(max_d_sk=0.0, max_d_cub=0.0, max_rel=0.0, eq=0, n=0, taken=0)
    for name, W, lnw, gdn in routers + bas:
        x_all = normed_input(ROWS - ROWS % M, lnw)
        ys, yc = [], []
        for r in range(0, x_all.shape[0], M):
            x = x_all[r:r + M]
            ys.append(op(x, W, None))
            yc.append(F.linear(x, W))
        ys, yc = torch.cat(ys), torch.cat(yc)
        ref = F.linear(x_all.float(), W.float())
        agg["taken"] += int(U._use_skinny_bf16_gemm(x_all[:M], W, None))
        d_sk = (ys.float() - ref).abs().max().item()
        d_cub = (yc.float() - ref).abs().max().item()
        agg["max_d_sk"] = max(agg["max_d_sk"], d_sk)
        agg["max_d_cub"] = max(agg["max_d_cub"], d_cub)
        agg["max_rel"] = max(agg["max_rel"], d_sk / ref.abs().max().item())
        agg["eq"] += (ys == yc).sum().item(); agg["n"] += ys.numel()
        if name.startswith("router") and M == 3:
            t_sk = torch.topk(torch.softmax(ys.float(), -1), 8).indices.sort(-1).values
            t_cb = torch.topk(torch.softmax(yc.float(), -1), 8).indices.sort(-1).values
            t_rf = torch.topk(torch.softmax(ref.bfloat16().float(), -1), 8).indices.sort(-1).values
            topk_flip_sk += (t_sk != t_cb).any(-1).sum().item()
            topk_flip_ref += (t_rf != t_cb).any(-1).sum().item()
            topk_tot += t_sk.shape[0]
        if gdn is not None and M == 3:
            A_log, dt_bias = gdn
            def post(y):
                b, a = y.float()[:, :32], y.float()[:, 32:]
                return torch.sigmoid(b), -torch.exp(A_log.float()) * F.softplus(a + dt_bias.float())
            bs, gs = post(ys); bc, gc = post(yc); br, gr = post(ref.bfloat16())
            berr = max(berr, (bs - br).abs().max().item()); gerr = max(gerr, (gs - gr).abs().max().item())
            berr_cub = max(berr_cub, (bc - br).abs().max().item()); gerr_cub = max(gerr_cub, (gc - gr).abs().max().item())
    print(f"M={M:2d}: skinny taken for {agg['taken']:2d}/{len(routers)+len(bas)} weights | max|y-ref| op {agg['max_d_sk']:.3e} "
          f"cuBLAS {agg['max_d_cub']:.3e} | max rel {agg['max_rel']:.2e} | bitwise==cuBLAS {agg['eq']/agg['n']:.4f}")
print(f"router top-8 set differs (op vs cuBLAS) in {topk_flip_sk}/{topk_tot} rows at M=3; "
      f"(bf16(fp32 ref) vs cuBLAS, the noise floor: {topk_flip_ref}/{topk_tot})")
print(f"in_proj_ba downstream at M=3, max err vs bf16(ref): beta op {berr:.2e} cuBLAS {berr_cub:.2e}; g op {gerr:.2e} cuBLAS {gerr_cub:.2e}")

print("\n== 2. edge cases")
W = routers[0][1]; Wb = bas[0][1]
x0 = torch.empty(0, 2048, device=dev, dtype=torch.bfloat16)
print("M=0 ->", tuple(op(x0, W, None).shape))
x17 = normed_input(17, routers[0][2])
print("M=17 uses skinny?", U._use_skinny_bf16_gemm(x17, Wb, None), " equal to F.linear:", torch.equal(op(x17, Wb, None), F.linear(x17, Wb)))
xt = normed_input(2048, routers[0][2]).t()[:3]          # [3, 2048] with stride(-1) = 2048
print("non-unit last stride, max|d| vs F.linear:", (op(xt, W, None).float() - F.linear(xt, W).float()).abs().max().item())
Wbig = torch.randn(64, 4096, device=dev).bfloat16()
Wv = Wbig[:, :2048]                                        # row stride 4096
x3 = normed_input(3, routers[0][2])
print("row-strided weight view, max|d|:", (op(x3, Wv, None).float() - F.linear(x3, Wv).float()).abs().max().item())
bias = torch.randn(256, device=dev).bfloat16()
print("bias, max|d|:", (op(x3, W, bias).float() - F.linear(x3, W, bias).float()).abs().max().item())
x3d = normed_input(6, routers[0][2]).view(2, 3, 2048)
print("3-D in_proj_ba, max|d|:", (op(x3d, Wb, None).float() - F.linear(x3d, Wb).float()).abs().max().item())
for N, K in ((16, 2048), (100, 2048), (512, 2048), (64, 1000), (128, 4096), (256, 2048 + 8)):
    Wr = (torch.randn(N, K, device=dev) * 0.02).bfloat16()
    for M in (1, 3, 7, 16):
        x = torch.randn(M, K, device=dev).bfloat16()
        y = skinny_bf16_gemm(x, Wr); r = F.linear(x.float(), Wr.float())
        rel = (y.float() - r).abs().max().item() / r.abs().max().item()
        if rel > 8e-3:
            print(f"  ODD SHAPE FAIL N={N} K={K} M={M} rel={rel:.2e}")
print("odd N/K shapes (kernel called directly, M in 1,3,7,16): checked, failures printed above if any")

print("\n== 3. determinism / CUDA graph / torch.compile")
x = normed_input(3, routers[0][2])
a = op(x, W, None); b = op(x, W, None)
print("bitwise repeatable:", torch.equal(a, b))
xs = x.clone(); out = {}
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    op(xs, W, None); op(xs, Wb, None)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out["r"] = op(xs, W, None); out["b"] = op(xs, Wb, None)
new = normed_input(3, routers[0][2]); xs.copy_(new); g.replay(); torch.cuda.synchronize()
print("graph replay with new input equals eager op:", torch.equal(out["r"], op(new, W, None)), torch.equal(out["b"], op(new, Wb, None)))


def f(x, w):
    return torch.relu(op(x, w, None)) * 2


cf = torch.compile(f, dynamic=True, fullgraph=True)
ok = True
for M in (1, 3, 5, 12, 16, 40, 300):
    xx = normed_input(M, routers[0][2])
    ok &= torch.equal(cf(xx, Wb), f(xx, Wb)) and torch.equal(cf(xx, W), f(xx, W))
print("torch.compile(dynamic, fullgraph) matches eager for M in 1..300:", ok)

print("\n== 4. timing: CUDA graph of 256 calls cycling weight copies (> L2), median of 30 replays")


def time_graph(fns, reps=30):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for fn in fns:
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for fn in fns:
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / len(fns))
    return statistics.median(ts)


for name, W in (("router 256x2048", routers[0][1]), ("in_proj_ba 64x2048", bas[0][1])):
    ncopy = max(1, (384 << 20) // W.numel() // 2)
    copies = [W.clone() for _ in range(ncopy)]
    for M in (1, 2, 3, 6, 12):
        x = normed_input(M, routers[0][2])
        seq = [copies[i % ncopy] for i in range(256)]
        r = []
        for order in (0, 1):
            fc = [lambda c=c: F.linear(x, c) for c in seq]
            fo = [lambda c=c: op(x, c, None) for c in seq]
            if order == 0:
                tc = time_graph(fc); to = time_graph(fo)
            else:
                to = time_graph(fo); tc = time_graph(fc)
            r.append((tc, to))
        taken = U._use_skinny_bf16_gemm(x, W, None)
        print(f"{name:20s} M={M:2d} ({'skinny' if taken else 'F.linear fallback'}): cuBLAS {r[0][0]:6.2f}/{r[1][0]:6.2f} us, "
              f"op {r[0][1]:6.2f}/{r[1][1]:6.2f} us  (two orders), {ncopy} copies")
    del copies
    torch.cuda.empty_cache()
