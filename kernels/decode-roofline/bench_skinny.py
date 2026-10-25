"""Correctness + timing: skinny BF16 GEMM vs torch F.linear (cuBLAS) on the real BF16 decode shapes of
Qwen3.6-35B-A3B, with real weights where they exist in the checkpoint.

usage (in the vLLM image, repo mounted at /work): python3 kernels/decode-roofline/bench_skinny.py
Timing: CUDA graphs of 20 back-to-back calls (each call reads a different weight copy so the weights
come from DRAM, not L2), median of 30 graph replays, reported per call.
"""
import glob, json, os, sys
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skinny_bf16_gemm import skinny_bf16_gemm  # noqa: E402

snap = glob.glob("/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/*/")[0]
wmap = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]


def real(name):
    with safe_open(snap + wmap[name], "pt", device="cuda") as f:
        return f.get_tensor(name)


L = "model.language_model.layers."
shapes = {
    # name: (weight tensor, calls per verify step)
    "router gate 256x2048": (real(L + "0.mlp.gate.weight"), 40),
    "GDN in_proj_ba 64x2048": (torch.cat([real(L + "0.linear_attn.in_proj_b.weight"),
                                          real(L + "0.linear_attn.in_proj_a.weight")]), 30),
    "shared_expert_gate 1x2048": (real(L + "0.mlp.shared_expert_gate.weight"), 40),
    "MTP q_proj 8192x2048": (real("mtp.layers.0.self_attn.q_proj.weight"), 0),
    "MTP fc 2048x4096": (real("mtp.fc.weight"), 0),
    "MTP o_proj 2048x4096": (real("mtp.layers.0.self_attn.o_proj.weight"), 0),
    "MTP shared gate_up 1024x2048": (torch.cat([real("mtp.layers.0.mlp.shared_expert.gate_proj.weight"),
                                               real("mtp.layers.0.mlp.shared_expert.up_proj.weight")]), 0),
}

# realistic activations: hidden states after RMSNorm have O(1) entries with heavy-ish tails
torch.manual_seed(0)


def act(M, K):
    x = torch.randn(M, K, device="cuda") * (1 + 3 * (torch.rand(1, K, device="cuda") < 0.01))
    return x.bfloat16()


def time_graph(fn_list, reps=30):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fn_list:
            f()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fn_list:
            f()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / len(fn_list))
    ts.sort()
    return ts[len(ts) // 2]


print(f"{'shape':30s} {'M':>3s} {'cuBLAS us':>10s} {'skinny us':>10s} {'GB/s':>6s}  max|d|   max rel   ref|y|max")
tot_saved = 0.0
for name, (W, calls) in shapes.items():
    W = W.contiguous()
    N, K = W.shape
    copies = [W.clone() for _ in range(20)]
    for M in (1, 2, 3, 4, 6, 9, 12, 16):
        x = act(M, K)
        ref = F.linear(x.float(), W.float())          # fp32 reference
        y_cub = F.linear(x, W)
        y_sk = skinny_bf16_gemm(x, W)
        d_sk = (y_sk.float() - ref).abs().max().item()
        d_cub = (y_cub.float() - ref).abs().max().item()
        rel = d_sk / max(ref.abs().max().item(), 1e-9)
        t_cub = time_graph([lambda c=c: F.linear(x, c) for c in copies])
        t_sk = time_graph([lambda c=c: skinny_bf16_gemm(x, c) for c in copies])
        print(f"{name:30s} {M:3d} {t_cub:10.2f} {t_sk:10.2f} {N*K*2/t_sk/1e3:6.0f}  {d_sk:.2e} {rel:.2e} {ref.abs().max().item():8.3f}"
              f"   (cuBLAS err {d_cub:.2e}, skinny==cuBLAS bitwise frac {(y_sk == y_cub).float().mean().item():.3f})")
        if M == 3:
            tot_saved += calls * (t_cub - t_sk)
# varlen / strided input: rows of a wider buffer (as produced by splits)
xb = act(3, 4096)[:, :2048]
W = shapes["router gate 256x2048"][0]
assert torch.allclose(skinny_bf16_gemm(xb, W).float(), F.linear(xb, W).float(), atol=2e-2, rtol=2e-2)
b = torch.randn(W.shape[0], device="cuda").bfloat16()
x5 = act(5, 2048)
assert torch.allclose(skinny_bf16_gemm(x5, W, b).float(), F.linear(x5, W, b).float(), atol=2e-2, rtol=2e-2)
x3d = act(6, 2048).view(2, 3, 2048)
assert skinny_bf16_gemm(x3d, W).shape == (2, 3, W.shape[0])
print("strided-input, bias and 3-D input checks passed")
print(f"\nestimated target-forward saving at M=3 (router x40 + in_proj_ba x30 + shared gate x40): {tot_saved:.0f} us/step")
