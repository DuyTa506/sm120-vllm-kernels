"""CPU-only: relative error of expert matmuls when the weight global scale is baked into e4m3 block scales
(vLLM FlashInferB12xExperts) vs the exact dequantized NVFP4 weights the CUTLASS path uses.
x ~ N(0,1) (relative error of W x is insensitive to the input scale). usage: ... [layer ...]"""
import sys, torch
sys.path.insert(0, "/work/kernels/moe-nvfp4")
from moe_common import get_tensor, _FP4_LUT

def deq(packed, scale_f32):
    lo, hi = _FP4_LUT[(packed & 0xF).long()], _FP4_LUT[(packed >> 4).long()]
    w = torch.stack([lo, hi], -1).flatten(-2)
    N, K = w.shape
    return (w.view(N, K // 16, 16) * scale_f32.unsqueeze(-1)).view(N, K)

torch.manual_seed(0)
for L in [int(a) for a in sys.argv[1:]] or [0, 19, 39]:
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        errs = []
        for e in range(0, 256, 16):
            p = f"model.language_model.layers.{L}.mlp.experts.{e}.{proj}."
            q, s, gs = get_tensor(p + "weight_packed"), get_tensor(p + "weight_scale").float(), get_tensor(p + "weight_global_scale").float()
            W = deq(q, s / gs)
            Wb = deq(q, (s / gs).to(torch.float8_e4m3fn).float())
            x = torch.randn(W.shape[1], 64)
            y, yb = W @ x, Wb @ x
            errs.append(((yb - y).norm() / y.norm()).item())
        print(f"layer {L:2d} {proj:9s}: rel L2 error of W@x with baked scales: mean {sum(errs)/len(errs):.3f} max {max(errs):.3f}")
