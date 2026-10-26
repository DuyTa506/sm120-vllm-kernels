"""Where does the fused kernel's logit error come from? Compare bf16 logits of cuBLAS, the Triton skinny GEMM and the
fused kernel (several configs) against an fp64 reference, on real router weights, with and without outlier channels."""
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, "/dr")
from skinny_bf16_gemm import skinny_bf16_gemm
from common import real, L
import fused_router as fr
dev = "cuda"
torch.manual_seed(0)


def inputs(rows, ln_w, outliers):
    h = torch.randn(rows, 2048, device=dev)
    if outliers:
        h[:, torch.randint(0, 2048, (12,), device=dev)] *= 20
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    return (h * (1 + ln_w.float())).bfloat16()


def stats(name, lg, ref64):
    # lg bf16 [M,256]; ref64 fp64 exact logits. mismatch vs bf16 rounding of the exact value
    ref_bf = ref64.float().bfloat16()
    mism = (lg != ref_bf).float().mean().item()
    err = (lg.double() - ref64).abs()
    ulp = (ref64.abs().clamp_min(1e-30).log2().floor().exp2() * 2 ** -7)   # bf16 ulp of the exact value
    return mism, (err / ulp).max().item(), (err / ulp).mean().item()


for outliers in (True, False):
    print(f"== outlier channels: {outliers}")
    acc = {}
    for layer in (0, 7, 23, 39):
        Wg = real(L + f"{layer}.mlp.gate.weight"); lnw = real(L + f"{layer}.post_attention_layernorm.weight")
        Wc = torch.cat([Wg, real(L + f"{layer}.mlp.shared_expert_gate.weight")]).contiguous()
        x = inputs(48, lnw, outliers)
        ref = x.double() @ Wg.double().T
        print(f"   layer {layer}: |logit| median {ref.abs().median().item():.3f} max {ref.abs().max().item():.2f}")
        cands = {
            "cuBLAS F.linear": F.linear(x[:3], Wg),
            "triton skinny": skinny_bf16_gemm(x[:3], Wg),
            "fp32 F.linear->bf16": F.linear(x[:3].float(), Wg.float()).bfloat16(),
        }
        for split in (1, 4, 8, 16):
            for am in (0, 1):
                cands[f"fused split={split} accmode={am}"] = fr.fused_router(x[:3], Wc, 8, num_experts=256, write_logits=True, cfg=(1, 4, split), accmode=am)[3]
        cands["fused nt=3 split=8 M=24 accmode=0"] = None; cands["fused nt=3 split=8 M=24 accmode=1"] = None
        r3 = ref[:3]
        for k, v in cands.items():
            if v is None:
                v = fr.fused_router(x[:24], Wc, 8, num_experts=256, write_logits=True, cfg=(3, 16, 8), accmode=int(k[-1]))[3]
                m, mx, mean = stats(k, v, ref[:24]); 
            else:
                m, mx, mean = stats(k, v, r3)
            a = acc.setdefault(k, []); a.append((m, mx, mean))
    for k, v in acc.items():
        print(f"   {k:26s} mismatch vs round(exact) {sum(a[0] for a in v)/len(v):.4f}   max err {max(a[1] for a in v):.2f} ulp   mean err {sum(a[2] for a in v)/len(v):.3f} ulp")
