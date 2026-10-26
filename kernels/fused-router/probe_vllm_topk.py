"""What exactly does vLLM's topk_softmax do? (arithmetic, tie-breaking, padding rows). Run on GPU."""
import torch
import vllm._custom_ops as ops
torch.manual_seed(0)
dev = "cuda"


def vllm_topk(logits, k, renorm, bias=None, pad=None, kind="softmax", scale=1.0):
    M = logits.shape[0]
    w = torch.empty(M, k, dtype=torch.float32, device=dev)
    i = torch.empty(M, k, dtype=torch.int32, device=dev)
    t = torch.empty(M, k, dtype=torch.int32, device=dev)
    if kind == "softmax":
        ops.topk_softmax(w, i, t, logits, renorm, bias, is_padding=pad)
    else:
        ops.topk_sigmoid(w, i, t, logits, renorm, bias, scale, is_padding=pad)
    return w, i


E, k = 256, 8
x = (torch.randn(4096, E, device=dev) * 2).bfloat16()
for renorm in (False, True):
    w, i = vllm_topk(x, k, renorm)
    xf = x.float()
    mx = xf.max(-1, keepdim=True).values
    e = torch.exp(xf - mx)
    s = e.sum(-1, keepdim=True)
    variants = {"e/sum": e / s, "e*(1/sum)": e * (1.0 / s), "torch.softmax": torch.softmax(xf, -1)}
    ids_ref = torch.topk(xf, k, dim=-1, sorted=True).indices.int()
    print(f"renorm={renorm}: ids equal torch.topk order on random logits: {(i == ids_ref).all(-1).float().mean().item():.4f}")
    for name, p in variants.items():
        sel = p.gather(1, i.long())
        if renorm:
            sel = sel / sel.sum(-1, keepdim=True)
        print(f"   {name:14s} bitwise equal weights: {(sel == w).float().mean().item():.4f}  max rel diff {((sel - w).abs() / w).max().item():.2e}")
        if renorm:
            sel2 = p.gather(1, i.long()); sel2 = sel2 * (1.0 / sel2.sum(-1, keepdim=True))
            print(f"   {name:14s} (renorm via reciprocal) bitwise: {(sel2 == w).float().mean().item():.4f}")
# ties: only 3 distinct values in a row
xt = torch.zeros(8, E, device=dev, dtype=torch.bfloat16)
xt[:, 10] = 1; xt[:, 200] = 1; xt[:, 5] = 1; xt[:, 100] = 1; xt[:, 7] = 0.5; xt[:, 250] = 0.5
w, i = vllm_topk(xt, k, True)
print("ties ids row0:", i[0].tolist(), " weights", [round(v, 6) for v in w[0].tolist()])
# padding rows
pad = torch.zeros(64, dtype=torch.bool, device=dev); pad[10:20] = True
w, i = vllm_topk(x[:64], k, True, pad=pad)
print("padding rows ids:", i[10].tolist(), "w:", w[10].tolist(), " unpadded row9:", i[9].tolist())
# sigmoid + bias + scale
b = torch.randn(E, device=dev)
w, i = vllm_topk(x[:64], k, True, bias=b, kind="sigmoid", scale=2.5)
sc = torch.sigmoid(x[:64].float())
ids_ref = torch.topk(sc + b, k, dim=-1).indices.int()
print("sigmoid+bias ids match torch.topk(score+bias):", (i == ids_ref).all(-1).float().mean().item())
sel = sc.gather(1, i.long()); sel = sel / sel.sum(-1, keepdim=True) * 2.5
print("   weights bitwise:", (sel == w).float().mean().item(), "max rel", ((sel - w).abs() / w).max().item())
w, i = vllm_topk(x[:64], k, False, bias=None, kind="sigmoid", scale=1.0)
sel = torch.sigmoid(x[:64].float()).gather(1, i.long())
print("sigmoid no-renorm bitwise:", (sel == w).float().mean().item(), "max rel", ((sel - w).abs() / w).max().item())
w, i = vllm_topk(x[:64], k, False, bias=b, kind="softmax")
print("softmax+bias ids match topk(p+bias)?:", (i == torch.topk(torch.softmax(x[:64].float(), -1) + b, k, -1).indices.int()).all(-1).float().mean().item(),
      " ids match topk(logit+bias)?:", (i == torch.topk(x[:64].float() + b, k, -1).indices.int()).all(-1).float().mean().item())
