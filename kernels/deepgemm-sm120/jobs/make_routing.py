# Recompute the real 8k-prefill routing (router GEMM + softmax + top-8) from captured hidden states; CPU only.
import torch, json
d=torch.load('/moe/data/moe_inputs.pt',map_location='cpu',weights_only=False)
out={}
for L in d['layers']:
    x=d['hidden'][L]; g=d['gate'][L]
    logits=torch.nn.functional.linear(x.float(),g.float())
    p=torch.softmax(logits,-1); w,ids=p.topk(8,-1)
    c=torch.bincount(ids.view(-1),minlength=256)
    print(L, 'total',int(c.sum()),'max',int(c.max()),'active',int((c>0).sum()),'mean',c.float().mean().item())
    out[str(L)]=dict(counts=c.tolist())
    torch.save(ids.to(torch.int32),f'/out/ids_L{L}.pt')
json.dump(out,open('/out/routing_counts.json','w'))
