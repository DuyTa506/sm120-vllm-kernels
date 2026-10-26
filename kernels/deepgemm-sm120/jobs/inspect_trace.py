import torch
d=torch.load('/moe/data/moe_inputs.pt',map_location='cpu',weights_only=False)
print(type(d), list(d.keys()) if isinstance(d,dict) else '')
for k,v in d.items():
    if hasattr(v,'shape'): print(k,v.shape,v.dtype)
    elif isinstance(v,dict):
        print(k,{kk:(vv.shape if hasattr(vv,'shape') else vv) for kk,vv in v.items()})
    else: print(k,type(v), str(v)[:100])
