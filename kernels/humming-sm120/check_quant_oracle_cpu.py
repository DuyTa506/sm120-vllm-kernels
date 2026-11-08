import json,torch
from quant_oracle import pack_fp4,quantize
x=torch.tensor([[.25,.75,1.25,1.75,2.5,3.5,5.,-0.]])
assert pack_fp4(x).tolist()==[[32,66,100,134]]
y=torch.arange(-16,16).float().reshape(2,16)/7
rn,sn=quantize(y,torch.tensor(.125),'rn');ru,su=quantize(y,torch.tensor(.125),'ru')
assert bool((su.float()>=sn.float()).all())
assert quantize(torch.zeros(2,16),torch.tensor(.125),'rn')[0].count_nonzero()==0
print(json.dumps(dict(tie_even_pack_pass=True,zero_pass=True,cuda_initialized=torch.cuda.is_initialized())))
