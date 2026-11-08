"""Independent torch FP4 packing with ties-to-even and selectable E4M3 scale rounding."""
def pack_fp4(values):
    import torch
    mids=torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5.],device=values.device)
    absolute=values.abs().contiguous()
    code=torch.bucketize(absolute,mids,right=False)
    tie=(code<7)&(absolute==mids[code.clamp(max=6)])&((code%2)==1)
    code=(code+tie.to(code.dtype)).to(torch.uint8)
    code=code+torch.signbit(values).to(torch.uint8)*8
    return code[...,::2] | (code[...,1::2]<<4)

def quantize(values,dequant_scale,rounding):
    import torch
    assert rounding in ('ru','rn')
    shape=values.shape
    blocks=values.float().reshape(*shape[:-1],-1,16)
    raw=(blocks.abs().amax(-1)/6).clamp_min(1e-30)/dequant_scale.float()
    scale=raw.clamp(max=448).to(torch.float8_e4m3fn)
    if rounding=='ru':
        b=scale.view(torch.uint8).clone();b+=((scale.float()<raw)&(b<126)).to(torch.uint8)
        scale=b.view(torch.float8_e4m3fn)
    denom=scale.float()*dequant_scale.float()
    multiplier=torch.where(denom>0,1/denom,0)
    q=(blocks*multiplier.unsqueeze(-1)).reshape(shape)
    return pack_fp4(q),scale
