"""Run directly; coverage/duplicate gates at tile0 and full output width."""
import copy
import torch
from compare import canonical
for width in (128,2048):
    p=torch.full((2,128,width),float('nan'),dtype=torch.bfloat16)
    m=torch.full((2,128,4),-1,dtype=torch.int32)
    for group in range(2):
        for token in range(2):p[group,token].fill_(group+1);m[group,token]=torch.tensor([token,0,group*2,2])
    d=dict(payload=p,metadata=m,ids=torch.zeros((2,1),dtype=torch.int32),counts=torch.tensor([2]),split=1,tiles=1,output=torch.zeros((2,width)))
    k,v,r=canonical(d);assert len(k)==4 and torch.equal(r,torch.full((2,width),3,dtype=torch.float64))
    for corrupt in ('missing','duplicate','column','tail'):
        bad=copy.deepcopy(d)
        if corrupt=='missing':bad['metadata'][0,0].fill_(-1);bad['payload'][0,0].fill_(float('nan'))
        if corrupt=='duplicate':bad['metadata'][0,1]=bad['metadata'][0,0]
        if corrupt=='column':bad['payload'][0,0,width-1]=float('nan')
        if corrupt=='tail':bad['payload'][0,127,width-1]=0
        try:canonical(bad)
        except AssertionError:pass
        else:raise AssertionError('gate accepted '+corrupt)
print('both widths: canonical coverage, duplicate, last-column, tail and FP64 reference checks passed')
