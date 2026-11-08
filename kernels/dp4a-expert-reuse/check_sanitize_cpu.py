"""Execute exact Triton kernel under interpreter on CPU; no CUDA initialization."""
import os
assert os.environ.get('TRITON_INTERPRET')=='1'
import torch
from direct_sanitize import _sanitize
assert not torch.cuda.is_initialized()
for m in (1,3,4,8):
 x=torch.arange(m*2048,dtype=torch.float32).reshape(m,2048).to(torch.bfloat16)
 ids=torch.arange(8,dtype=torch.int32).expand(m,8).clone();w=torch.ones((m,8),dtype=torch.float32)
 for fixture in ('valid','all_invalid_nan','partial','malformed','restore'):
  ids[:]=torch.arange(8);w.fill_(1);x.fill_(2)
  if fixture=='all_invalid_nan':ids.fill_(-1);w.fill_(float('nan'));x.fill_(float('nan'))
  elif fixture=='partial':ids[:,::2]=-1;w[:,::2]=float('nan')
  elif fixture=='malformed':ids[:,0]=-2;ids[:,1]=256
  sx=torch.empty_like(x);si=torch.empty_like(ids);sw=torch.empty_like(w);err=torch.empty((m,),dtype=torch.int32)
  _sanitize[(m,)](x,ids,w,sx,si,sw,err,H=2048,num_warps=4)
  valid=(ids>=0)&(ids<256)
  assert torch.equal(si,torch.where(valid,ids,0))
  assert torch.equal(sw,torch.where(valid,w,0))
  assert torch.equal(sx,torch.where(valid.any(dim=1)[:,None],x,0))
  assert torch.equal(err,((ids < -1)|(ids>=256)).sum(dim=1).int())
assert not torch.cuda.is_initialized()
print('20 exact Triton-interpreter fixtures pass; CUDA initialized false')
