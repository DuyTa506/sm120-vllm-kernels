"""Checkpoint -> explicit qf contract. No installed FlashInfer mutations."""
import torch

def swizzle(scale):
 e,n,k=scale.shape
 assert n%128==0 and k%4==0 and scale.dtype==torch.float8_e4m3fn
 return scale.reshape(e,n//128,4,32,k//4,4).permute(0,1,4,3,2,5).contiguous().reshape(e,n,k)

def prepare_raw(raw):
 # raw uses gate/up; CUTLASS/qf uses up/gate. No b12x permutation input.
 w=raw['w13'];s=raw['s13'];i=w.shape[1]//2
 w13=torch.cat([w[:,i:],w[:,:i]],1).contiguous()
 s13=torch.cat([s[:,i:],s[:,:i]],1).contiguous()
 a1=raw['igs13'].float().reciprocal().amax().reshape(1)
 a2=raw['igs2'].float().reciprocal().amax().reshape(1)
 return dict(w13=w13,s13=swizzle(s13),w2=raw['w2'].contiguous(),s2=swizzle(raw['s2']),
             g1=raw['wgs13'].float().reciprocal()*a1,g2=raw['wgs2'].float().reciprocal()*a2,
             a1=a1.reciprocal(),a2=a2.reciprocal())

def workspace(device):
 # One object per concurrent caller/stream, retained across graph lifetime.
 return dict(hq=torch.empty(64*8*64,device=device,dtype=torch.int32),
             hsf=torch.empty(64*8*32,device=device,dtype=torch.float32),
             part=torch.empty(64*8*2048,device=device,dtype=torch.float32),
             done=torch.empty(16,device=device,dtype=torch.int32))

def call(module,prepared,x,ids,weights,work,out,reuse=True):
 return module.apply(x,ids,weights,prepared['w13'],prepared['s13'].view(torch.uint8),prepared['g1'],prepared['a1'],
                     prepared['w2'],prepared['s2'].view(torch.uint8),prepared['g2'],prepared['a2'],
                     work['hq'],work['hsf'],work['part'],work['done'],out,reuse)
