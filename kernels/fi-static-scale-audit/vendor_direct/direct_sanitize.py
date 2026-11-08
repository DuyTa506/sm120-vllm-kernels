"""Graph-safe token/route sanitization for FI direct's valid-ID-only ABI.

Legal IDs: -1 padding or [0,255]. Each call writes every destination and one
malformed-ID count per token. Count checking is diagnostic-only outside graphs.
No allocations or host reads in __call__. Always included in adapter timing.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _sanitize(X, IDS, W, SX, SI, SW, ERR, H:tl.constexpr):
 t=tl.program_id(0)
 slots=tl.arange(0,8)
 ids=tl.load(IDS+t*8+slots)
 valid=(ids>=0)&(ids<256)
 malformed=(ids < -1)|(ids>=256)
 any_valid=tl.sum(valid.to(tl.int32),0)>0
 weights=tl.load(W+t*8+slots)
 tl.store(SI+t*8+slots,tl.where(valid,ids,0))
 tl.store(SW+t*8+slots,tl.where(valid,weights,0.0))
 cols=tl.arange(0,H)
 values=tl.load(X+t*H+cols)
 tl.store(SX+t*H+cols,tl.where(any_valid,values,0.0))
 tl.store(ERR+t,tl.sum(malformed.to(tl.int32),0))

class Sanitizer:
 def __init__(self,x,ids,weights):
  if x.dtype!=torch.bfloat16 or ids.dtype not in (torch.int32,torch.int64) or weights.dtype!=torch.float32:
   raise ValueError('expected BF16 x, int32/int64 IDs and FP32 route weights')
  m=x.shape[0]
  if x.shape!=(m,2048) or ids.shape!=(m,8) or weights.shape!=(m,8) or m not in (1,3,4,8):
   raise ValueError('unsupported shape')
  if not all(t.is_cuda and t.is_contiguous() and t.device==x.device for t in (x,ids,weights)):
   raise ValueError('expected contiguous same-device CUDA operands')
  self.x=x;self.ids=ids;self.weights=weights;self.m=m
  self.safe_x=torch.empty_like(x);self.safe_ids=torch.empty((m,8),device=x.device,dtype=torch.int32)
  self.safe_weights=torch.empty_like(weights);self.errors=torch.empty((m,),device=x.device,dtype=torch.int32)
  self.bytes=sum(t.numel()*t.element_size() for t in (self.safe_x,self.safe_ids,self.safe_weights,self.errors))
 def __call__(self,x=None,ids=None,weights=None):
  if x is None:x,ids,weights=self.x,self.ids,self.weights
  if x is None or ids is None or weights is None:raise ValueError('live tensors required')
  if x.shape!=(self.m,2048) or ids.shape!=(self.m,8) or weights.shape!=(self.m,8):raise ValueError('live shape differs from prepared handle')
  if x.dtype!=torch.bfloat16 or ids.dtype not in (torch.int32,torch.int64) or weights.dtype!=torch.float32:raise ValueError('live dtype differs from contract')
  if not all(t.is_cuda and t.is_contiguous() and t.device==self.safe_x.device for t in (x,ids,weights)):raise ValueError('live device/stride differs from contract')
  _sanitize[(self.m,)](x,ids,weights,self.safe_x,self.safe_ids,self.safe_weights,self.errors,H=2048,num_warps=4)

class SanitizedCall:
 def __init__(self,sanitizer,prepared):self.sanitizer=sanitizer;self.prepared=prepared
 def __call__(self):
  self.sanitizer()
  return self.prepared()

def prepare(dispatch,expert,prep,x,ids,weights,out,*,native=False,multiplier=True):
 from fi_plan import Plan
 from direct_native import DirectNative
 sanitizer=Sanitizer(x,ids,weights)
 plan=Plan(dispatch,expert,prep,sanitizer.safe_x,sanitizer.safe_ids,sanitizer.safe_weights,out,'direct_micro',direct_multiplier=multiplier)
 if native:plan=DirectNative(dispatch,plan)
 return SanitizedCall(sanitizer,plan)

class LiveDirect:
 """Serving-shaped interface: no retained external warmup data/output pointers."""
 def __init__(self, prepared):
  from direct_native import DirectNative
  if not isinstance(prepared.prepared,DirectNative):raise ValueError('native prepared handle required')
  self.sanitizer=prepared.sanitizer;self.native=prepared.prepared
  self.sanitizer.x=None;self.sanitizer.ids=None;self.sanitizer.weights=None
  self.native.kw.pop('out')
 def __call__(self,x,ids,weights,out):
  if out.shape!=(self.sanitizer.m,2048) or out.dtype!=torch.bfloat16 or not out.is_cuda or not out.is_contiguous() or out.device!=self.sanitizer.safe_x.device:
   raise ValueError('live output differs from prepared contract')
  # Output must not alias the private scratch. External x/output alias is also
  # rejected: callers should preserve original input across competing paths.
  for t in (x,ids,weights,self.sanitizer.safe_x,self.sanitizer.safe_ids,self.sanitizer.safe_weights):
   lo,hi=t.data_ptr(),t.data_ptr()+t.numel()*t.element_size()
   if max(lo,out.data_ptr())<min(hi,out.data_ptr()+out.numel()*out.element_size()):raise ValueError('output aliases input/scratch')
  self.sanitizer(x,ids,weights)
  return self.native(out=out)
