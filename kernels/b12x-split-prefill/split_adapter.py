"""Opt-in standalone native NVFP4 phase chain with caller-owned buffers.

Nothing launches on import. Allocate Workspace and construct/compile SplitCall
outside capture. run() includes input quantization, all publisher work, both
phases and output clearing; no host counts or allocation based on routing.
"""
import sys
from pathlib import Path
from dataclasses import dataclass
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'adapter'))
from device_publisher import publish

@dataclass
class Workspace:
    m:int
    capacity_tiles:int
    q:torch.Tensor
    sf:torch.Tensor
    sf_bytes:torch.Tensor
    counts:torch.Tensor
    write_rows:torch.Tensor
    errors:torch.Tensor
    tile_bases:torch.Tensor
    packed_a:torch.Tensor
    scale_a:torch.Tensor
    token_map:torch.Tensor
    route_slots:torch.Tensor
    token_weights:torch.Tensor
    task_expert:torch.Tensor
    task_rows:torch.Tensor
    intermediate:torch.Tensor
    output:torch.Tensor

    @classmethod
    def allocate(cls,m,device):
        if m not in (1024,4096):raise ValueError('initial scope is M1024/4096')
        from vllm import _custom_ops as ops
        q,sf=ops.create_fp4_output_tensors(m,2048,device,True)
        ct=(m*8+127)//128+255;rows=ct*128
        alloc=lambda shape,dtype:torch.empty(shape,dtype=dtype,device=device)
        return cls(m,ct,q,sf,sf.view(torch.uint8).flatten(),
            alloc(256,torch.int32),alloc(256,torch.int32),alloc(1,torch.int32),alloc(257,torch.int32),
            alloc(rows*1024,torch.uint8),alloc(rows*128,torch.uint8),alloc(rows,torch.int32),
            alloc(rows,torch.int32),alloc(rows,torch.float32),alloc(ct*4,torch.int32),alloc(ct*4,torch.int32),
            alloc(rows*72,torch.int32),alloc((m,2048),torch.bfloat16))

class SplitCall:
    def __init__(self,w,prep,*,sm_count):
        # prep is the existing exact compressed-tensors preparation dictionary,
        # NOT upstream ModelOpt planner input. No reconstruction via raw reciprocal.
        self.w=w;self.prep=prep;self.sm_count=int(sm_count);self.compiled=None
        expected={'w13':((256,1024,1024),torch.uint8),'w2':((256,2048,256),torch.uint8),
                  's13':((256,1024,128),torch.float8_e4m3fn),'s2':((256,2048,32),torch.float8_e4m3fn),
                  'g1':((256,),torch.float32),'g2':((256,),torch.float32),
                  'a1_gscale':((256,),torch.float32),'a2_gscale':((256,),torch.float32)}
        for name,(shape,dtype) in expected.items():
            t=prep[name]
            assert tuple(t.shape)==shape and t.dtype==dtype and t.is_contiguous() and t.device==w.output.device,name
        assert w.output.is_cuda and 0<self.sm_count<=256
        # Exact current prep already selects a single calibrated a1 across experts.
        # Caller must validate uniform a1 outside capture, using assert_contract().
        self.a1=prep['a1_gscale'][:1]
        self.tensor_args1=[w.packed_a,w.scale_a,prep['w13'].flatten(),prep['s13'].view(torch.uint8).flatten(),
            w.intermediate,w.token_map,w.task_expert,w.task_rows,w.tile_bases,prep['g1'],prep['a2_gscale']]
        self.tensor_args2=[w.intermediate,prep['w2'].flatten(),prep['s2'].view(torch.uint8).flatten(),
            w.output,w.token_map,w.token_weights,w.task_expert,w.task_rows,w.tile_bases,prep['g2']]
        self.contract_checked=False

    def assert_contract(self):
        # Setup-only host checks; never invoked during capture or run().
        p=self.prep
        for name in ('a1_gscale','a2_gscale','g1','g2'):
            assert bool(torch.isfinite(p[name]).all()) and bool((p[name]>0).all()),name
        assert bool((p['a1_gscale']==p['a1_gscale'][0]).all())
        self.contract_checked=True

    def compile(self):
        assert self.contract_checked
        import cutlass,cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        import cuda.bindings.driver as cuda
        from nvfp4_phase1 import Nvfp4MaterializedPhase1Kernel
        from nvfp4_phase2 import Nvfp4MaterializedPhase2Kernel
        self.args1=[from_dlpack(x,assumed_align=16) for x in self.tensor_args1]
        self.args2=[from_dlpack(x,assumed_align=16) for x in self.tensor_args2]
        stream=cuda.CUstream(torch.cuda.current_stream(self.w.output.device).cuda_stream)
        self.ints1=[cutlass.Int32(x) for x in (16,4,8,self.sm_count)]
        self.ints2=[cutlass.Int32(x) for x in (4,16,self.sm_count)]
        self.compiled=(cute.compile(Nvfp4MaterializedPhase1Kernel(source_tile_m=128),*self.args1,*self.ints1,stream),
                       cute.compile(Nvfp4MaterializedPhase2Kernel(source_tile_m=128,deterministic_output=False),*self.args2,*self.ints2,stream))

    def run(self,x,ids,weights):
        if self.compiled is None:raise RuntimeError('compile and warm explicitly before capture')
        w=self.w
        assert tuple(x.shape)==(w.m,2048) and x.dtype==torch.bfloat16 and x.is_contiguous()
        assert tuple(ids.shape)==(w.m,8) and ids.dtype==torch.int32 and ids.is_contiguous()
        assert tuple(weights.shape)==(w.m,8) and weights.dtype==torch.float32 and weights.is_contiguous()
        assert x.device==ids.device==weights.device==w.output.device
        import cuda.bindings.driver as cuda
        # Same out kernel as vLLM scaled_fp4_quant, without its allocations.
        torch.ops._C.scaled_fp4_quant.out(x,self.a1,True,output=w.q,output_scale=w.sf)
        publish(ids,weights,w)
        w.intermediate.zero_();w.output.zero_()
        stream=cuda.CUstream(torch.cuda.current_stream(w.output.device).cuda_stream)
        self.compiled[0](*self.args1,*self.ints1,stream)
        self.compiled[1](*self.args2,*self.ints2,stream)
        return w.output
