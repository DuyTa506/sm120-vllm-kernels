"""Run each mode in a fresh CUDA-hidden interpreter. No kernel compile or GPU work."""
import json,sys
import torch
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as d,moe_dynamic_kernel as f
from plugin import install
mode=sys.argv[1]
kw=dict(activation_precision='fp4',quant_mode='nvfp4',E=256,k=2048,n=512,num_topk=8,mac=188,mma_tiler_mn=(128,128),topk_ids_dtype=torch.int32,input_scales_are_reciprocal=True,fast_math=False,activation='silu',swiglu_alpha=1.702,swiglu_beta=1.0,swiglu_limit=None,share_input_across_experts=True)
original=d._dynamic_kernel_cache_key(**kw);static=d._static_kernel_cache_key;micro=d._micro_kernel_cache_key
if mode=='dirty':
 d._DYNAMIC_KERNEL_CACHE['fixture']=object()
 try:install('on')
 except AssertionError as exc:assert 'cache must be empty' in str(exc)
 else:raise AssertionError('dirty cache accepted')
 print(json.dumps(dict(mode=mode,dirty_cache_rejected=True,cuda_initialized=torch.cuda.is_initialized())))
else:
 result=install(mode);new=d._dynamic_kernel_cache_key(**kw)
 assert new[:-2]==original and new[-2]=='defined_fc2_source_sha256'
 assert static is d._static_kernel_cache_key and micro is d._micro_kernel_cache_key
 for change in [dict(k=4096),dict(n=1024),dict(mma_tiler_mn=(64,128)),dict(quant_mode='w4a16')]:
  try:d._dynamic_kernel_cache_key(**(kw|change))
  except AssertionError:pass
  else:raise AssertionError('unsupported geometry accepted')
 obj=f.MoEGatedDynamicKernel(16,(128,128))
 assert obj.mma_register_requirement==232 and not torch.cuda.is_initialized()
 result.update(key=repr(new),unmodified_geometry_fields=True,static_micro_unchanged=True,unsupported_rejected=True,cuda_initialized=False)
 print(json.dumps(result))
