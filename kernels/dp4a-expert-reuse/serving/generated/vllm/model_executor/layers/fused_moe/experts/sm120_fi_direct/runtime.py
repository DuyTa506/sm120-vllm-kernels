"""Isolated experimental M1/M3/M4 FI direct overlay; default off, not serving-validated."""
import hashlib,json,time
from pathlib import Path
import torch
from vllm import envs

PINNED_DISPATCH='c518e65d6bfd7f08db1e5261e20795fd020e82e681699729171bc2fd5331239a'
PINNED_KERNEL='9b18f54588c3f24cab74dff07545f323e2af8235a4532233655a5a75498e89cc'

def eligible(expert,x,activation,global_experts,expert_map,router_input):
 parallel=expert.moe_config.moe_parallel_config
 return (x.shape[0] in (1,3,4) and x.shape[1]==2048 and x.dtype==torch.bfloat16
  and expert.hidden_dim==2048 and expert.intermediate_size_per_partition==512
  and expert.topk==8 and expert.num_local_experts==256 and global_experts==256
  and expert_map is None and not router_input and expert._activation_str=='silu'
  and getattr(activation,'name',None)=='SILU' and getattr(parallel,'ep_size',1)==1
  and getattr(parallel,'tp_size',1)==1 and getattr(parallel,'dp_size',1)==1
  and not getattr(parallel,'use_ep',False) and getattr(expert,'_hybrid_min_tokens',0)>=1024
  and expert.a1_gscale is not None and expert.a2_gscale is not None)

def try_apply(expert,out,x,w1,w2,weights,ids,activation,global_experts,expert_map,router_input):
 if not envs.VLLM_SM120_FI_DIRECT_SMALL_M:return False
 if not eligible(expert,x,activation,global_experts,expert_map,router_input):return False
 from vllm.v1.worker.workspace import current_workspace_manager
 from .direct_sanitize import prepare,LiveDirect
 import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch as dispatch
 import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_direct_micro_kernel as kernel
 manager=current_workspace_manager()
 if not hasattr(manager,'get_persistent_resource'):raise RuntimeError('owner-isolated WorkspaceManager required')
 def create():
  if torch.cuda.is_current_stream_capturing():raise RuntimeError('FI direct handles must be warmed before graph capture')
  for module,pinned in [(dispatch,PINNED_DISPATCH),(kernel,PINNED_KERNEL)]:
   if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=pinned:raise RuntimeError('FI direct source pin mismatch')
  from .model_identity import require
  identity=require(expert)
  torch.cuda.synchronize();start=time.perf_counter();before=torch.cuda.memory_allocated()
  # Existing packed operands and calibrated alphas are reused. Multiplier
  # planes expand only when original scale storage is scalar; no weight copy.
  prep=dict(w13=w1,w2=w2,g1=expert._cutlass_g1,g2=expert._cutlass_g2,
   a13=expert._input_global_scale,a2=expert._fc2_input_scale,
   a1_gscale=expert.a1_gscale.expand(256),a2_gscale=expert.a2_gscale.expand(256))
  handle=LiveDirect(prepare(dispatch,expert,prep,x,ids,weights,out,native=True))
  # Runs actual sanitizer/JIT outside capture, initializes persistent epochs.
  handle(x,ids,weights,out);torch.cuda.synchronize()
  print('SM120_FI_DIRECT_SETUP '+json.dumps(dict(shape=int(x.shape[0]),expert_owner=id(expert),
   model=identity['model'],model_type=identity['model_type'],seconds=time.perf_counter()-start,live_allocated_delta=torch.cuda.memory_allocated()-before,
   allocator_peak_observed=torch.cuda.max_memory_allocated(),
   sanitizer_bytes=handle.sanitizer.bytes,dispatch_sha256=PINNED_DISPATCH,kernel_sha256=PINNED_KERNEL)),flush=True)
  return handle
 handle=manager.get_persistent_resource(('sm120_fi_direct_small_m',expert,int(x.shape[0])),create)
 handle(x,ids,weights,out)
 return True

def bind_model_identity(expert):
 # Only invoked by process_weights_after_loading, while Worker.load_model
 # supplies the real set_current_vllm_config context. Never called in forward.
 from vllm.config import get_current_vllm_config
 from .model_identity import bind
 bind(expert,envs.VLLM_SM120_FI_DIRECT_SMALL_M,get_current_vllm_config)
