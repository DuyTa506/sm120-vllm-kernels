"""Default-off RN indexed NVFP4 shadow weights; exact M1024 dispatch only."""
import dataclasses,time,hashlib,json,atexit,os
from pathlib import Path
import torch
from vllm import envs
from .sm120_humming_policy import eligible
ENABLED=envs.VLLM_SM120_HUMMING_RN_M1024
PROBE=envs.VLLM_SM120_HUMMING_RN_PROBE
STATES=[]
class State:pass

def prepare(layer,w13,s13,w2,s2):
 if not ENABLED:return
 from .sm120_humming_rounding import install
 from vllm.model_executor.layers.quantization.utils.humming.moe import convert_to_humming_moe_kernel_format,get_humming_moe_quant_config
 from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import HummingIndexedExperts
 from humming.schema.compressed_tensors import CompressedTensorsInputSchema
 from humming import dtypes
 cfg=layer.moe_config
 assert (cfg.num_experts,cfg.experts_per_token,cfg.hidden_dim,cfg.intermediate_size_per_partition)==(256,8,2048,512)
 pc=cfg.moe_parallel_config
 assert (pc.tp_size,pc.dp_size,pc.ep_size)==(1,1,1) and not pc.enable_eplb
 assert str(cfg.activation.value)=='silu' and cfg.in_dtype==torch.bfloat16
 assert not hasattr(layer,'_sm120_humming_state')
 torch.cuda.synchronize(w13.device);started=time.perf_counter();before=torch.cuda.memory_allocated(w13.device)
 identity=install('rn')
 shadow=torch.nn.Module();shadow.moe_config=dataclasses.replace(cfg,moe_backend='humming');shadow.params_dtype=torch.bfloat16;shadow.layer_name=layer.layer_name
 params=dict(w13_weight_packed=w13,w13_weight_scale=s13,w2_weight_packed=w2,w2_weight_scale=s2)
 for n in ['w13_weight_global_scale','w2_weight_global_scale','w13_input_global_scale','w2_input_global_scale']:params[n]=getattr(layer,n)
 for n,v in params.items():shadow.register_parameter(n,torch.nn.Parameter(v.detach().clone().contiguous(),requires_grad=False))
 assert torch.equal(shadow.w13_weight_global_scale[:,0],shadow.w13_weight_global_scale[:,1])
 assert torch.equal(shadow.w13_input_global_scale[:,0],shadow.w13_input_global_scale[:,1])
 expected={n:getattr(shadow,n+'_input_global_scale').float().reciprocal().amax().reshape(1) for n in ('w13','w2')}
 convert_to_humming_moe_kernel_format(shadow,quant_config=dict(quant_method='compressed-tensors',format='nvfp4-pack-quantized',type='float',num_bits=4,strategy='group',group_size=16),input_schema=CompressedTensorsInputSchema(format='nvfp4-pack-quantized',type='float',num_bits=4,dynamic='local',group_size=16),allow_input_schema_fallback=False)
 for n,expected_scale in expected.items():
  assert torch.equal(getattr(shadow,n+'_input_scale_2'),expected_scale)
  schema=shadow.input_schemas[n]
  assert schema.a_dtype==dtypes.float4e2m1 and schema.input_scale_group_size==16 and schema.input_quant_mode.value=='static_tensor_dynamic_group'
  assert shadow.humming_configs[n].as_dtype==dtypes.float8e4m3
 expert=HummingIndexedExperts(moe_config=shadow.moe_config,quant_config=get_humming_moe_quant_config(shadow))
 shapes=expert._workspace_shapes(1024,8,cfg.activation)
 state=State();state.layer=shadow;state.expert=expert;state.work=[torch.empty(s,device=w13.device,dtype=torch.bfloat16) for s in shapes[:2]]
 state.counter=torch.zeros((),device=w13.device,dtype=torch.int64) if PROBE else None
 torch.cuda.synchronize(w13.device)
 storages={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in list(shadow.parameters())+state.work}
 state.record=dict(layer=layer.layer_name,expert_class=type(expert).__module__+'.'+type(expert).__qualname__,setup_seconds=time.perf_counter()-started,allocated_delta_bytes=torch.cuda.memory_allocated(w13.device)-before,owned_parameter_workspace_bytes=sum(storages.values()),workspace_shapes=[list(s) for s in shapes[:2]],rounding=identity,effective_scales={n:float(v.item()) for n,v in expected.items()},activation_schema={n:str(shadow.input_schemas[n]) for n in expected},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),probe=PROBE)
 layer._sm120_humming_state=state;STATES.append(state)

def try_apply(state,output,x,weights,ids,activation,global_experts,expert_map,a1q_scale,apply_router_weight_on_input,expert_tokens_meta):
 if state is None or not eligible(ENABLED,x.shape[0],x.shape[-1],ids.shape[1],global_experts,expert_map is not None,bool(apply_router_weight_on_input)):return False
 assert a1q_scale is None and x.dtype==torch.bfloat16
 # Retain real serving ForwardContext and metadata. No fake context override.
 if state.counter is not None:state.counter.add_(1)
 state.expert.apply(output,x,state.layer.w13_weight,state.layer.w2_weight,weights,ids,activation,global_experts,expert_map,None,None,*state.work,expert_tokens_meta,False)
 return True

def report():
 # Explicit boundary/shutdown inspection only; never called in forward.
 return dict(enabled=ENABLED,probe=PROBE,layers=[dict(s.record,observed_gpu_calls=int(s.counter.item()) if s.counter is not None else None) for s in STATES])
def flush():
 target=os.environ.get('SM120_HUMMING_REPORT')
 if target:
  try:Path(target).write_text(json.dumps(report(),indent=2))
  except Exception as e:Path(target+'.error').write_text(repr(e))
atexit.register(flush)
