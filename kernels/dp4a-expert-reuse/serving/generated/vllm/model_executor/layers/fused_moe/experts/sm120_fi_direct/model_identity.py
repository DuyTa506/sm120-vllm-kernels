"""Immutable per-expert model identity captured only during real load lifecycle."""
from types import MappingProxyType

def validated(config):
 if config is None or getattr(config,'model_config',None) is None:raise RuntimeError('actual vLLM model config missing at weight load')
 model=config.model_config;hf=model.hf_config
 if getattr(hf,'model_type',None)!='qwen3_5_moe' or 'Qwen3_5MoeForConditionalGeneration' not in getattr(hf,'architectures',[]):
  raise RuntimeError('FI direct overlay requires actual Qwen3.6 checkpoint architecture')
 text=getattr(hf,'text_config',None)
 dims=tuple(getattr(text,k,None) for k in ['hidden_size','moe_intermediate_size','num_experts','num_experts_per_tok'])
 if dims!=(2048,512,256,8):raise RuntimeError('Qwen checkpoint dimensions changed')
 return MappingProxyType(dict(model=str(model.model),model_type=hf.model_type,architecture='Qwen3_5MoeForConditionalGeneration',dimensions=dims))

def bind(expert,enabled,getter):
 if not enabled:return
 identity=validated(getter())
 old=getattr(expert,'_sm120_fi_direct_identity',None)
 if old is not None and dict(old)!=dict(identity):raise RuntimeError('expert model identity changed')
 expert._sm120_fi_direct_identity=identity

def require(expert):
 identity=getattr(expert,'_sm120_fi_direct_identity',None)
 if not isinstance(identity,MappingProxyType):raise RuntimeError('expert lacks validated load-time model identity')
 return identity
