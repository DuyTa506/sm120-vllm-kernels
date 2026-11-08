"""Exercise installed spec classes through the exact startup serialization path."""
import hashlib,json,inspect
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec,MambaSpec
from cache_geometry import describe,encode_spec_field,validate_preempt_geometry
full=FullAttentionSpec(block_size=2176,num_kv_heads=2,head_size=256,dtype=torch.float8_e4m3fn)
mamba=MambaSpec(block_size=2176,shapes=((16,128,128),(4096,4)),dtypes=(torch.float32,torch.bfloat16),mamba_cache_mode='align',num_speculative_blocks=2,num_prefill_checkpoint_blocks=1,prefill_checkpoint_alignment=64)
runner=NS(kv_cache_config=NS(num_blocks=50,kv_cache_groups=[NS(layer_names=['fixture.full'],kv_cache_spec=full)]+[NS(layer_names=[f'fixture.gdn.{i}'],kv_cache_spec=mamba) for i in range(3)]),vllm_config=NS(model_config=NS(max_model_len=16384),parallel_config=NS(decode_context_parallel_size=1),cache_config=NS(mamba_cache_mode='align')))
geometry=describe(runner)
serialized=json.dumps(geometry,indent=2,default=encode_spec_field)
restored=json.loads(serialized)
validate_preempt_geometry(restored)
assert restored['groups'][0]['spec_fields']['dtype']=={'torch_dtype':'torch.float8_e4m3fn'}
assert restored['groups'][1]['spec_fields']['dtypes'][0]=={'torch_dtype':'torch.float32'}
assert restored['groups'][1]['spec_fields']['mamba_type']['name']=='MAMBA2'
try:json.dumps({'unsupported':object()},default=encode_spec_field)
except TypeError:pass
else:raise AssertionError('arbitrary object silently serialized')
source=Path(inspect.getsourcefile(FullAttentionSpec))
print(json.dumps(dict(cuda_initialized=torch.cuda.is_initialized(),installed_spec_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),description='actual installed spec instances with representative CPU-only fixture geometry; not observed runtime allocation',geometry=restored),indent=2))
