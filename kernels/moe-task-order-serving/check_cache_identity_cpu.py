"""Run in a fresh CUDA-hidden process per profile; no native compilation or GPU work."""
import hashlib,json,os
from pathlib import Path
import torch
from vllm import envs
here=Path(__file__).resolve().parent
manifest=json.loads((here/'envs_source_manifest.json').read_text())
assert hashlib.sha256(Path(envs.__file__).read_bytes()).hexdigest()==manifest['base_sha256']
# Apply exactly the appended registration to the same base env module on CPU.
base=(here/'envs_base.py').read_text();generated=(here/'envs_task_order.py').read_text();assert generated.startswith(base)
exec(generated[len(base):],envs.__dict__)
profile=os.environ['TASK_ORDER_PROFILE'];expected=profile=='on'
assert envs.VLLM_SM120_TASK_ORDER_LONG_FIRST is expected
factors=envs.compile_factors();assert factors['VLLM_SM120_TASK_ORDER_LONG_FIRST'] is expected
import task_order_probe
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
kwargs=dict(activation_precision='fp4',quant_mode='nvfp4',E=256,k=2048,n=512,num_topk=8,mac=188,mma_tiler_mn=(128,128),topk_ids_dtype=torch.int32,input_scales_are_reciprocal=False,fast_math=True,activation='silu',swiglu_alpha=1.702,swiglu_beta=1.0,swiglu_limit=None,share_input_across_experts=True)
old=moe_dispatch._dynamic_kernel_cache_key(**kwargs);old_disk=moe_dispatch._disk_kernel_name('dynamic_e256_k2048_n512_t8',old)
task_order_probe.register();task_order_probe.register()
new=moe_dispatch._dynamic_kernel_cache_key(**kwargs);new_disk=moe_dispatch._disk_kernel_name('dynamic_e256_k2048_n512_t8',new)
assert new[:-2]==old and new[-2]=='task_order_source_sha256'
assert new_disk!=old_disk
assert not torch.cuda.is_initialized()
print(json.dumps(dict(profile=profile,flag=expected,compile_factors_sha256=hashlib.sha256(repr(sorted(factors.items())).encode()).hexdigest(),old_disk_key=old_disk,new_disk_key=new_disk,native_source_sha256=new[-1],cuda_initialized=False)))
