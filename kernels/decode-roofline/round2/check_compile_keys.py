"""Execute in pinned container; imports only composed envs, never initializes CUDA."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

path=Path(__file__).parent/'generated/vllm/envs.py'
results=[]
for name,bf16,fp8,max_m in [('off',0,0,64),('bf16',1,0,64),('fp8',0,1,64),('both',1,1,64),('fp8_m32',0,1,32)]:
 for key,value in dict(VLLM_SKINNY_BF16_GEMM=bf16,VLLM_SM120_SKINNY_FP8=fp8,VLLM_SM120_SKINNY_FP8_MAX_M=max_m).items():os.environ[key]=str(value)
 spec=importlib.util.spec_from_file_location('skinny_envs',path)
 module=importlib.util.module_from_spec(spec);sys.modules['skinny_envs']=module;spec.loader.exec_module(module)
 factors=module.compile_factors()
 assert factors['VLLM_SKINNY_BF16_GEMM']==bool(bf16)
 assert factors['VLLM_SM120_SKINNY_FP8']==bool(fp8)
 assert factors['VLLM_SM120_SKINNY_FP8_MAX_M']==max_m
 results.append(dict(profile=name,sha256=hashlib.sha256(repr(sorted(factors.items())).encode()).hexdigest()))
assert len({r['sha256'] for r in results})==len(results)
import torch
assert not torch.cuda.is_initialized()
print(json.dumps(dict(profiles=results,cuda_initialized=False),indent=2))
