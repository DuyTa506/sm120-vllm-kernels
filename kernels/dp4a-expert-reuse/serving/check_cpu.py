"""CPU actual generated vLLM env getter and compilation-factor transitions."""
import os,sys,importlib.util,json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
p=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fi_direct_env_check',p/'generated/vllm/envs.py')
env=importlib.util.module_from_spec(spec);sys.modules[spec.name]=env;spec.loader.exec_module(env)
flag='VLLM_SM120_FI_DIRECT_SMALL_M';rows=[]
for value in [0,1,0]:
 os.environ[flag]=str(value);actual=env.compile_factors()[flag];assert actual==bool(value);rows.append(actual)
assert not torch.cuda.is_initialized()
print(json.dumps(dict(compile_factor_transitions=rows,cuda_initialized=False)))
