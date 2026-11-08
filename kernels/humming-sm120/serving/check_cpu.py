"""CPU-only actual env getter/compile-factor coverage; no overlay installation."""
import os,ast,importlib.util,sys,json,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
p=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('humming_serving_env_check',p/'generated/vllm/envs.py');env=importlib.util.module_from_spec(spec);sys.modules[spec.name]=env;spec.loader.exec_module(env)
flags=['VLLM_SM120_HUMMING_RN_M1024','VLLM_SM120_HUMMING_RN_PROBE'];rows=[]
for enable,probe in [(0,0),(1,0),(1,1),(0,0)]:
 for flag,value in zip(flags,[enable,probe]):os.environ[flag]=str(value)
 factors=env.compile_factors();values={flag:factors[flag] for flag in flags}
 assert values==dict(zip(flags,[bool(enable),bool(probe)])),values
 rows.append(values)
audit=json.loads((p.parent/'checkpoint_scales_cpu.json').read_text())
assert len(audit['contracts'])==120 and all(c['exact'] for c in audit['contracts'])
assert all(x['gate_up_exact_equal'] for x in audit['results'] if 'gate_up_exact_equal' in x)
assert not torch.cuda.is_initialized()
print(json.dumps(dict(cuda_initialized=False,compile_factor_transitions=rows,checkpoint_contracts=120,all40_gate_up_globals_exact=True,checkpoint_audit_sha256=hashlib.sha256((p.parent/'checkpoint_scales_cpu.json').read_bytes()).hexdigest(),generated_env_sha256=hashlib.sha256((p/'generated/vllm/envs.py').read_bytes()).hexdigest(),extra_weight_and_local_sf_gib_estimate=16.875),indent=2))
