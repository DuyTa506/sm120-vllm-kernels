"""Provenance rejection and explicit-launch contract checks; CPU only."""
import json, shutil, tempfile
from pathlib import Path
from types import SimpleNamespace
from prepare import HERE, PROFILES
from serve import verify,command
reports=[]
for profile in PROFILES:
    bundle=HERE/'bundles'/f'{profile}_ready2';m=verify(bundle)
    assert m['profile']==profile
    assert list(map(int,m['flags'].values()))==list(PROFILES[profile])
    a=SimpleNamespace(name='stack-round2-check',gpu='0',port=8140,hf_cache='/opt/intramind/hf-cache',inductor_pdl=0)
    args=command(a,bundle,m,Path('/tmp/stack-round2-never-launched'),'sha256:CPU_ONLY')
    values=[args[i+1] for i,v in enumerate(args) if v=='-e'];keys=[v.split('=',1)[0] for v in values]
    assert len(keys)==len(set(keys)), 'Ambiguous duplicate environment assignments'
    assert 'TORCHINDUCTOR_ENABLE_PDL=0' in values
    assert 'VLLM_PLUGINS=stack_round2,qwen_fi_control' in values
    assert not any('dg_integration_probe' in x or 'task_order_probe' in x for x in args)
    reports.append({'profile':profile,'verified_source_id':m['source_id'],'default_flags':m['flags'],'unique_env_count':len(keys)})
base=HERE/'bundles/combined_ready2'
with tempfile.TemporaryDirectory(prefix='stack-round2-negative-') as tmp:
    copied=Path(tmp)/'bundle';shutil.copytree(base,copied)
    p=copied/'vllm/envs.py';p.write_text(p.read_text()+'\n# unexpected mutation\n')
    try:verify(copied)
    except AssertionError as e:assert 'Bundle changed' in str(e)
    else:raise AssertionError('Mutated actual source accepted')
    shutil.copyfile(base/'vllm/envs.py',p)
    p=copied/'manifest.json';m=json.loads(p.read_text());m['profile']='parent';p.write_text(json.dumps(m))
    try:verify(copied)
    except AssertionError as e:assert 'Manifest identity mismatch' in str(e)
    else:raise AssertionError('Mutated profile accepted')
print(json.dumps({'profiles':reports,'negative_checks':['mutated actual source rejected','mutated manifest/profile rejected'],'gpu_used':False},indent=2))
