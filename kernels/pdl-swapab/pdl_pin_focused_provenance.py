"""CPU-only source/env/generated-code evidence for the isolated actual-vLLM PDL probe."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OWN=('run_vllm_pdl_pin_focused.sh','pdl_pin_focused_provenance.py','pin_inductor_core.py','pin_inductor_hook.py','pin_bootstrap/sitecustomize.py','verify_pin_events.py','pdl_focused_freeze.py')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)
def sources():return {n:digest(ROOT/'kernels/pdl-swapab'/n) for n in OWN}
def environment(container,mode):
    env=dict(x.split('=',1) for x in container['Config']['Env'])
    expected={'TORCHINDUCTOR_ENABLE_PDL':str(mode),'VLLM_CACHE_ROOT':'/bench-results/pdl_cache/vllm','TORCHINDUCTOR_CACHE_DIR':'/bench-results/pdl_cache/inductor','TRITON_CACHE_DIR':'/bench-results/pdl_cache/triton'}
    require(all(env.get(k)==v for k,v in expected.items()),'PDL mode/cache isolation mismatch')
    require(env.get('QWEN_INDUCTOR_PIN_MODE') in ('record','pin'),'invalid pin mode')
    require(env.get('PYTHONPATH')=='/opt/pdl-pin/pin_bootstrap:/opt/qwen-fi-control','pin bootstrap missing from subprocess path')
    require(env.get('QWEN_INDUCTOR_PIN_OUT')=='/bench-results/pin_control','pin output root mismatch')
    return {**expected,'QWEN_INDUCTOR_PIN_MODE':env['QWEN_INDUCTOR_PIN_MODE']}

def run(action,out,mode):
    path=out/'pdl_source_manifest.json'
    if action=='snapshot':
        require(not path.exists(),'refusing source manifest overwrite')
        pin_mode=(out/'pin_mode.txt').read_text().strip()
        require(pin_mode in ('record','pin'),'invalid requested hook mode')
        state={'mode':mode,'pin_mode':pin_mode,'files':sources()}
        for name in OWN:
            target=out/'pdl_source_files'/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/'kernels/pdl-swapab'/name).read_bytes())
        path.write_text(json.dumps(state,indent=2)+'\n');return state
    state=json.loads(path.read_text())
    require(state['mode']==mode and state['files']==sources(),'live PDL source changed')
    require(all(digest(out/'pdl_source_files'/n)==h for n,h in state['files'].items()),'captured PDL source changed')
    config=json.loads((out/'container.json').read_text())[0]
    env=environment(config,mode)
    require(env['QWEN_INDUCTOR_PIN_MODE']==state['pin_mode']==(out/'pin_mode.txt').read_text().strip(),'requested and actual hook modes differ')
    if env['QWEN_INDUCTOR_PIN_MODE']=='pin':
        values=dict(x.split('=',1) for x in config['Config']['Env'])
        require(values.get('QWEN_INDUCTOR_PIN_REFERENCE')=='/bench-results/pin_reference.json','pin reference path mismatch')
        require(digest(out/'serving/pin_reference.json')==values.get('QWEN_INDUCTOR_PIN_REFERENCE_SHA256'),'pin reference hash mismatch')
    if action=='verify':return {'status':'passed','mode':mode,'sources':state['files'],'environment':env}
    records=[]
    for p in sorted((out/'serving/pdl_cache').rglob('*.py')):
        txt=p.read_text(errors='replace')
        if 'async_compile' not in txt and 'triton' not in txt:continue
        wait='tl.extra.cuda.gdc_wait()' in txt
        signal='tl.extra.cuda.gdc_launch_dependents()' in txt
        launch="'launch_pdl': True" in txt or '"launch_pdl": True' in txt
        calls=[line.strip()[:700] for line in txt.splitlines() if '.run(' in line or 'torch.ops.' in line or 'extern_kernels.' in line]
        records.append({'path':str(p.relative_to(out)),'sha256':digest(p),'wait':wait,'signal':signal,'launch_pdl':launch,'call_sites':calls})
    require(records,'no actual generated code collected')
    participants=sum(r['wait'] and r['launch_pdl'] for r in records)
    require(participants>0 if mode else participants==0,'actual PDL codegen mode not demonstrated')
    return {'status':'passed','mode':mode,'environment':env,'generated_python_files':records,'participating_files':participants,'consumer_coverage':'Call sites recorded; cooperative downstream consumer and runtime overlap require trace inspection. This is not a speedup or parity claim.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('snapshot','verify','coverage'));p.add_argument('output',type=Path);p.add_argument('mode',type=int,choices=(0,1));a=p.parse_args()
    print(json.dumps(run(a.action,a.output,a.mode),indent=2))
