"""Print a reproducible Docker argv; --execute explicitly starts one owned server."""
import argparse, fcntl, hashlib, json, os, signal, subprocess, time
from pathlib import Path
from prepare import ROOT, FLAGS, sha
PKG='/usr/local/lib/python3.12/dist-packages/vllm'
LOCK='/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock'
def verify(bundle):
    manifest=json.loads((bundle/'manifest.json').read_text());identity=manifest.pop('source_id')
    assert hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()==identity,'Manifest identity mismatch'
    manifest['source_id']=identity
    for rel,digest in manifest['outputs'].items():assert sha(bundle/rel)==digest,f'Bundle changed: {rel}'
    for rel,digest in manifest['inputs'].items():assert sha(ROOT/rel)==digest,f'Preparation input changed: {rel}'
    dgroot=ROOT/manifest['dg_source_root']
    actual={str(p.relative_to(dgroot)) for p in dgroot.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pyc') and '.git' not in p.parts}
    assert actual==set(manifest['dg_files']), 'DG file set changed'
    for rel,digest in manifest['dg_files'].items():assert sha(ROOT/manifest['dg_source_root']/rel)==digest,f'DG source changed: {rel}'
    return manifest

def command(a,bundle,m,out,image):
    key=m['source_id']+'-pdl'+str(a.inductor_pdl)
    args=['docker','run','--name',a.name,'--gpus','device='+a.gpu,'--ipc=host','-p',f'127.0.0.1:{a.port}:8000']
    def env(k,v):args.extend(['-e',f'{k}={v}'])
    def mount(src,dst,readonly=True):args.extend(['-v',f'{src}:{dst}'+(':ro' if readonly else '')])
    mount(a.hf_cache,'/root/.cache/huggingface');mount(bundle,'/stack')
    mount(ROOT/m['dg_source_root'],'/work/src');mount(out,'/out',False)
    for rel in m['outputs']:
        if rel.startswith('vllm/'):mount(bundle/rel,PKG+'/'+rel[5:])
    for k,v in m['flags'].items():env(k,v)
    settings={'STACK_PROFILE':m['profile'],'HF_HUB_OFFLINE':'1','PYTHONPATH':'/work/src:/stack/plugin:/stack/fi','VLLM_PLUGINS':'stack_round2,qwen_fi_control','VLLM_SM120_STACK_SOURCE_ID':m['source_id'],
      'TORCHINDUCTOR_ENABLE_PDL':str(a.inductor_pdl),'VLLM_SM120_SKINNY_FP8_MAX_M':'64','VLLM_CACHE_ROOT':'/out/cache/'+key+'/vllm','TORCHINDUCTOR_CACHE_DIR':'/out/cache/'+key+'/inductor','TRITON_CACHE_DIR':'/out/cache/'+key+'/triton','DG_JIT_CACHE_DIR':'/out/cache/'+key+'/deepgemm','TORCH_EXTENSIONS_DIR':'/out/cache/'+key+'/extensions','FLASHINFER_WORKSPACE_BASE':'/out/cache/'+key+'/flashinfer',
      'CUDA_HOME':'/usr/local/cuda-13.0','CPLUS_INCLUDE_PATH':'/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include','LIBRARY_PATH':'/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib',
      'VLLM_SM120_FP8_PREFILL':'1','VLLM_SM120_FP8_PREFILL_NOGLUE':'1','VLLM_FI_KERNEL_PAGES':'16','VLLM_GDN_INTERNAL_CKPT':'1','VLLM_GDN_FUSED_CONV':'1','VLLM_B12X_FIX_SCALES':'1','VLLM_B12X_SHARED_WORKSPACE':'1','VLLM_B12X_SCALAR_INPUT_SCALE':'1',
      'QWEN_FI_IMPORT_JSON':'/stack/fi/references/next_cpu_audit_autotune.json','QWEN_FI_IMPORT_SHA256':sha(bundle/'fi/references/next_cpu_audit_autotune.json'),'QWEN_FI_IMPORT_OUT':'/out/fi_control','VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR':'/out/fi_control/cache/autotune_import'}
    for k,v in settings.items():env(k,v)
    args+=['--entrypoint','vllm',image,'serve','unsloth/Qwen3.6-35B-A3B-NVFP4-Fast','--served-model-name','bench','--language-model-only','--max-model-len','131072','--max-num-seqs','64','--gpu-memory-utilization','0.78','--enable-chunked-prefill','--enable-prefix-caching','--speculative-config','{"method":"mtp","num_speculative_tokens":2}','--async-scheduling','--max-num-batched-tokens','16384','--block-size','2176','--moe-backend','flashinfer_b12x','--shutdown-timeout','30','--cudagraph-capture-sizes']
    args+=(bundle/'cudagraph_sizes.txt').read_text().split()
    return args

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--port',type=int,default=8140);p.add_argument('--gpu',default='0');p.add_argument('--hf-cache',default='/opt/intramind/hf-cache');p.add_argument('--inductor-pdl',type=int,choices=[0,1],default=0);p.add_argument('--execute',action='store_true');a=p.parse_args()
    assert a.name.startswith('stack-round2-') and all(c.isalnum() or c in '_-' for c in a.name)
    bundle=a.bundle.resolve();out=a.output.resolve();m=verify(bundle)
    image='dg-sm120-env:1';args=command(a,bundle,m,out,image)
    if not a.execute:
        print(json.dumps({'profile':m['profile'],'source_id':m['source_id'],'dry_run':True,'argv':args},indent=2));return
    assert not out.exists(),'Use a fresh output/cache directory'
    with open(LOCK,'w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def vacancy():return subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        assert not vacancy(),'GPU is occupied'
        assert subprocess.run(['docker','container','inspect',a.name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0,'Container already exists'
        image=subprocess.check_output(['docker','image','inspect','dg-sm120-env:1','--format','{{.Id}}'],text=True).strip()
        args=command(a,bundle,m,out,image);out.mkdir(parents=True)
        (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');(out/'launch.json').write_text(json.dumps({'argv':args,'image':image},indent=2)+'\n')
        rc=1;proc=None
        def interrupted(signum,frame):raise KeyboardInterrupt(signum)
        signal.signal(signal.SIGTERM,interrupted)
        try:
            with (out/'server.log').open('w') as log:
                proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
                print(f'Owned container {a.name}; docker PID {proc.pid}; log {out}/server.log',flush=True)
                rc=proc.wait()
        except KeyboardInterrupt as e:rc=143 if e.args and e.args[0]==signal.SIGTERM else 130
        finally:
            subprocess.run(['docker','stop','-t','45',a.name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            with (out/'docker.log').open('w') as log:subprocess.run(['docker','logs',a.name],stdout=log,stderr=subprocess.STDOUT)
            subprocess.run(['docker','rm','-f',a.name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            if proc is not None:proc.wait(timeout=60)
            remaining=subprocess.check_output(['docker','ps','-a','--filter',f'name=^/{a.name}$','--format','{{.ID}}'],text=True)
            (out/'container_after.txt').write_text(remaining)
            gpu=vacancy();(out/'gpu_after.txt').write_text(gpu)
            if remaining.strip() or gpu:rc=8
            with (out/'fi_shutdown_verify.log').open('w') as log:
                check=subprocess.run(['python3',str(bundle/'fi/qwen_fi_control_plugin.py'),str(out/'fi_control'),'--source-sha256',sha(bundle/'fi/references/next_cpu_audit_autotune.json')],stdout=log,stderr=subprocess.STDOUT)
            if check.returncode and rc==0:rc=5
            try:verify(bundle)
            except Exception as e:(out/'source_error.txt').write_text(repr(e));rc=4
            (out/'exit_code.txt').write_text(str(rc)+'\n')
        raise SystemExit(rc)
if __name__=='__main__':main()
