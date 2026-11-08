"""CPU-only launch-plan preparation for a same-checkpoint target-MoE control.
No --execute option: root reviews eligibility and the experiment stages first.
"""
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'kernels/stack-round2'))
from serve import command,verify
from prepare import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=ROOT/'kernels/stack-round2/bundles/parent_ready2');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 bundle=a.bundle.resolve();m=verify(bundle);assert m['profile']=='parent' and all(v=='0' for v in m['flags'].values())
 out=a.output.resolve();assert not out.exists();out.mkdir(parents=True)
 checkpoint=ROOT/'kernels/deepgemm-sm120/results/gdn_actual_capture_round2_1/checkpoint_config.json'
 cfg=json.loads(checkpoint.read_text());assert cfg['quantization_config']['quant_method']=='compressed-tensors'
 audit=HERE/'results/runtime_source_audit.json';runtime=json.loads(audit.read_text());assert runtime['selection']['nvfp4_weight_a16_scheme_supported'] and not runtime['cuda_initialized']
 launch={}
 for mode,backend in [('native','flashinfer_b12x'),('marlin','marlin')]:
  runout=out/mode
  args=command(SimpleNamespace(name='stack-round2-marlin-'+mode,gpu='0',port=8142,hf_cache='/opt/intramind/hf-cache',inductor_pdl=0),bundle,m,runout,'dg-sm120-env:1')
  args[args.index('--moe-backend')+1]=backend
  args[args.index('--speculative-config')+1]=json.dumps({'method':'mtp','num_speculative_tokens':2,'moe_backend':'flashinfer_b12x'},separators=(',',':'))
  assert '--quantization' not in args
  assert 'unsloth/Qwen3.6-35B-A3B-NVFP4-Fast' in args
  launch[mode]=args
  (out/(mode+'_argv.json')).write_text(json.dumps(args,indent=2)+'\n')
 # With only location/owned-container differences removed, target backend is the sole CLI delta.
 normalized={}
 for mode,args in launch.items():
  normalized[mode]=[x.replace(str(out/mode),'/CONTROL_OUTPUT').replace('stack-round2-marlin-'+mode,'stack-round2-marlin-CONTROL') for x in args]
 diffs=[{'index':i,'native':x,'marlin':y} for i,(x,y) in enumerate(zip(normalized['native'],normalized['marlin'],strict=True)) if x!=y]
 assert len(diffs)==1 and diffs[0]['native']=='flashinfer_b12x' and diffs[0]['marlin']=='marlin',diffs
 manifest={'scope':'Prepared CPU launch plans, not a GPU grant or completed benchmark','checkpoint':'unsloth/Qwen3.6-35B-A3B-NVFP4-Fast','checkpoint_config_sha256':sha(checkpoint),'checkpoint_index_sha256':sha(checkpoint.with_name('checkpoint_model.safetensors.index.json')),'parent_source_id':m['source_id'],'sources':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),HERE/'audit_runtime.py',audit,ROOT/'kernels/stack-round2/serve.py',ROOT/'kernels/stack-round2/prepare.py']},'allowed_cli_delta':diffs,'draft_backend':'flashinfer_b12x','draft_tokens':2,'arithmetic':'Native W4A4 versus Marlin W4A16; same checkpoint weights, not bitwise-equivalent operations','required_before_execution':['full deployment selection gate','actual target/draft class readback','metadata/packed-weight preparation audit','owned server lifecycle harness and GPU grant']}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
