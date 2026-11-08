#!/usr/bin/env python3
"""Native same-format forced swap-AB policy ablation; never a vLLM speedup claim."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--run-gpu',action='store_true');p.add_argument('--ms',default='1,3,8,16');p.add_argument('--shapes',default='2048x4096,12288x2048');p.add_argument('--iterations',type=int,default=25);p.add_argument('--cold-mib',type=int,default=640);args=p.parse_args()
 ms=[int(x) for x in args.ms.split(',')];shapes=[tuple(map(int,x.split('x'))) for x in args.shapes.split(',')]
 if not all(1<=m<=16 for m in ms) or not all(n>0 and k>0 and k%128==0 and n%16==0 for n,k in shapes):raise ValueError('unsupported original native swap contract')
 if args.iterations<2 or args.cold_mib<256:raise ValueError('insufficient sampling/cold weight pool')
 plan={'scope':'Native FP8 UE8M0, same quantized bytes/scales; forced orientation policy with PDL disabled. Not rowwise/NVFP4 checkpoint or serving equivalence.', 'ms':ms,'shapes_nk':shapes,'iterations':args.iterations,'cold_mib':args.cold_mib}
 if not args.run_gpu:print(json.dumps({'status':'cpu_plan_only',**plan}));return
 root=args.source_root.resolve();manifest=json.loads((root.parent/'manifest.json').read_text())
 if sha(root/manifest['file'])!=manifest['patched_sha256']:raise ValueError('swap control source hash mismatch')
 sys.path.insert(0,str(root));import torch;import deep_gemm
 if Path(deep_gemm.__file__).resolve().parent!=root/'deep_gemm':raise ValueError('wrong DeepGEMM module loaded')
 binary,=list((root/'deep_gemm').glob('_C*.so'));binary=binary.resolve()
 if not binary.is_relative_to(root):raise ValueError('extension resolves outside isolated source')
 here=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('dg_helpers',here.parent/'deepgemm-sm120/common.py');helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
 deep_gemm.set_pdl(False);torch.manual_seed(20261028);torch.backends.cuda.matmul.allow_tf32=False
 report={'status':'running','plan':plan,'source_manifest':manifest,'binary_sha256':sha(binary),'script_sha256':sha(Path(__file__)),'helper_sha256':sha(here.parent/'deepgemm-sm120/common.py'),'gpu':torch.cuda.get_device_name(),'cases':[]}
 args.output.parent.mkdir(parents=True,exist_ok=True)
 def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
 def error(y,ref):
  diff=y.float()-ref;den=ref.square().mean().sqrt().clamp_min(1e-12)
  result={'relative_rms':float(diff.square().mean().sqrt()/den),'max_error_over_rms':float(diff.abs().max()/den),'finite':bool(torch.isfinite(y).all())}
  if not result['finite'] or result['relative_rms']>0.01 or result['max_error_over_rms']>0.05:raise ValueError('native dequantized-reference error gate failed: '+str(result))
  return result
 try:
  for n,k in shapes:
   count=max(2,math.ceil(args.cold_mib*2**20/(n*k)))
   weight=helpers.fp8_rand((n,k));weights=[weight]+[weight.clone() for _ in range(count-1)]
   sfw=helpers.pow2_sf((n,k//128),-7,-3);packedw=deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sfw)
   deqw=weight.float()*sfw.repeat_interleave(128,dim=1)
   for m in ms:
    aq=helpers.fp8_rand((m,k));original=aq.clone();sfa=helpers.pow2_sf((m,k//128),-7,-3);packeda=deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sfa)
    ref=(aq.float()*sfa.repeat_interleave(128,dim=1))@deqw.T
    outputs={mode:torch.empty((m,n),device='cuda',dtype=torch.bfloat16) for mode in ('off','on')};graphs={};proof={};checks={}
    for mode in ('off','on'):
     os.environ['DG_SM120_SWAP_AB_CONTROL']=mode;os.environ['DG_SM120_SWAP_AB_TRACE']='1';os.environ['DG_PRINT_CONFIGS']='1'
     def call(i=0,mode=mode):deep_gemm.fp8_fp4_gemm_nt((aq,packeda),(weights[i],packedw),outputs[mode],recipe_a=(1,128),recipe_b=(1,128))
     _,trace=helpers.capture_fd1(call)
     expected=f'SWAP_AB_CONTROL_V1 m={m} n={n} k={k} eligible=1 selected={int(mode=="on")}'
     if expected not in trace:raise ValueError('actual native swap selection not proven: '+trace[-1000:])
     proof[mode]=trace;os.environ.pop('DG_SM120_SWAP_AB_TRACE');os.environ.pop('DG_PRINT_CONFIGS')
     checks[mode]=error(outputs[mode],ref)
     stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
     with torch.cuda.stream(stream):
      for i in range(count):call(i)
     torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
     graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph):
      for i in range(count):call(i)
     graphs[mode]=graph
    paired=error(outputs['on'],outputs['off'].float())
    aq.copy_((-original.float()).to(aq.dtype))
    changed={}
    for mode in ('off','on'):
     graphs[mode].replay();torch.cuda.synchronize();changed[mode]=error(outputs[mode],-ref)
    aq.copy_(original)
    for mode in ('off','on'):
     for _ in range(3):graphs[mode].replay()
    torch.cuda.synchronize();samples={mode:[] for mode in graphs}
    for index in range(args.iterations):
     for mode in (('off','on') if index%2==0 else ('on','off')):
      begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
      begin.record();graphs[mode].replay();end.record();end.synchronize();samples[mode].append(begin.elapsed_time(end)*1000/count)
    row={'m':m,'n':n,'k':k,'weight_pool_bytes':count*n*k,'native_selection_trace':proof,'reference_error':checks,'on_vs_off_error':paired,'changed_input_replay':changed,'samples_us':samples,'median_us':{mode:statistics.median(v) for mode,v in samples.items()}}
    report['cases'].append(row);save();print(json.dumps({k:row[k] for k in ('m','n','k','median_us')}),flush=True)
    del graphs,graph,outputs,aq,original,ref
   del weights,weight,deqw,sfw,packedw
  report['status']='complete_native_policy_micro';save()
 except BaseException as exc:
  report['status']='failed';report['error']=repr(exc);save();raise
if __name__=='__main__':main()
