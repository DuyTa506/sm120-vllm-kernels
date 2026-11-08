#!/usr/bin/env python3
"""Real-QKVZ tiny-M rowwise swap experiment; CPU preflight unless --run-gpu."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def load_pick(path):
 tree=ast.parse(path.read_text());fn,=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_pick']
 require(not any(isinstance(n,(ast.Call,ast.Attribute,ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal)) for n in ast.walk(fn)),'unexpected skinny dispatch code')
 scope={'__builtins__':{}};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),scope);return scope['_pick']
def main():
 p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--skinny-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--ms',default='1,3,8,16');p.add_argument('--layers',default='0,9,29');p.add_argument('--iterations',type=int,default=25);p.add_argument('--cold-mib',type=int,default=640);p.add_argument('--run-gpu',action='store_true');a=p.parse_args()
 ms=list(map(int,a.ms.split(',')));layers=set(map(int,a.layers.split(',')))
 require(all(1<=m<=16 for m in ms) and a.iterations>=2 and a.cold_mib>=256,'invalid native small-M experiment')
 capture=json.loads((a.inputs/'results.json').read_text());require(capture['status']=='passed','incomplete actual capture')
 selected=[d for d in capture['captures'] if d['layer'] in layers];require({d['layer'] for d in selected}==layers,'missing requested actual layer')
 for d in selected:
  require(digest(a.inputs/d['file'])==d['file_sha256'],'captured payload changed')
  require(d['operands']['a']['shape'][0]>=2*max(ms),'need two disjoint actual activation slices')
 manifest=json.loads((a.skinny_root/'manifest.json').read_text());require(manifest['module_name']=='sm120_skinny_fp8','wrong skinny extension')
 for f,h in manifest['files'].items():require(digest(a.skinny_root/f)==h,'skinny artifact changed')
 pick=load_pick(a.skinny_root/'reference_dispatch.py')
 require(all(pick(m,12288,2048)==(1,4) for m in ms),'actual tiny-QKVZ dispatch changed')
 patch=json.loads((a.source_root.parent/'manifest.json').read_text())
 for f,info in patch['files'].items():require(digest(a.source_root/f)==info['patched_sha256'],'rowwise prototype changed')
 if not a.run_gpu:
  print(json.dumps({'status':'CPU_preflight_passed','layers':sorted(layers),'ms':ms,'skinny_configs':{m:pick(m,12288,2048) for m in ms},'capture_sha256':digest(a.inputs/'results.json'),'patch_sha256':patch['patch_sha256']}));return
 require(not a.output.exists(),'result output exists')
 root=a.source_root.resolve();sys.path.insert(0,str(root));import torch;import deep_gemm
 from vllm import _custom_ops as ops
 require(Path(deep_gemm.__file__).resolve().parent==root/'deep_gemm','wrong DG module')
 binary,=list((root/'deep_gemm').glob('_C*.so'));require(binary.resolve().is_relative_to(root),'extension escapes owned source')
 spec=importlib.util.spec_from_file_location('sm120_skinny_fp8',a.skinny_root/'build/sm120_skinny_fp8.so');skinny=importlib.util.module_from_spec(spec);spec.loader.exec_module(skinny)
 here=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('dg_helpers',here.parent/'deepgemm-sm120/common.py');common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
 torch.backends.cuda.matmul.allow_tf32=False;deep_gemm.set_pdl(False)
 report={'status':'running','scope':'Captured actual QKVZ FP8 bytes/FP32 scales, sliced token rows; controlled operator experiment, not a vLLM serving run.',
  'capture_sha256':digest(a.inputs/'results.json'),'binary_sha256':digest(binary),'patch_manifest':patch,'script_sha256':digest(Path(__file__)),
  'helper_sha256':digest(here.parent/'deepgemm-sm120/common.py'),'skinny_manifest':manifest,'gpu':torch.cuda.get_device_name(),'cases':[]}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 def save():a.output.write_text(json.dumps(report,indent=2)+'\n')
 def rel(y,ref):return float((y.double()-ref.double()).norm()/ref.double().norm().clamp_min(1e-20))
 def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
 try:
  for desc in selected:
   payload=torch.load(a.inputs/desc['file'],weights_only=True,map_location='cpu');ts=payload['tensors']
   for name,t in ts.items():require(tensor_hash(t)==desc['operands'][name]['sha256_logical_bytes'],'captured tensor bytes differ')
   af=ts['a'].cuda();bf=ts['b'].cuda();saf=ts['sa'].cuda();sb=ts['sb'].cuda();original=ts['output'].cuda()
   fresh=ops.cutlass_scaled_mm(af,bf,saf,sb,torch.bfloat16);require(torch.equal(original,fresh),'captured baseline not reproducible')
   n,k=bf.shape[1],bf.shape[0];require((n,k)==(12288,2048),'unexpected actual QKVZ shape')
   weight=bf.T.contiguous();count=max(2,(a.cold_mib*1024**2+n*k-1)//(n*k));weights=[weight]+[weight.clone() for _ in range(count-1)]
   bs=deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(torch.ones(n,k//128,device='cuda',dtype=torch.float32))
   os.environ['DG_ROWWISE_SWAP_TILE']='auto'
   rejected=[]
   for invalid_m in (0,17):
    bad_a=af[:invalid_m].clone();bad_sa=saf[:invalid_m].clone();bad_out=torch.empty((invalid_m,n),dtype=torch.bfloat16,device='cuda')
    # Kernel API must reject the unsupported shape before scale layout conversion.
    try:deep_gemm._C.fp8_rowwise_gemm_nt_sm120_tiny((bad_a,bs),(weight,bs),bad_out,bad_sa,sb,True)
    except RuntimeError:rejected.append(invalid_m)
    else:raise ValueError('unsupported tiny M was silently accepted')
   require(rejected==[0,17],'tiny shape rejection incomplete')
   for m in ms:
    av=af[:m].clone();sa=saf[:m].clone();sa0=sa.clone();sb0=sb.clone();av0=av.clone()
    unit=deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(torch.ones(m,k//128,device='cuda',dtype=torch.float32))
    labels=('cutlass','skinny','off','on','tile64');out={q:torch.empty((m,n),dtype=torch.bfloat16,device='cuda') for q in labels};cfg=pick(m,n,k)
    def call(label,i=0):
     os.environ['DG_ROWWISE_SWAP_TILE']='64x16' if label=='tile64' else 'auto'
     if label=='cutlass':torch.ops._C.cutlass_scaled_mm(out[label],av,weights[i].T,sa,sb,None)
     elif label=='skinny':skinny.mm(out[label],av,weights[i],sa.reshape(-1),sb.reshape(-1),*cfg)
     else:deep_gemm._C.fp8_rowwise_gemm_nt_sm120_tiny((av,unit),(weights[i],bs),out[label],sa,sb,label!='off')
    traces={}
    os.environ['DG_ROWWISE_SWAP_TRACE']='1';os.environ['DG_PRINT_CONFIGS']='1'
    for mode in ('off','on','tile64'):
     _,trace=common.capture_fd1(lambda mode=mode:call(mode));require(f'ROWWISE_SWAP_V1 m={m} n={n} k={k} selected={int(mode!="off")}' in trace,'unproven tiny rowwise policy');traces[mode]=trace
     expected_policy={'off':'off','on':'auto_swap','tile64':'64x16'}[mode]
     require('ROWWISE_POLICY_V1 policy='+expected_policy+' ' in trace,'final selected policy trace missing')
    os.environ.pop('DG_ROWWISE_SWAP_TRACE');os.environ.pop('DG_PRINT_CONFIGS')
    for label in labels:call(label)
    def check():
     ref=(av.float()@weight.float().T)*sa.reshape(-1,1);ref=ref*sb.reshape(1,-1)
     baseline_error=rel(out['cutlass'],ref);rows={}
     for label in labels:
      pair=rel(out[label],out['cutlass']);error=rel(out[label],ref)
      require(bool(torch.isfinite(out[label]).all()) and pair<=0.001,label+' differs from CUTLASS beyond screening margin')
      require(error<=max(1e-6,baseline_error*1.01),label+' exceeds FP32 baseline error margin')
      rows[label]={'relative_l2_vs_cutlass':pair,'relative_l2_vs_fp32':error,'bit_mismatches_vs_cutlass':int((out[label]!=out['cutlass']).sum())}
     return rows
    row={'layer':desc['layer'],'module':desc['module'],'m':m,'n':n,'k':k,'weight_pool_bytes':count*n*k,'skinny_pick':cfg,
     'captured_baseline_exact':True,'rejected_unsupported_m':rejected,'input_hashes':{name:tensor_hash(t) for name,t in {'a':av,'sa':sa,'sb':sb,'weight':weight}.items()},'native_selection_trace':traces,'initial_numerics':check(),'changed_checks':[]}
    report['cases'].append(row);save()
    graphs={}
    for label in labels:
     graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph):call(label)
     graphs[label]=graph
    for change in ('actual_second_slice','zero_token','nonuniform_scales'):
     av.copy_(af[m:2*m] if change=='actual_second_slice' else av0);sa.copy_(saf[m:2*m] if change=='actual_second_slice' else sa0);sb.copy_(sb0)
     if change=='zero_token':sa.reshape(-1)[0]=0
     if change=='nonuniform_scales':
      sa.mul_(torch.logspace(-2,2,m,device='cuda').reshape_as(sa));sb.mul_(torch.logspace(-2,2,n,device='cuda').reshape_as(sb));sb.reshape(-1)[::97]=0
     graph_exact={}
     for label in labels:
      graphs[label].replay();replay=out[label].clone();call(label);require(torch.equal(replay,out[label]),label+' changed-input graph/eager mismatch');graph_exact[label]=True
     row['changed_checks'].append({'change':change,'graph_eager_exact':graph_exact,'numerics':check()});save()
    av.copy_(av0);sa.copy_(sa0);sb.copy_(sb0)
    row['timing_inputs']='original captured slice and exact original scales'
    row['timings']={label:[] for label in labels};row['gpu_state']=common.gpu_state()
    for order in (labels,tuple(reversed(labels))):
     for label in order:
      row['timings'][label].append(common.bench_graph([lambda i=i,label=label:call(label,i) for i in range(count)],iters=a.iterations))
    save();print(json.dumps({'layer':desc['layer'],'m':m,'timings':row['timings']}),flush=True)
    del graphs,graph,out,av,sa,sa0,sb0,av0,unit
   del weights,weight,bs,af,bf,saf,sb,original,fresh
  for f,h in manifest['files'].items():require(digest(a.skinny_root/f)==h,'skinny comparator changed during run')
  report['status']='passed_actual_rowwise_tile';save()
 except BaseException:
  report['status']='failed';report['error']=traceback.format_exc();save();raise
if __name__=='__main__':main()
