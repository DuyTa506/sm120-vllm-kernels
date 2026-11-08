"""One fresh-process mode; compare saved OFF/ON outputs before interpreting timing."""
import argparse,hashlib,json,os,statistics,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['off','on'],required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 for k in ['CUTE_DSL_CACHE_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','FLASHINFER_WORKSPACE_BASE']:os.environ[k]=str(a.output/k)
 for k,v in dict(VLLM_B12X_FIX_SCALES='1',VLLM_B12X_SHARED_WORKSPACE='1',VLLM_B12X_SCALAR_INPUT_SCALE='1',VLLM_B12X_HYBRID_MIN_TOKENS='1024',VLLM_B12X_ACT_SCALES='static').items():os.environ[k]=v
 sys.path.insert(0,str(HERE.parent/'moe-nvfp4'))
 import torch
 from plugin import install
 registered=install(a.mode)
 from moe_common import load_layer_raw,prep_cutlass,run_cutlass,snap
 from test_shared_workspace import make_expert
 from verify_scalar_dispatch import prepare_exact
 from verify_shared_workspace import load_backend,apply,errors,capture
 from vllm.v1.worker import workspace as ws
 assert torch.cuda.get_device_capability()==(12,0);torch.backends.cuda.matmul.allow_tf32=False
 c=torch.load(a.capture,map_location='cpu',weights_only=True);assert c['input'].shape==(4096,2048)
 checkpoint=Path(snap());identity={f:hashlib.sha256((checkpoint/f).read_bytes()).hexdigest() for f in ['config.json','model.safetensors.index.json']}
 assert identity=={'config.json':'46cb11d0ce8048b7efa09694c7725ab8be6f0c02d4b2421c1d89316aae280850','model.safetensors.index.json':'d0ab515cf0dcc4fae3daf192b9c6af1d3a9b0e79a4e699892bc18a3013315fd7'}
 report=dict(status='running',mode=a.mode,registration=registered,capture_sha256=hashlib.sha256(a.capture.read_bytes()).hexdigest(),checkpoint_identity=identity,cases=[],gate_rel_l2=.02,quality_equivalence=False)
 def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
 save()
 try:
  raw=load_layer_raw(19);prep=prep_cutlass(raw);ws.init_workspace_manager(torch.device('cuda:0'));e=prepare_exact(load_backend('scalar'),raw,prep,make_expert)
  scratch=e._ensure_wrapper()._dynamic_workspace;assert scratch.tile_m==128
  outputs={}
  for m in [1024,4096]:
   x=c['input'][:m].cuda().contiguous();ids=c['ids'][:m].cuda().int().contiguous();weights=c['weights'][:m].cuda().float().contiguous();out=torch.empty_like(x)
   original=[v.clone() for v in [x,ids,weights]]
   def call():
    assert e._ensure_wrapper()._dynamic_workspace is scratch and scratch.tile_m==128
    apply(e,prep,x,ids,weights,out)
   graph=capture(call);case=dict(tokens=m,tile_m=128,checks=[],timing_us=[]);report['cases'].append(case)
   ws.lock_workspace()
   for name in ['actual','changed_input_routes','sparse_tail','restored']:
    if name=='changed_input_routes':x.copy_(original[0]*.37);ids.copy_(original[1].roll(17,0));weights.copy_(original[2].roll(17,0)*.75)
    elif name=='sparse_tail':
     # All expert tails are 32 rows for M1024; M4096 gets64 per expert
     # plus128 for half the experts: boundaries exercise inactive/full warps.
     ids.copy_((torch.arange(m*8,device=x.device).reshape(m,8)//8+torch.arange(8,device=x.device)*31)%256)
     if m==4096:ids[2048:]=ids[2048:]%128
    elif name=='restored':
     for v,old in zip([x,ids,weights],original):v.copy_(old)
    graph.replay();torch.cuda.synchronize();g=out.clone();call();torch.cuda.synchronize();fresh=out.clone();cut=run_cutlass(prep,x,ids,weights);torch.cuda.synchronize()
    counts=torch.bincount(ids.flatten().long(),minlength=256).cpu().tolist();tails=[min(128,n-i) for n in counts for i in range(0,n,128)]
    row=dict(name=name,input_hashes={k:tensor_hash(v) for k,v in zip(['x','ids','weights'],[x,ids,weights])},graph_fresh=errors(g,fresh),vs_cutlass=errors(g,cut),valid_rows_hist={str(v):tails.count(v) for v in sorted(set(tails))})
    case['checks'].append(row);outputs[f'{m}_{name}']=g.cpu();save();assert row['graph_fresh']['finite'] and row['graph_fresh']['rel_l2']<=.02
   for block in range(3):
    samples=[]
    for _ in range(30):
     start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();graph.replay();end.record();end.synchronize();samples.append(start.elapsed_time(end)*1000)
    case['timing_us'].append(dict(median=statistics.median(samples),samples=samples))
   # Trace is after timed blocks, so profiler overhead is excluded from timings.
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    graph.replay();torch.cuda.synchronize()
   trace=a.output/f'trace_m{m}.json';prof.export_chrome_trace(str(trace));events=json.loads(trace.read_text())['traceEvents'];names=[v['name'] for v in events if v.get('cat')=='kernel'];case['trace_kernel_names']=names
   expected='MoEGatedDynamicKernelDefinedFC2' if a.mode=='on' else 'MoEGatedDynamicKernel'
   assert any(expected in n for n in names),f'no actual native class in graph trace: {names}'
   if a.mode=='off':assert not any('DefinedFC2' in n for n in names)
   save()
  torch.save(outputs,a.output/'outputs.pt');report['status']='passed';save()
 except BaseException as exc:report['status']='failed';report['exception']=repr(exc);save();raise
if __name__=='__main__':main()
