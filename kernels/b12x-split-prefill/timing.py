"""Full-call CUDA-graph event screen, only after in-run numerical references pass."""
import argparse,hashlib,json,os,sys,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--tokens',type=int,choices=[1024,4096],default=1024);p.add_argument('--components',action='store_true');p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 for k in ['CUTE_DSL_CACHE_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','FLASHINFER_WORKSPACE_BASE']:os.environ[k]=str(a.output/k)
 for k,v in dict(VLLM_B12X_FIX_SCALES='1',VLLM_B12X_SHARED_WORKSPACE='1',VLLM_B12X_SCALAR_INPUT_SCALE='1',VLLM_B12X_HYBRID_MIN_TOKENS='1024',VLLM_B12X_ACT_SCALES='static').items():os.environ[k]=v
 import torch
 sys.path.insert(0,str(HERE.parent/'moe-nvfp4'))
 from moe_common import load_layer_raw,prep_cutlass,run_cutlass,snap
 from test_shared_workspace import make_expert,wrapper
 from verify_scalar_dispatch import prepare_exact
 from verify_shared_workspace import load_backend,apply
 from vllm.v1.worker import workspace as ws
 from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
 from split_adapter import Workspace,SplitCall
 from numerical_reference import validate_domain,reference_check,metrics
 assert moe_dispatch._FORCED_BACKEND is None and torch.cuda.get_device_capability()==(12,0)
 torch.backends.cuda.matmul.allow_tf32=False
 pin=json.loads((HERE/'capture_manifest.json').read_text());assert hashlib.sha256(a.capture.read_bytes()).hexdigest()==pin['capture_sha256']
 c=torch.load(a.capture,map_location='cpu',weights_only=True)
 x=c['input'][:a.tokens].cuda().contiguous();ids=c['ids'][:a.tokens].cuda().contiguous();weights=c['weights'][:a.tokens].cuda().contiguous()
 ck=Path(snap());identity={f:hashlib.sha256((ck/f).read_bytes()).hexdigest() for f in ['config.json','model.safetensors.index.json']}
 assert identity=={'config.json':'46cb11d0ce8048b7efa09694c7725ab8be6f0c02d4b2421c1d89316aae280850','model.safetensors.index.json':'d0ab515cf0dcc4fae3daf192b9c6af1d3a9b0e79a4e699892bc18a3013315fd7'}
 report=dict(status='running',shape=dict(layer=19,m=a.tokens,k=2048,i=512,e=256,topk=8),capture_sha256=pin['capture_sha256'],checkpoint_identity=identity,timing_scope='CUDA event average over complete CUDA graph replay; quantization/publisher/clears/FC1/FC2 included; router/topk and weight preparation excluded identically',samples=[])
 def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 save()
 try:
  raw=load_layer_raw(19);prep=prep_cutlass(raw);w=Workspace.allocate(a.tokens,x.device);candidate=SplitCall(w,prep,sm_count=torch.cuda.get_device_properties(x.device).multi_processor_count);candidate.assert_contract();candidate.compile()
  candidate.run(x,ids,weights);torch.cuda.synchronize()
  report['domain_precheck']=validate_domain(w,ids,weights);report['reference_precheck']=reference_check(w,prep,raw,x,ids,weights);save()
  assert report['reference_precheck']['passed'],'no timing after failed arithmetic gate'
  ws.init_workspace_manager(x.device);expert=prepare_exact(load_backend('scalar'),raw,prep,make_expert)
  scalar_out=torch.empty_like(x);cutlass_out=torch.empty_like(x)
  calls={'split':lambda:candidate.run(x,ids,weights),'scalar':lambda:apply(expert,prep,x,ids,weights,scalar_out),'cutlass':lambda:run_cutlass(prep,x,ids,weights,out=cutlass_out)}
  graphs={};stream=torch.cuda.Stream()
  for name,fn in calls.items():
   stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    for _ in range(3):fn()
   torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
   g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):fn()
   graphs[name]=g
   for _ in range(3):g.replay()
   torch.cuda.synchronize()
  scalar_workspace=wrapper(expert)._dynamic_workspace
  report['scalar_workspace_runtime']={'tile_m':int(scalar_workspace.tile_m),'task_count':int(scalar_workspace.task_tail),'physical_tiles':int(scalar_workspace.expert_tile_base[256]),'scope':'Live field read after graph warmup, outside timing'}
  report['pre_timing_diagnostics']={'candidate_scalar':metrics(w.output,scalar_out),'candidate_cutlass':metrics(w.output,cutlass_out)};save()
  # Fixed inputs/routes and all captures/compilation outside the event interval.
  for order in [('scalar','split','cutlass'),('cutlass','split','scalar')]:
   samples={name:[] for name in order}
   for repeat in range(8):
    for name in order:
     graphs[name].replay()
     start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
     start.record()
     for _ in range(50):graphs[name].replay()
     end.record();end.synchronize();samples[name].append(start.elapsed_time(end)/50)
   report['samples'].append(dict(order=list(order),milliseconds=samples,medians_ms={k:statistics.median(v) for k,v in samples.items()}));save()
  if a.components:
   from component_graphs import measure
   report['component_diagnostic']=measure(candidate,x,ids,weights);save()
  # Post-measurement correctness and unchanged input buffers; no observers timed.
  graphs['split'].replay();torch.cuda.synchronize();report['domain_postcheck']=validate_domain(w,ids,weights)
  report['post_reference']=reference_check(w,prep,raw,x,ids,weights)
  assert report['post_reference']['passed']
  assert torch.equal(x.cpu(),c['input'][:a.tokens]) and torch.equal(ids.cpu(),c['ids'][:a.tokens]) and torch.equal(weights.cpu(),c['weights'][:a.tokens])
  report['post_timing_diagnostics']={'candidate_scalar':metrics(w.output,scalar_out),'candidate_cutlass':metrics(w.output,cutlass_out)}
  for values in [w.output,scalar_out,cutlass_out]:assert bool(torch.isfinite(values).all())
  report['status']='passed';report['production_selected']=False;report['scope_limit']='One capture/layer/shape, exploratory arithmetic gates; no vLLM serving or model quality claim';save()
 except BaseException as exc:report['status']='failed';report['exception']=repr(exc);save();raise
if __name__=='__main__':main()
