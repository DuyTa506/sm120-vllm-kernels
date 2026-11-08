"""Actual L19/M1024 numerical screen. No timers or speed claims."""
import argparse,hashlib,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 for key in ['CUTE_DSL_CACHE_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','FLASHINFER_WORKSPACE_BASE']:os.environ[key]=str(a.output/key)
 for k,v in dict(VLLM_B12X_FIX_SCALES='1',VLLM_B12X_SHARED_WORKSPACE='1',VLLM_B12X_SCALAR_INPUT_SCALE='1',VLLM_B12X_HYBRID_MIN_TOKENS='1024',VLLM_B12X_ACT_SCALES='static').items():os.environ[k]=v
 sys.path.insert(0,str(HERE.parent/'moe-nvfp4'))
 import torch
 from moe_common import load_layer_raw,prep_cutlass,run_cutlass,snap
 from test_shared_workspace import make_expert
 from verify_scalar_dispatch import prepare_exact
 from verify_shared_workspace import load_backend,apply
 from vllm.v1.worker import workspace as ws
 from split_adapter import Workspace,SplitCall
 from numerical_reference import validate_domain,reference_check,metrics
 torch.backends.cuda.matmul.allow_tf32=False
 assert torch.cuda.get_device_capability()==(12,0)
 c=torch.load(a.capture,map_location='cpu',weights_only=True)
 x=c['input'][:1024].cuda().contiguous();ids=c['ids'][:1024].cuda().to(torch.int32).contiguous();weights=c['weights'][:1024].cuda().float().contiguous()
 assert x.shape==(1024,2048) and ids.shape==(1024,8)
 checkpoint=Path(snap());identity={f:hashlib.sha256((checkpoint/f).read_bytes()).hexdigest() for f in ['config.json','model.safetensors.index.json']}
 assert identity=={'config.json':'46cb11d0ce8048b7efa09694c7725ab8be6f0c02d4b2421c1d89316aae280850','model.safetensors.index.json':'d0ab515cf0dcc4fae3daf192b9c6af1d3a9b0e79a4e699892bc18a3013315fd7'}
 from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
 assert moe_dispatch._FORCED_BACKEND is None
 report=dict(status='running',scope='L19/M1024 actual capture prefix; exploratory numerical screen only, no timings',capture=str(a.capture),capture_sha256=hashlib.sha256(a.capture.read_bytes()).hexdigest(),checkpoint=str(checkpoint),checkpoint_identity=identity,cases=[],gates={'phase1_quantized_intermediate_rel_l2_max':.01,'phase2_atomic_vs_actual_intermediate_fp64_rel_l2_max':.02,'graph_vs_fresh_eager_rel_l2_max':.02,'scope':'Exploratory screening thresholds, NOT production quality acceptance; exact mismatch counts and errors retained'})
 def save(): (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 save()
 try:
  raw=load_layer_raw(19);prep=prep_cutlass(raw);w=Workspace.allocate(1024,x.device)
  call=SplitCall(w,prep,sm_count=torch.cuda.get_device_properties(x.device).multi_processor_count);call.assert_contract();call.compile()
  # Warm all publisher/quant kernels before graph construction.
  call.run(x,ids,weights);torch.cuda.synchronize()
  domain=validate_domain(w,ids,weights)
  torch.save({k:getattr(w,k).cpu() for k in ['q','sf','intermediate','token_map','route_slots','token_weights','counts','tile_bases','task_expert','task_rows','output']},a.output/'initial_domain.pt')
  ref=reference_check(w,prep,raw,x,ids,weights)
  report['cases'].append(dict(name='actual_eager',domain=domain,reference=ref));save()
  actual=w.output.clone()
  ws.init_workspace_manager(x.device)
  expert=prepare_exact(load_backend('scalar'),raw,prep,make_expert)
  baseline=torch.empty_like(x);apply(expert,prep,x,ids,weights,baseline);torch.cuda.synchronize();base1=baseline.clone()
  apply(expert,prep,x,ids,weights,baseline);torch.cuda.synchronize()
  cut=run_cutlass(prep,x,ids,weights);torch.cuda.synchronize()
  report['diagnostic_comparators']={'scalar_repeat':metrics(base1,baseline),'candidate_vs_scalar':metrics(actual,base1),'candidate_vs_cutlass':metrics(actual,cut),'scope':'Current complete scalar call and CUTLASS are diagnostic comparisons, not sole arithmetic oracle'};save()
  torch.save(dict(candidate=actual.cpu(),scalar=base1.cpu(),scalar_repeat=baseline.cpu(),cutlass=cut.cpu(),ids=ids.cpu(),weights=weights.cpu()),a.output/'actual_outputs.pt')
  assert ref['passed'],'initial reference gate failed; no graph followup'
  # Side-stream warmup then capture stable pointers; all mutations below are outside graph.
  side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(side):call.run(x,ids,weights)
  torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):call.run(x,ids,weights)
  original_x=x.clone();original_ids=ids.clone();original_weights=weights.clone()
  for name in ['changed_input_routes','tail_sentinels','all_empty','restore_original']:
   if name=='changed_input_routes':x.copy_(original_x*.37);ids.copy_(original_ids.roll(17,0));weights.copy_(original_weights.roll(17,0))
   elif name=='tail_sentinels':ids[::3,1]=-1
   elif name=='all_empty':ids.fill_(-1)
   else:x.copy_(original_x);ids.copy_(original_ids);weights.copy_(original_weights)
   graph.replay();torch.cuda.synchronize();graph_out=w.output.clone()
   d=validate_domain(w,ids,weights)
   r=reference_check(w,prep,raw,x,ids,weights)
   call.run(x,ids,weights);torch.cuda.synchronize();fresh_domain=validate_domain(w,ids,weights)
   err=metrics(graph_out,w.output)
   row=dict(name=name,graph_domain=d,fresh_domain=fresh_domain,reference=r,graph_vs_fresh_eager=err)
   report['cases'].append(row);save()
   assert r['passed'] and err['finite'] and err['rel_l2']<=.02
   if name=='all_empty':assert not bool(graph_out.any()) and int(w.tile_bases[-1])==0
  report['status']='passed';report['timing_claim']=False;save()
 except BaseException as exc:
  report['status']='failed';report['exception']=repr(exc);save();raise
if __name__=='__main__':main()
