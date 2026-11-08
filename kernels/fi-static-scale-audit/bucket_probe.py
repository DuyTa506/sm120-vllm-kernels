"""Corrected MMA versus safe direct/CUTLASS, full-call graph bucket screen."""
import argparse,os,sys,json,statistics,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--tokens',type=int,choices=[4,8],required=True);a=p.parse_args();m=a.tokens
here=Path(__file__).resolve().parent
for k,v in dict(VLLM_B12X_FIX_SCALES='1',VLLM_B12X_SHARED_WORKSPACE='1',VLLM_B12X_SCALAR_INPUT_SCALE='1',VLLM_B12X_HYBRID_MIN_TOKENS='1024',VLLM_B12X_ACT_SCALES='static').items():os.environ[k]=v
sys.path[:0]=[str(here/'vendor_direct'),str(here.parent/'moe-nvfp4')]
import torch
from moe_common import load_layer_raw,prep_cutlass,route,reference
from verify_shared_workspace import apply,load_backend,capture
from verify_scalar_dispatch import prepare_exact
from test_shared_workspace import make_expert
from vllm.v1.worker import workspace as vworkspace
import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch as dispatch
from direct_sanitize import Sanitizer,SanitizedCall,prepare as prepare_direct
from mma_plan import MMA
r=dict(status='incomplete',tokens=m,scope='captured real L19 hot-layer complete-call graph; not vLLM latency/quality promotion',checks=[],timings=[],timing_scope='MMA includes sanitizer+micro compaction when M4+native compute; direct includes sanitizer+native compute; CUTLASS current scalar includes input quant+compute. Excludes router and weight/view allocation. Explicit private preplanned buffers.',gate='exploratory FP32 relativeL2<=.25; repeated graph/eager relativeL2<=.01; all-invalid/padded outputs exactlyzero; valid graph state restored before timing')
def err(y,z):
 y=y.float();z=z.float();return dict(finite=bool(torch.isfinite(y).all() and torch.isfinite(z).all()),rel_l2=float((y-z).norm()/z.norm().clamp_min(1e-30)),max_abs=float((y-z).abs().max()),norm=float(y.norm()))
try:
 vworkspace.init_workspace_manager(torch.device('cuda:0'))
 raw=load_layer_raw(19);prep=prep_cutlass(raw);expert=prepare_exact(load_backend('scalar'),raw,prep,make_expert)
 hidden=torch.load(here.parent/'moe-nvfp4/data/moe_inputs.pt',map_location='cpu',weights_only=True)['hidden'][19]
 x=hidden[:m].cuda().contiguous();w,ids=route(x,raw['gate']);outputs=[torch.empty_like(x) for _ in range(3)]
 backend='micro' if m==4 else 'static'
 san=Sanitizer(x,ids,w);mma=MMA(expert,prep,san.safe_x,san.safe_ids,san.safe_weights,outputs[0],backend)
 calls=[SanitizedCall(san,mma),prepare_direct(dispatch,expert,prep,x,ids,w,outputs[1],native=True),lambda:apply(expert,prep,x,ids,w,outputs[2])]
 names=[f'sanitized_{backend}_divisor','sanitized_direct_multiplier','current_scalar_cutlass']
 trace=[];originals={}
 for name in ['_get_micro_kernel','_get_static_kernel','_get_direct_micro_kernel']:
  f=getattr(dispatch,name);originals[name]=f
  def wrapper(*args,_f=f,_name=name,**kw):trace.append(dict(factory=_name,reciprocal=kw.get('input_scales_are_reciprocal')));return _f(*args,**kw)
  setattr(dispatch,name,wrapper)
 try:graphs=[capture(c) for c in calls]
 finally:
  for name,f in originals.items():setattr(dispatch,name,f)
 assert any(t['factory']=='_get_'+backend+'_kernel' and t['reciprocal'] is False for t in trace)
 r['factory_trace']=trace;r['factory_scope']='Python factory selection only; native direct compiled in constructor before trace, no CUDA trace in this screen'
 fixtures=[('real0',0,m),('real64',64,m),('padded_last',0,m-1),('all_invalid',0,0),('restore',0,m)]
 for label,offset,live in fixtures:
  x.copy_(hidden[offset:offset+m]);nw,ni=route(x,raw['gate']);ids.copy_(ni);w.copy_(nw)
  if live<m:ids[live:].fill_(-1);w[live:].fill_(0)
  row=dict(fixture=label,offset=offset,live=live,repeat={});r['checks'].append(row)
  for name,g,call,out in zip(names,graphs,calls,outputs):
   g.replay();first=out.clone();g.replay();second=out.clone();call();torch.cuda.synchronize()
   row['repeat'][name]=dict(graph_eager=err(first,out),graph_repeat=err(first,second))
   if live<m:assert not bool(out[live:].count_nonzero()),name+' padding output nonzero'
   for metric in row['repeat'][name].values():assert metric['finite'] and metric['rel_l2']<=.01,(name,metric)
  assert not bool(calls[0].sanitizer.errors.count_nonzero()) and not bool(calls[1].sanitizer.errors.count_nonzero())
  torch.backends.cuda.matmul.allow_tf32=False;assert not torch.backends.cuda.matmul.allow_tf32
  row['reference_precision']=dict(dtype='float32',allow_tf32=False)
  ref=torch.zeros_like(x,dtype=torch.float32)
  if live:ref[:live]=reference(x[:live],raw,ids[:live],w[:live])
  row['reference_norm']=float(ref.norm());row['fp32']={n:err(o,ref) for n,o in zip(names,outputs)}
  row['vs_cutlass']={n:err(o,outputs[2]) for n,o in zip(names[:2],outputs[:2])}
  a.output.write_text(json.dumps(r,indent=2))
  for name,e in row['fp32'].items():
   assert e['finite'] and (e['norm']==0 if live==0 else e['norm']>0 and e['rel_l2']<=.25),(name,e)
 # All numerical fixtures passed before any timing. Re-establish exact fixture for each case.
 r['numeric_gate_passed']=True
 for label,live in [('real0',m),('padded_last',m-1)]:
  x.copy_(hidden[:m]);nw,ni=route(x,raw['gate']);ids.copy_(ni);w.copy_(nw)
  if live<m:ids[live:].fill_(-1);w[live:].fill_(0)
  for g in graphs:g.replay()
  for order in [list(range(3)),list(reversed(range(3)))]:
   for i in order:
    for _ in range(20):graphs[i].replay()
    times=[]
    for _ in range(10):
     start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
     for _ in range(50):graphs[i].replay()
     end.record();end.synchronize();times.append(start.elapsed_time(end)*1000/50)
    r['timings'].append(dict(fixture=label,live=live,backend=names[i],order=[names[j] for j in order],median_us=statistics.median(times),samples_us=times,replays_per_sample=50))
  for call in calls:call()
  torch.cuda.synchronize()
  assert not bool(calls[0].sanitizer.errors.count_nonzero()) and not bool(calls[1].sanitizer.errors.count_nonzero())
 r['status']='screen_complete_no_promotion'
except BaseException:r['status']='failed';r['failure']=traceback.format_exc();raise
finally:a.output.write_text(json.dumps(r,indent=2))
