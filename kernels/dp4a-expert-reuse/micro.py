"""Hot-layer full-call timing screen; setup outside graphs, no serving claim."""
import argparse,os,sys,json,hashlib,importlib.util,statistics,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tokens',type=int,choices=[1,3,4,8],required=True);p.add_argument('--direct-checks',action='store_true');p.add_argument('--live-checks',action='store_true');p.add_argument('--skip-static-auto',action='store_true');p.add_argument('--measurement',choices=['single','batched'],default='single');a=p.parse_args();root=Path(__file__).resolve().parent
result=dict(status='incomplete',scope='hot-layer repeated-input full-call CUDA graph screen; no serving/quality promotion; original input/routing copies and weight/scale/view preparation excluded',tokens=a.tokens,measurement=a.measurement,replays_per_sample=50 if a.measurement=='batched' else 1,samples_per_backend=8 if a.measurement=='batched' else 100,checks=[],timings=[])
def failure(kind,value,tb):
 result['status']='failed';result['failure']=''.join(traceback.format_exception(kind,value,tb));a.output.write_text(json.dumps(result,indent=2));sys.__excepthook__(kind,value,tb)
sys.excepthook=failure
for k,v in dict(VLLM_B12X_FIX_SCALES='1',VLLM_B12X_SHARED_WORKSPACE='1',VLLM_B12X_SCALAR_INPUT_SCALE='1',VLLM_B12X_HYBRID_MIN_TOKENS='1024',VLLM_B12X_ACT_SCALES='static').items():os.environ[k]=v
sys.path.insert(0,str(root.parent/'moe-nvfp4'))
import torch
from prepare import prepare_raw,workspace,call
from fi_plan import Plan
from direct_sanitize import prepare as prepare_safe_direct, LiveDirect
from fi_comparator import run as numeric_fi
from moe_common import load_layer_raw,prep_cutlass,route,reference
from verify_shared_workspace import apply,load_backend,capture
from verify_scalar_dispatch import prepare_exact
from test_shared_workspace import make_expert
from vllm.v1.worker import workspace as vworkspace
import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch as dispatch
import flashinfer.jit.env as fi_env
from flashinfer.cute_dsl import fp4_common
from route_contract import validate
from reference_gate import check as check_reference
assert a.capture.stat().st_size>1000000
result['cache_paths']=dict(flashinfer=str(fi_env.FLASHINFER_WORKSPACE_DIR),cute=os.environ.get('CUTE_DSL_CACHE_DIR'))
assert str(fi_env.FLASHINFER_WORKSPACE_DIR).startswith(str(a.output.parent/'cache/flashinfer_base'))
spec=importlib.util.spec_from_file_location('dp4a_expert_reuse_sm120',root/'build/dp4a_expert_reuse_sm120.so');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
vworkspace.init_workspace_manager(torch.device('cuda:0'))
raw=load_layer_raw(19);prep=prep_cutlass(raw);qprep=prepare_raw(raw);expert=prepare_exact(load_backend('scalar'),raw,prep,make_expert)
hidden=torch.load(a.capture,map_location='cpu',weights_only=True)['hidden'][19]
x=hidden[:a.tokens].cuda().contiguous();weights,ids=route(x,raw['gate']);outputs=[torch.empty_like(x) for _ in range(7)];work=[workspace(x.device) for _ in range(2)]
plans=[Plan(dispatch,expert,prep,x,ids,weights,outputs[3+j],forced,direct_multiplier=multiplier) for j,(forced,multiplier) in enumerate([(None,False),('direct_micro',False),('direct_micro',True)])]
calls=[lambda:call(mod,qprep,x,ids,weights,work[0],outputs[0],False),lambda:call(mod,qprep,x,ids,weights,work[1],outputs[1],True),lambda:apply(expert,prep,x,ids,weights,outputs[2]),*plans]
live_handle=LiveDirect(prepare_safe_direct(dispatch,expert,prep,x,ids,weights,outputs[6],native=True))
calls.append(lambda:live_handle(x,ids,weights,outputs[6]))
names=['dp4a_reuse_off','dp4a_reuse_on','current_scalar_cutlass','fi_auto_calibrated','fi_direct_calibrated','fi_direct_preplanned_multiplier','fi_direct_native_sanitized']
def err(y,r):
 return dict(finite=bool(torch.isfinite(y).all() and torch.isfinite(r).all()),rel_l2=float((y.float()-r.float()).norm()/r.float().norm().clamp_min(1e-30)),max_abs=float((y.float()-r.float()).abs().max()))
source_paths=list(Path(dispatch.__file__).parent.glob('*.py'))+[Path(fp4_common.__file__)]+[root.parent/'moe-nvfp4'/n for n in ['moe_common.py','verify_shared_workspace.py','verify_scalar_dispatch.py','test_shared_workspace.py','scalar/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py']]
hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_paths};(a.output.parent/'runtime_sources.json').write_text(json.dumps(hashes,indent=2))
trace=[];originals={}
for key in ['_get_direct_micro_kernel','_get_micro_kernel','_get_static_kernel','_get_dynamic_kernel']:
 fn=getattr(dispatch,key);originals[key]=fn
 def wrap(*args,_name=key,_fn=fn,**kwargs):trace.append(_name);return _fn(*args,**kwargs)
 setattr(dispatch,key,wrap)
result['fi_direct_preplanned_multiplier_contract']='forced direct only; input quant multipliers prepared outside graph, reciprocal flag false, cached weightviews retain same g1/g2 (no second fold); exact output parity with diagnosed reciprocal=True required'
result['factory_trace_scope']='Python constructor/factory evidence, not actual GPU kernel trace';result['capture_sha256']=hashlib.sha256(a.capture.read_bytes()).hexdigest()
try:
 if a.live_checks:
  from live_pointer_checks import run as live_checks
  result['live_pointer_checks']=live_checks(dispatch,expert,prep,raw,hidden,a.output.parent)
 if a.direct_checks:
  from direct_graph_checks import run as direct_checks
  result['direct_checks']=direct_checks(dispatch,expert,prep,raw,hidden,a.output.parent)
 active=list(range(len(names)))
 if a.skip_static_auto:
  assert a.tokens in (4,8),'static exclusion only at known static shapes'
  active.remove(3);result['excluded_comparators']={names[3]:'legacy reciprocal=True FI-auto static ABI invalid; corrected static requires separately reviewed comparator'}
 graphs=[None]*len(names);result['factory_selection']={}
 for idx in active:
  name,c=names[idx],calls[idx]
  trace.clear();graphs[idx]=capture(c);result['factory_selection'][name]=list(trace)
 for fixture,offset in [('real_0',0),('real_64',64)]+([('synthetic_eight_hot_permuted',0)] if a.tokens>1 else []):
  x.copy_(hidden[offset:offset+a.tokens]);nw,ni=route(x,raw['gate'])
  if fixture.startswith('synthetic'):ni=torch.stack([torch.arange(8,device=x.device,dtype=torch.int32).roll(t) for t in range(a.tokens)])
  cpu_ids=ni.cpu();validate(cpu_ids.tolist());weights.copy_(nw);ids.copy_(ni)
  counts=torch.bincount(cpu_ids.flatten().long(),minlength=256);hot=counts[counts>0].tolist()
  row=dict(fixture=fixture,unique_experts=len(hot),expert_token_multiplicities=hot,own_replay={},prepared_fi_parity={});result['checks'].append(row)
  for idx in active:
   name,g,c,out=names[idx],graphs[idx],calls[idx],outputs[idx]
   g.replay();saved=out.clone();g.replay();repeat=out.clone();c()
   metrics=dict(graph_eager=err(saved,out),graph_repeat=err(saved,repeat));row['own_replay'][name]=metrics
   assert all(v['finite'] for v in metrics.values())
   if name.startswith('dp4a'):assert torch.equal(saved,out) and torch.equal(saved,repeat)
  assert torch.equal(outputs[0],outputs[1]),'reuse changed output'
  assert torch.equal(outputs[6],outputs[5]),'sanitized native direct differs from prepared direct'
  assert not bool(live_handle.sanitizer.errors.count_nonzero()),'malformed expert IDs'
  # Validate cached weight views/alphas against the previously diagnosed eager adapter.
  for j,forced in enumerate([None,'direct_micro','direct_micro']):
   if 3+j not in active:continue
   other=torch.empty_like(x);old=dispatch._FORCED_BACKEND;dispatch._FORCED_BACKEND=forced
   try:numeric_fi(dispatch,expert,raw,prep,x,ids,weights,other)
   finally:dispatch._FORCED_BACKEND=old
   row['prepared_fi_parity'][names[3+j]]=err(outputs[3+j],other)
   assert torch.equal(outputs[3+j],other),'preplanned FI differs from numerical adapter'
  torch.backends.cuda.matmul.allow_tf32=False;assert not torch.backends.cuda.matmul.allow_tf32
  yref=reference(x,raw,ids,weights);row['reference_precision']=dict(dtype='float32',allow_tf32=False);row['fp32_errors']={names[i]:err(outputs[i],yref) for i in active}
  row['reference_norm']=float(yref.float().norm());row['output_norms']={names[i]:float(outputs[i].float().norm()) for i in active}
  row['reference_sanity']=check_reference(row['fp32_errors'],row['output_norms'],row['reference_norm'])
  for order in [active,list(reversed(active))]:
   for idx in order:
    for _ in range(20):graphs[idx].replay()
    times=[];batch_totals=[]
    for _ in range(result['samples_per_backend']):
     start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
     start.record()
     for repeat in range(result['replays_per_sample']):graphs[idx].replay()
     end.record();end.synchronize();elapsed=start.elapsed_time(end)*1000
     batch_totals.append(elapsed);times.append(elapsed/result['replays_per_sample'])
    result['timings'].append(dict(fixture=fixture,order=[names[i] for i in order],backend=names[idx],measurement=a.measurement,replays_per_sample=result['replays_per_sample'],median_us=statistics.median(times),samples_us=times,event_batch_totals_us=batch_totals))
 assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in hashes.items())
 result['status']='screen_complete_no_promotion'
finally:
 dispatch._FORCED_BACKEND=None
 for key,fn in originals.items():setattr(dispatch,key,fn)
 a.output.write_text(json.dumps(result,indent=2))
