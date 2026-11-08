"""Numerical-only smoke. Explicit grant required; never benchmark timings."""
import argparse,os,sys,json,hashlib,importlib.util,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tokens',type=int,choices=[1,2,3,4,8],default=1);a=p.parse_args()
result=dict(status='incomplete',correctness_promotion=False,checks=[],stage={})
def failure_report(kind,value,tb):
 result['status']='failed';result['failure']=''.join(traceback.format_exception(kind,value,tb))
 a.output.write_text(json.dumps(result,indent=2))
 sys.__excepthook__(kind,value,tb)
sys.excepthook=failure_report
root=Path(__file__).resolve().parent
for k,v in dict(VLLM_B12X_FIX_SCALES='1',VLLM_B12X_SHARED_WORKSPACE='1',VLLM_B12X_SCALAR_INPUT_SCALE='1',VLLM_B12X_HYBRID_MIN_TOKENS='1024',VLLM_B12X_ACT_SCALES='static').items():os.environ[k]=v
sys.path.insert(0,str(root.parent/'moe-nvfp4'))
import torch
from prepare import prepare_raw,workspace,call
from fi_comparator import run as run_fi_comparator
from moe_common import load_layer_raw,prep_cutlass,route,reference,dequant
from verify_shared_workspace import apply,load_backend,capture
from verify_scalar_dispatch import prepare_exact
from test_shared_workspace import make_expert
from vllm.v1.worker import workspace as vworkspace
from vllm import _custom_ops
import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch as dispatch
spec=importlib.util.spec_from_file_location('dp4a_expert_reuse_diag',root/'build_diag/dp4a_expert_reuse_diag.so');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
spec=importlib.util.spec_from_file_location('dp4a_expert_reuse_sm120',root/'build/dp4a_expert_reuse_sm120.so');normal=importlib.util.module_from_spec(spec);spec.loader.exec_module(normal)
assert a.capture.stat().st_size>1000000
vworkspace.init_workspace_manager(torch.device('cuda:0'));torch.backends.cuda.matmul.allow_tf32=False
raw=load_layer_raw(19);prep=prep_cutlass(raw);qprep=prepare_raw(raw)
expert=prepare_exact(load_backend('scalar'),raw,prep,make_expert)
hidden=torch.load(a.capture,map_location='cpu',weights_only=True)['hidden'][19]
x=hidden[:a.tokens].cuda().contiguous();weights,ids=route(x,raw['gate'])
# Host fixture validation BEFORE capture, never GPU->CPU reads inside graph.
from route_contract import validate
def validate_routes(v):validate(v.tolist())
validate_routes(ids.cpu())
work=[];outs=[]
for _ in range(2):
 w=workspace(x.device);w['hsf']=torch.empty(64*8*(32+1536),device=x.device,dtype=torch.float32);work.append(w);outs.append(torch.empty_like(x))
calls=[lambda j=j:call(module,qprep,x,ids,weights,work[j],outs[j],bool(j)) for j in range(2)]
# CUDA wrapper guard tests reject unsafe calls before any kernel launch/memset.
original_x=x.clone()
for bad in ['short_weights','output_alias','workspace_alias']:
 w=dict(work[0]);output=outs[0];ww=weights
 if bad=='short_weights':ww=weights[:,:1].contiguous()
 if bad=='output_alias':output=x
 if bad=='workspace_alias':w['done']=w['hq'][:16]
 try:call(module,qprep,x,ids,ww,w,output,False)
 except RuntimeError:pass
 else:raise AssertionError('missing wrapper rejection: '+bad)
assert torch.equal(x,original_x),'rejected call mutated input'
graphs=[capture(c) for c in calls]
normal_work=[workspace(x.device) for _ in range(2)];normal_out=[torch.empty_like(x) for _ in range(2)]
normal_calls=[lambda j=j:call(normal,qprep,x,ids,weights,normal_work[j],normal_out[j],bool(j)) for j in range(2)]
normal_graphs=[capture(c) for c in normal_calls]
def err(y,r):
 y=y.float();r=r.float();return dict(finite=bool(torch.isfinite(y).all() and torch.isfinite(r).all()),rel_l2=float((y-r).norm()/r.norm().clamp_min(1e-30)),max_abs=float((y-r).abs().max()))
def same(a,b):return dict(exact=bool(torch.equal(a,b)),mismatch=float((a!=b).float().mean()))
LUT=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],device=x.device)
def decode(q,s,gs):
 b=q.contiguous().view(torch.uint8);v=torch.stack([LUT[(b&15).long()],LUT[(b>>4).long()]],-1).flatten(-2)
 return (v.reshape(v.shape[0],-1,16)*s.reshape(v.shape[0],-1,1)/gs).flatten(1)
def quant(v,gs):
 v=v.float().contiguous();q=torch.empty((v.shape[0],v.shape[1]//8),device=v.device,dtype=torch.int32);s=torch.empty((v.shape[0],v.shape[1]//16),device=v.device)
 module.quant_probe(v,float(gs.item()),q,s);return q,s
trace=[];originals={}
for name in ['_get_direct_micro_kernel','_get_micro_kernel','_get_static_kernel','_get_dynamic_kernel']:
 original=getattr(dispatch,name);originals[name]=original
 def wrap(*args,_fn=original,_name=name,**kwargs):trace.append(_name);return _fn(*args,**kwargs)
 setattr(dispatch,name,wrap)
runtime_paths=list((Path(dispatch.__file__).parent).glob('*.py'))+[root.parent/'moe-nvfp4'/n for n in ['moe_common.py','verify_shared_workspace.py','verify_scalar_dispatch.py','test_shared_workspace.py','scalar/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py']]
runtime_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime_paths}
(a.output.parent/'runtime_sources.json').write_text(json.dumps(runtime_hashes,indent=2))
result.update(scope='numerical-only instrumented diagnostic; no timings',tokens=a.tokens,dispatch_evidence='Python constructor/factory selection trace; NOT actual GPU kernel trace',reference_precision=[],capture_sha256=hashlib.sha256(a.capture.read_bytes()).hexdigest())
def precision(label):
 torch.backends.cuda.matmul.allow_tf32=False
 assert torch.backends.cuda.matmul.allow_tf32 is False
 result['reference_precision'].append(dict(stage=label,dtype='float32',allow_tf32=False))
try:
 fixtures=[('real_0',0),('real_64',64)]
 if a.tokens>1:fixtures.append(('synthetic_eight_hot_permuted',0))
 for fixture,offset in fixtures:
  x.copy_(hidden[offset:offset+a.tokens]);w,newids=route(x,raw['gate'])
  if fixture.startswith('synthetic_'):
   newids=torch.stack([torch.arange(8,device=x.device,dtype=torch.int32).roll(t) for t in range(a.tokens)])
  route_cpu=newids.cpu();validate_routes(route_cpu);weights.copy_(w);ids.copy_(newids)
  multiplicities=torch.bincount(route_cpu.flatten().long(),minlength=256)
  hot_counts=multiplicities[multiplicities>0].tolist()
  if fixture.startswith('synthetic_'):assert len(hot_counts)==8 and all(n==a.tokens for n in hot_counts)
  row=dict(fixture=fixture,offset=offset,unique_experts=len(hot_counts),expert_token_multiplicities=hot_counts,max_tokens_per_expert=max(hot_counts),own_graph_eager=[],baselines={});result['checks'].append(row)
  for j,g in enumerate(graphs):
   g.replay();saved=outs[j].clone();calls[j]()
   own=dict(reuse=bool(j),comparison=same(saved,outs[j]),error=err(saved,outs[j]));row['own_graph_eager'].append(own)
   assert own['comparison']['exact'],'own graph/eager mismatch'
  row['reuse_comparison']=same(outs[0],outs[1]);row['reuse_error']=err(outs[0],outs[1])
  assert row['reuse_comparison']['exact'],'reuse off/on mismatch'
  row['normal_vs_diagnostic']=[]
  for j,g in enumerate(normal_graphs):
   g.replay();saved=normal_out[j].clone();normal_calls[j]();
   record=dict(reuse=bool(j),own_graph_eager=same(saved,normal_out[j]),diagnostic=same(normal_out[j],outs[j]),error=err(normal_out[j],outs[j]))
   row['normal_vs_diagnostic'].append(record)
   assert record['own_graph_eager']['exact'] and record['diagnostic']['exact'],'instrumentation changes outputs'
  precision('full_output_offset_'+str(offset))
  reference_output=reference(x,raw,ids,weights);row['candidate_vs_fp32']=err(outs[1],reference_output)
  for name,forced in [('current_scalar',None),('fi_auto_calibrated',None),('fi_forced_direct_calibrated','direct_micro')]:
   dispatch._FORCED_BACKEND=forced;trace.clear();out=torch.empty_like(x)
   if name=='current_scalar':apply(expert,prep,x,ids,weights,out)
   else:run_fi_comparator(dispatch,expert,raw,prep,x,ids,weights,out)
   row['baselines'][name]=dict(factory_selection_trace=list(trace),candidate_error=err(outs[1],out),fp32_error=err(out,reference_output),scale_contract='current scalar CUTLASS' if name=='current_scalar' else 'explicit launch_sm120_moe reciprocal=True; calibrated a13/a2; weight alpha folded once internally')
 # Last fixture (explicitly synthetic hot routes when M>1): per-stage diagnosis.
 result['stage_fixture']=fixtures[-1][0]
 calls[0]();T=a.tokens;flat=ids.flatten().long();tok=torch.arange(T,device=x.device).repeat_interleave(8);slot=torch.arange(T*8,device=x.device)
 debug=work[0]['hsf'][64*8*32:].reshape(64,8,1536)[slot,tok]
 q1,s1=quant(x,qprep['a1']);fiq,fis=_custom_ops.scaled_fp4_quant(x,qprep['a1'],is_sf_swizzled_layout=False)
 fis=fis.view(torch.float8_e4m3fn)[:T,:128]
 result['stage']['fc1_quant_payload_vs_vllm']=same(q1.view(torch.uint8),fiq.view(torch.uint8))
 result['stage']['fc1_quant_scale_vs_vllm']=same(s1.to(torch.float8_e4m3fn).view(torch.uint8),fis.view(torch.uint8))
 dx=decode(q1,s1,qprep['a1']);W13=dequant(raw['w13'],raw['s13'],raw['wgs13']);W2=dequant(raw['w2'],raw['s2'],raw['wgs2'])
 precision('fc1_exact_quant_operand')
 ref1=torch.stack([dx[tok[k]]@W13[flat[k]].T for k in range(T*8)])
 actual1=torch.cat([debug[:,512:1024],debug[:,:512]],1)
 result['stage']['fc1_precast_vs_exact_quant_operand_fp32']=err(actual1,ref1)
 expected_act=(torch.nn.functional.silu(actual1[:,:512].bfloat16().float())*actual1[:,512:].bfloat16().float()).bfloat16().float()
 result['stage']['swiglu_bf16']=err(debug[:,1024:],expected_act)
 q2,s2=quant(debug[:,1024:],qprep['a2']);hq=work[0]['hq'].reshape(64,8,64)[slot,tok];hs=work[0]['hsf'][:64*8*32].reshape(64,8,32)[slot,tok]
 result['stage']['fc2_quant_payload_vs_own_probe']=same(hq,q2);result['stage']['fc2_quant_scale_vs_own_probe']=same(hs,s2)
 assert torch.equal(hq,q2) and torch.equal(hs,s2),'FC2 quant inconsistent with same helper'
 dq=decode(hq,hs,qprep['a2'])
 precision('fc2_exact_quant_operand')
 ref2=torch.stack([(dq[k]@W2[flat[k]].T)*weights.flatten()[k] for k in range(T*8)])
 parts=work[0]['part'].reshape(64,8,2048)[slot,tok]
 result['stage']['fc2_weighted_part_vs_exact_quant_operand_fp32']=err(parts,ref2)
 # Real graph replay after all-invalid and partially padded input; independent workspaces.
 valid_ids=ids.clone();valid_w=weights.clone()
 for mode in ['all_invalid','one_row_invalid']:
  ids.copy_(valid_ids);weights.copy_(valid_w)
  if mode=='all_invalid':ids.fill_(-1)
  else:ids[0].fill_(-1)
  for j,g in enumerate(graphs):
   g.replay();saved=outs[j].clone();calls[j]();assert torch.equal(saved,outs[j]);assert torch.count_nonzero(outs[j][0])==0
   if mode=='all_invalid':assert torch.count_nonzero(outs[j])==0
  assert torch.equal(outs[0],outs[1])
  for j,g in enumerate(normal_graphs):
   g.replay();assert torch.equal(normal_out[j],outs[j]),'normal/diagnostic padded mismatch'
 ids.copy_(valid_ids);weights.copy_(valid_w)
 for j,g in enumerate(graphs):g.replay();saved=outs[j].clone();calls[j]();assert torch.equal(saved,outs[j])
 result['padding_changed_route_two_workspace']=True
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in runtime_hashes.items()),'runtime source changed'
 assert all(c['candidate_vs_fp32']['finite'] for c in result['checks'])
 quant_exact=all(result['stage'][key]['exact'] for key in ['fc1_quant_payload_vs_vllm','fc1_quant_scale_vs_vllm'])
 result['quant_contract_exact']=quant_exact
 result['status']='diagnostic_complete_quant_exact' if quant_exact else 'diagnostic_complete_quant_mismatch'
 result['correctness_promotion']=False
finally:
 dispatch._FORCED_BACKEND=None
 for name,fn in originals.items():setattr(dispatch,name,fn)
 a.output.write_text(json.dumps(result,indent=2))
