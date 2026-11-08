"""Numerical-only static scale probe. No timing, no shared-source edits."""
import argparse,hashlib,json,os,sys,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
here=Path(__file__).resolve().parent
for k,v in dict(VLLM_B12X_FIX_SCALES='1',VLLM_B12X_SHARED_WORKSPACE='1',VLLM_B12X_SCALAR_INPUT_SCALE='1',VLLM_B12X_HYBRID_MIN_TOKENS='1024',VLLM_B12X_ACT_SCALES='static').items():os.environ[k]=v
sys.path.insert(0,str(here.parent/'moe-nvfp4'))
import torch
from moe_common import load_layer_raw,prep_cutlass,route,reference,quant_input
from verify_shared_workspace import apply,load_backend,capture
from verify_scalar_dispatch import prepare_exact
from test_shared_workspace import make_expert
from vllm.v1.worker import workspace as vworkspace
import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch as dispatch
from flashinfer.cute_dsl import fp4_common
result=dict(status='incomplete',scope='M8 real L19 static scale numerical diagnostic, no timing or serving promotion',checks=[],gate='exploratory: finite/nonzero, FP32 relL2<=.25, graph repeat/eager exact, input scale bytes exact, signed-zero-canonical payload exact')
def err(y,r):
 y=y.float();r=r.float()
 return dict(finite=bool(torch.isfinite(y).all() and torch.isfinite(r).all()),rel_l2=float((y-r).norm()/r.norm().clamp_min(1e-30)),max_abs=float((y-r).abs().max()),norm=float(y.norm()))
def slots(rows,blocks=128):
 # Static source row/SF coordinates, also vLLM swizzled scale layout.
 r=rows[:,None];k=torch.arange(blocks,dtype=torch.long)[None,:]
 return (r//128)*(blocks//4)*512+(k//4)*512+(r%32)*16+((r%128)//32)*4+k%4

def quant_check(ws,x,ids,prep):
 torch.cuda.synchronize()
 q,s=quant_input(x,prep['a1_gscale']);q=q.cpu().view(torch.uint8);s=s.cpu().view(torch.uint8).flatten()
 counts=ws.row_counts.cpu();mapping=ws.token_map.cpu();packed=ws.packed_input.cpu();sf=ws.packed_input_scale.cpu().flatten();experts=ws.weight_expert_ids.cpu()
 gotq=[];gots=[];pairs=[]
 stride=ws.packed_input_scale.shape[1]*ws.packed_input_scale.shape[2]
 for e,n in enumerate(counts.tolist()):
  if not n:continue
  rr=torch.arange(n,dtype=torch.long);pi=mapping[e,:n].long();assert bool(((pi>=0)&(pi<64)).all())
  assert bool((ids.cpu().flatten()[pi]==experts[e]).all())
  pairs.extend(pi.tolist());gotq.append(packed[e,:n]);gots.append(sf[e*stride+slots(rr)])
 assert sorted(pairs)==list(range(64)), 'missing/duplicate route contributions'
 pi=torch.tensor(pairs,dtype=torch.long);tokens=pi//8
 gq=torch.cat(gotq);gs=torch.cat(gots);eq=q[tokens];es=s[slots(tokens)]
 def canonical(v):
  lo=v&15;hi=v>>4;lo=torch.where(lo==8,0,lo);hi=torch.where(hi==8,0,hi);return lo|(hi<<4)
 sample=(gs!=es).nonzero()[:16];samples=[dict(pair=int(pi[r]),block=int(k),actual=int(gs[r,k]),expected=int(es[r,k])) for r,k in sample.tolist()]
 return dict(scale_mismatch_examples=samples,routes=len(pairs),scale_mismatches=int((gs!=es).sum()),scale_bytes=gs.numel(),payload_mismatches=int((gq!=eq).sum()),canonical_payload_mismatches=int((canonical(gq)!=canonical(eq)).sum()),payload_bytes=gq.numel(),scale_nonzero=int(gs.count_nonzero()))
class Static:
 def __init__(self,expert,prep,x,ids,w,out,flag):
  self.ws=dispatch.allocate_sm120_static_workspace(state_E=256,weight_E=256,max_rows=64,k=2048,n=512,num_topk=8,device=x.device,activation_precision='fp4',quant_mode='nvfp4')
  views=dispatch._get_weight_views(w1_fp4=prep['w13'],w1_blockscale=expert.w1_sf_mma,w2_fp4=prep['w2'],w2_blockscale=expert.w2_sf_mma,w1_alphas=prep['g1'],w2_alphas=prep['g2'],n=512,k=2048,activation_precision='fp4',quant_mode='nvfp4')
  self.kw=dict(a=x,topk_ids=ids,topk_weights=w,w1_weight=prep['w13'],w1_weight_sf=expert.w1_sf_mma,w1_alpha=prep['g1'],input_global_scale=prep['a13'].expand(256).contiguous(),fc2_input_scale=prep['a2'].expand(256).contiguous(),w2_weight=prep['w2'],w2_weight_sf=expert.w2_sf_mma,w2_alpha=prep['g2'],num_experts=256,top_k=8,num_local_experts=256,scatter_output=out,input_scales_are_reciprocal=flag,activation='silu',activation_precision='fp4',quant_mode='nvfp4',source_format='modelopt',_workspace=self.ws,_weight_views=views)
 def __call__(self):
  old=dispatch._FORCED_BACKEND;dispatch._FORCED_BACKEND='static'
  try:return dispatch.launch_sm120_moe(**self.kw)
  finally:dispatch._FORCED_BACKEND=old
try:
 vworkspace.init_workspace_manager(torch.device('cuda:0'))
 raw=load_layer_raw(19);prep=prep_cutlass(raw);expert=prepare_exact(load_backend('scalar'),raw,prep,make_expert)
 hidden=torch.load(here.parent/'moe-nvfp4/data/moe_inputs.pt',map_location='cpu',weights_only=True)['hidden'][19]
 x=hidden[:8].cuda().contiguous();w,ids=route(x,raw['gate']);outs=[torch.empty_like(x) for _ in range(3)]
 plans=[Static(expert,prep,x,ids,w,outs[i],flag) for i,flag in enumerate([True,False])]
 calls=[*plans,lambda:apply(expert,prep,x,ids,w,outs[2])];names=['old_static_true','corrected_static_false','current_scalar_cutlass']
 trace=[];factory=dispatch._get_static_kernel
 def traced(*args,**kw):trace.append(dict(factory='_get_static_kernel',reciprocal=kw.get('input_scales_are_reciprocal')));return factory(*args,**kw)
 dispatch._get_static_kernel=traced
 try:graphs=[capture(c) for c in calls]
 finally:dispatch._get_static_kernel=factory
 assert len(trace)>=2;result['factory_trace']=trace;result['factory_scope']='Python factory evidence, not Nsight kernel trace'
 gates=[]
 result['scales']={k:float(prep[k]) for k in ('a13','a2')};result['fp4_common_sha256']=hashlib.sha256(Path(fp4_common.__file__).read_bytes()).hexdigest()
 torch.backends.cuda.matmul.allow_tf32=False
 for offset in [0,64,0]:
  x.copy_(hidden[offset:offset+8]);nw,ni=route(x,raw['gate']);w.copy_(nw);ids.copy_(ni)
  row=dict(offset=offset,repeat={},quantization={});result['checks'].append(row)
  for i,(name,g,call) in enumerate(zip(names,graphs,calls)):
   g.replay();first=outs[i].clone();g.replay();second=outs[i].clone();call();torch.cuda.synchronize()
   row['repeat'][name]=dict(graph_repeat=err(first,second),graph_eager=err(first,outs[i]))
   if i<2:row['quantization'][name]=quant_check(plans[i].ws,x,ids,prep)
   assert torch.equal(first,second) and torch.equal(first,outs[i]), name+' graph/eager mismatch'
  torch.backends.cuda.matmul.allow_tf32=False
  assert not torch.backends.cuda.matmul.allow_tf32
  row['reference_precision']=dict(dtype='float32',allow_tf32=False)
  ref=reference(x,raw,ids,w);row['fp32']={n:err(o,ref) for n,o in zip(names,outs)};row['corrected_vs_cutlass']=err(outs[1],outs[2]);row['reference_norm']=float(ref.norm())
  # Preserve all numerical diagnostics before gates raise.
  a.output.write_text(json.dumps(result,indent=2))
  for name in names[1:]:
   e=row['fp32'][name];gates.append(dict(offset=offset,name=name,passed=e['finite'] and e['norm']>0 and e['rel_l2']<=.25))
  q=row['quantization']['corrected_static_false'];gates.append(dict(offset=offset,name='input_quant_exact',passed=q['scale_mismatches']==0 and q['canonical_payload_mismatches']==0))
 result['gates']=gates
 assert all(g['passed'] for g in gates),gates
 result['status']='numerical_gate_passed_no_promotion'
except BaseException:
 result['status']='failed';result['failure']=traceback.format_exc();raise
finally:a.output.write_text(json.dumps(result,indent=2))
