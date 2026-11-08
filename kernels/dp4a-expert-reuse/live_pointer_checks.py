"""Distinct allocations A/B/C through one live direct handle, no stale bindings."""
def run(dispatch,expert,prep,raw,hidden,directory):
 import json
 import torch
 from direct_sanitize import prepare,LiveDirect
 from fi_plan import Plan
 from moe_common import route,reference
 from verify_shared_workspace import capture,apply
 from reference_gate import check
 report=dict(status='incomplete',rows=[])
 def save(): (directory/'live_pointer_checks.json').write_text(json.dumps(report,indent=2))
 def err(y,r):return dict(finite=bool(torch.isfinite(y).all() and torch.isfinite(r).all()),rel_l2=float((y.float()-r.float()).norm()/r.float().norm().clamp_min(1e-30)))
 for m in (1,3,4):
  bundles=[]
  for offset in (0,64,128):
   x=hidden[offset:offset+m].cuda().contiguous();w,ids=route(x,raw['gate']);out=torch.empty_like(x)
   bundles.append((x,ids,w,out))
  x,ids,w,out=bundles[0]
  handle=LiveDirect(prepare(dispatch,expert,prep,x,ids,w,out,native=True))
  graphs=[capture(lambda b=b:handle(*b)) for b in bundles[:2]]
  for index,mode in [(0,'graph_A'),(1,'graph_B'),(0,'graph_A_after_B'),(2,'eager_C'),(1,'graph_B_after_C'),(0,'eager_A')]:
   x,ids,w,out=bundles[index]
   if mode.startswith('graph'):graphs[index].replay()
   else:handle(x,ids,w,out)
   saved=out.clone();fresh=torch.empty_like(x)
   Plan(dispatch,expert,prep,x,ids,w,fresh,'direct_micro',direct_multiplier=True)()
   row=dict(m=m,mode=mode,pointers=[t.data_ptr() for t in bundles[index]],exact_best_plan=bool(torch.equal(saved,fresh)))
   report['rows'].append(row);save();assert row['exact_best_plan'],'stale live tensor binding'
   torch.backends.cuda.matmul.allow_tf32=False;assert not torch.backends.cuda.matmul.allow_tf32
   ref=reference(x,raw,ids,w);base=torch.empty_like(x);apply(expert,prep,x,ids,w,base)
   row['sanity']=check({'current_scalar_cutlass':err(base,ref),'live_native':err(saved,ref)},
    {'current_scalar_cutlass':float(base.float().norm()),'live_native':float(saved.float().norm())},float(ref.float().norm()))
   assert not bool(handle.sanitizer.errors.count_nonzero())
   save()
  assert len({b[0].data_ptr() for b in bundles})==3
  assert len({b[3].data_ptr() for b in bundles})==3
 report['status']='pass_no_serving_promotion';save();return report
