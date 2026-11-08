"""Diagnostic-only graph transitions, invalid routes and actual CUDA trace.
Independent workspace per shape/provider; sequential transitions, not concurrency.
"""
def run(dispatch, expert, prep, raw, hidden, directory):
 import json
 import torch
 from fi_plan import Plan
 from direct_sanitize import prepare as prepare_safe
 from moe_common import route, reference
 from verify_shared_workspace import capture, apply
 from reference_gate import check
 def error(y, r):
  return dict(finite=bool(torch.isfinite(y).all() and torch.isfinite(r).all()),
              rel_l2=float((y.float()-r.float()).norm()/r.float().norm().clamp_min(1e-30)))
 state={};report=dict(scope='sequential distinct per-shape/per-provider workspaces; no concurrency claim',checks=[])
 def save(): (directory/'direct_checks.json').write_text(json.dumps(report,indent=2))
 for m in [1,3,4,8]:
  x=hidden[:m].cuda().contiguous();w,ids=route(x,raw['gate'])
  out=[torch.empty_like(x) for _ in range(3)]
  plans=[prepare_safe(dispatch,expert,prep,x,ids,w,o,native=(j==2),multiplier=(j!=0)) for j,o in enumerate(out)]
  graphs=[capture(p) for p in plans]
  state[m]=(x,w,ids,out,plans,graphs)
 # Revisit prior shapes after every larger graph has been captured; each epoch
 # buffer persists through invalid, partial, valid and changed-input calls.
 for m in [1,3,4,8,3,1,8,4]:
  x,w,ids,out,plans,graphs=state[m]
  for fixture in ['valid','all_invalid','all_invalid_nan','one_token_invalid','partial_slots','duplicate_valid','restore_changed']:
   offset=64 if fixture=='restore_changed' else 0
   x.copy_(hidden[offset:offset+m]);nw,ni=route(x,raw['gate']);w.copy_(nw);ids.copy_(ni)
   if fixture in ('all_invalid','all_invalid_nan'):ids.fill_(-1)
   if fixture=='all_invalid_nan':x.fill_(float('nan'));w.fill_(float('nan'))
   elif fixture=='one_token_invalid':ids[0].fill_(-1);x[0].fill_(float('nan'));w[0].fill_(float('nan'))
   elif fixture=='partial_slots':ids[:,::2]=-1;w[:,::2]=float('nan')
   elif fixture=='duplicate_valid':ids.fill_(0)
   invalid=ids<0
   # Independent reference: ignored padded tokens use zero hidden input, and
   # ignored slots have zero weight. This does not mask native output afterward.
   safe_ids=ids.clamp_min(0);safe_w=w.masked_fill(invalid,0)
   safe_x=x.masked_fill(invalid.all(dim=1)[:,None],0)
   torch.backends.cuda.matmul.allow_tf32=False
   assert not torch.backends.cuda.matmul.allow_tf32
   ref=reference(safe_x,raw,safe_ids,safe_w)
   baseline=torch.empty_like(x);apply(expert,prep,safe_x,safe_ids,safe_w,baseline)
   saved=[];row=dict(m=m,fixture=fixture,precision='fp32_tf32_disabled',providers={});report['checks'].append(row)
   for name,p,g,o in zip(['reciprocal','preplanned_multiplier','native_prepared'],plans,graphs,out):
    g.replay();y=o.clone();g.replay();repeat=o.clone();p();eager=o.clone()
    row['providers'][name]=dict(graph_eager=error(y,eager),graph_repeat=error(y,repeat),fp32=error(y,ref))
    row['providers'][name]['sanitizer_errors']=p.sanitizer.errors.cpu().tolist()
    assert not any(row['providers'][name]['sanitizer_errors'])
    assert torch.equal(p.sanitizer.safe_x,safe_x) and torch.equal(p.sanitizer.safe_ids,safe_ids) and torch.equal(p.sanitizer.safe_weights,safe_w)
    save()
    assert torch.equal(y,eager) and torch.equal(y,repeat),'direct graph state mismatch'
    fully_invalid=invalid.all(dim=1)
    assert not bool(y[fully_invalid].count_nonzero()),'invalid token output not zero'
    row['providers'][name]['sanity']=check({'current_scalar_cutlass':error(baseline,ref),name:error(y,ref)},
      {'current_scalar_cutlass':float(baseline.float().norm()),name:float(y.float().norm())},float(ref.float().norm()))
    saved.append(y)
   if fixture in ('valid','duplicate_valid','restore_changed'):
    rawout=torch.empty_like(x);rawplan=Plan(dispatch,expert,prep,x,ids,w,rawout,'direct_micro',direct_multiplier=True);rawplan()
    assert torch.equal(saved[2],rawout),'sanitizer changed valid case'
   assert all(torch.equal(saved[0],y) for y in saved[1:]),'direct native/multiplier/reciprocal mismatch'
   save()
 # Trace actual device kernels after numerical gates, outside timing samples.
 x,w,ids,out,plans,graphs=state[1]
 nw,ni=route(x,raw['gate']);w.copy_(nw);ids.copy_(ni)
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  with torch.profiler.record_function('FI_DIRECT_PREPLANNED_NATIVE_EAGER'):plans[2]()
  with torch.profiler.record_function('FI_DIRECT_PREPLANNED_GRAPH'):graphs[2].replay()
  torch.cuda.synchronize()
 prof.export_chrome_trace(str(directory/'direct_native_trace.json'))
 events=[dict(name=e.name,device_type=str(e.device_type)) for e in prof.events() if 'CUDA' in str(e.device_type)]
 report['cuda_events']=events;save();assert events,'actual CUDA trace missing'
 report['status']='diagnostic_pass_no_serving_promotion';save()
 return report
