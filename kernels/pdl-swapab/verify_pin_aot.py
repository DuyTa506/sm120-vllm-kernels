#!/usr/bin/env python3
"""Bind guarded constructor identities to all autotuners in current AOT wrappers."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import pin_bundle_binding as B
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
S=module('symbols',HERE/'symbolic_pickle.py')
A=module('aot_opcodes',ROOT/'kernels/moe-nvfp4/audit_inductor_runtime_cpu.py')
def sha(data):return hashlib.sha256(data).hexdigest()
def config(state):
 if state.get('pre_hook') or state.get('ir_override'):raise ValueError('unsupported serialized config hook/IR')
 out=dict(state['kwargs']);out.update({k:state[k] for k in ('num_warps','num_stages','num_ctas')})
 for k in ('maxnreg','num_consumer_groups','num_buffers_warp_spec'):
  if state.get(k) is not None:out[k]=state[k]
 return out

def verify(root):
 binding=json.loads((root/'pdl_aot_binding.json').read_text())
 events=json.loads((root/'pin_events_verify.json').read_text())
 if events['status']!='passed_events_only':raise ValueError('event gate missing')
 constructors=set(events['constructor_identity_keys']);mode=events['mode']
 reference=json.loads((root/'serving/pin_reference.json').read_text())['reference'] if mode=='pin' else None
 records=[];covered=set();model_keys=set()
 for model in binding['models']:
  data=(root/model['path']).read_bytes()
  if sha(data)!=model['sha256']:raise ValueError('model artifact changed')
  wrappers={w['sha256'] for w in model['linked_wrappers']}
  pending_values=[];pending_sources={}
  for offsets,blob,ops in A.nested_opcodes(data):
   linked={sha(arg.encode()) for _,arg,_ in ops if isinstance(arg,str)}&wrappers
   if not linked:continue
   values=list(S.walk(S.load(blob)))
   tuners=[v for v in values if isinstance(v,S.Symbol) and v.tag[1]=='CachingAutotuner']
   if not tuners and mode!='pin':raise ValueError('non-pin bundle fallback unsupported')
   if mode=='pin':
    sources={sha(arg.encode()):arg for _,arg,_ in ops if isinstance(arg,str) and sha(arg.encode()) in linked}
    pending_values.extend(values);pending_sources.update(sources)
   for value in tuners:
    if not isinstance(value,S.Symbol) or value.tag[1]!='CachingAutotuner':continue
    state=value.state;key=state.get('_qwen_pin_key')
    if key not in constructors:raise ValueError('model kernel bypassed recorded constructor: '+str(key))
    configs=[config(x.state['config'].state) for x in state['compile_results']]
    if mode=='pin':
     if key not in reference or len(reference[key])!=1:raise ValueError('unknown model kernel identity')
     expected=reference[key][0]['config']
     if state.get('_qwen_pin_chosen')!=expected or configs!=[expected]:raise ValueError('model kernel config not pinned singleton')
     if state['inductor_meta'].get('dynamic_scale_rblock') is not False or state['inductor_meta'].get('coordinate_descent_tuning') is not False:raise ValueError('retuning not disabled in saved model')
    records.append({'model':model['path'],'model_sha256':model['sha256'],'wrapper_sha256':sorted(linked),'nested_offsets':offsets,'identity':key,'compiled_configs':configs,'custom_kernel':state['custom_kernel']})
    covered.update(linked);model_keys.add(key)
  if pending_sources:
   bundled,mapping=B.verify_bundle(root,pending_values,pending_sources,events,reference,S)
   for item in bundled:
    matched=sorted(digest for digest,keys in mapping.items() if item['identity'] in keys)
    records.append(dict(item,model=model['path'],model_sha256=model['sha256'],wrapper_sha256=matched,
      bundle_scope='all compiled fragments of this exact saved model; includes shared compiler helper kernels'))
   for keys in mapping.values():model_keys.update(keys)
   covered.update(mapping)
  if not wrappers<=covered:raise ValueError('wrapper lacks recorded autotuner binding')
 if not records:raise ValueError('no model-bound kernels')
 return {'status':'passed_model_bound_autotuners','mode':mode,'autotuner_records':sum('proof_mode' not in r for r in records),'bundled_records':sum('proof_mode' in r for r in records),'record_count':len(records),'model_identities':sorted(model_keys),
  'records':records,'analysis_sources':{p.name:sha(p.read_bytes()) for p in (Path(__file__),HERE/'pin_bundle_binding.py',HERE/'symbolic_pickle.py',HERE/'pin_inductor_core.py')},'event_report_sha256':sha((root/'pin_events_verify.json').read_bytes()),'binding_sha256':sha((root/'pdl_aot_binding.json').read_bytes()),
  'limitation':'Complete saved-wrapper CachingAutotuner coverage. External native GEMMs and non-CachingAutotuner kernels retain their own controls; no overlap or numerical equivalence assertion.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();x=verify(a.run);a.output.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({'status':x['status'],'autotuner_records':x['autotuner_records'],'model_identities':len(x['model_identities'])}))
