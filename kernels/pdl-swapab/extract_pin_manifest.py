#!/usr/bin/env python3
"""Extract a fail-closed launch reference from model-bound inert AOT data."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
P=module('pin_core',HERE/'pin_inductor_core.py')
S=module('symbolic',HERE/'symbolic_pickle.py')
A=module('audit_opcodes',ROOT/'kernels/moe-nvfp4/audit_inductor_runtime_cpu.py')
def sha(data):return hashlib.sha256(data).hexdigest()
def source_identity(source):
 tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
 decorators=[x for x in fn.decorator_list if isinstance(x,ast.Call) and ast.unparse(x.func).startswith('triton_heuristics.')]
 if len(decorators)!=1:raise ValueError('missing unique heuristic decorator')
 kw={k.arg:k.value for k in decorators[0].keywords};meta=kw['triton_meta']
 fields={ast.literal_eval(k):v for k,v in zip(meta.keys,meta.values)}
 return P.identity(ast.unparse(fn),ast.literal_eval(kw['size_hints']) if 'size_hints' in kw else None,
  ast.literal_eval(fields['signature']),ast.literal_eval(fields['constants']) if 'constants' in fields else None)
def config_state(state):
 if state.get('pre_hook') or state.get('ir_override'):raise ValueError('nontrivial config hook/IR override')
 out=dict(state['kwargs']);out.update({k:state[k] for k in ('num_warps','num_stages','num_ctas')})
 for k in ('maxnreg','num_consumer_groups','num_buffers_warp_spec'):
  if state.get(k) is not None:out[k]=state[k]
 return out

def extract(root):
 audit=json.loads((root/'serving/fi_control/autotune/compiler_config_audit.json').read_text())
 best={}
 for entry in audit['files']:
  path=root/'serving/pdl_cache/vllm'/entry['path']
  if sha(path.read_bytes())!=entry['sha256']:raise ValueError('config hash mismatch')
  candidates=[]
  for q in path.parent.glob('*.py'):
   if any(isinstance(n,ast.FunctionDef) and n.name.startswith('triton_') for n in ast.parse(q.read_text()).body):candidates.append(q)
  source,=candidates;key,evidence=source_identity(source.read_text())
  cfg=dict(entry['selected']);cfg.setdefault('num_ctas',1)
  best.setdefault(key,[]).append({'config':cfg,'source':str(source.relative_to(root))})
 binding=json.loads((root/'pdl_aot_binding.json').read_text());records={};unresolved=[];counts={};seen=set()
 for model in binding['models']:
  path=root/model['path'];data=path.read_bytes()
  if sha(data)!=model['sha256']:raise ValueError('model hash mismatch')
  hashes={w['sha256'] for w in model['linked_wrappers']}
  for offsets,blob,ops in A.nested_opcodes(data):
   linked={sha(arg.encode()) for _,arg,_ in ops if isinstance(arg,str)}&hashes
   if not linked:continue
   obj=S.load(blob)
   for v in S.walk(obj):
    if not isinstance(v,S.Symbol):continue
    if v.tag[1] not in ('CachingAutotuner',):continue
    d=v.state;fn=d['fn'];raw=fn.state.get('_src') or ''.join(fn.state['raw_src'])
    key,evidence=P.identity(raw,d['size_hints'],d['triton_meta'].get('signature'),d['triton_meta'].get('constants'))
    configs=[config_state(c.state['config'].state) for c in d['compile_results']]
    unique={json.dumps(c,sort_keys=True):c for c in configs}
    choices={json.dumps(e['config'],sort_keys=True):e['config'] for e in best.get(key,[])}
    if len(unique)==1:chosen=next(iter(unique.values()));kind='singleton_serialized_candidate'
    elif len(choices)==1 and next(iter(choices)) in unique:chosen=next(iter(choices.values()));kind='matched_best_config'
    else:
     unresolved.append({'identity':key,'name':fn.state.get('__name__'),'reason':'multi-candidate without unique matching best_config','candidates':configs,'best':best.get(key,[]),'custom_kernel':d['custom_kernel'],'source':raw,'identity_evidence':evidence});continue
    record={'config':chosen,'identity_evidence':evidence,'source':raw,'custom_kernel':d['custom_kernel'],
     'selection_evidence':kind,'model':model['path'],'model_sha256':model['sha256'],'wrapper_sha256':sorted(linked),'nested_offsets':offsets,
     'triton_meta':d['triton_meta'],'inductor_meta':d['inductor_meta']}
    # Metadata may contain inert DeviceProperties objects; only strict identity fields above enter reference.
    record.pop('triton_meta');record.pop('inductor_meta')
    if key in records and records[key][0]['config']!=chosen:
     unresolved.append({'identity':key,'reason':'ambiguous repeated selection','first_config':records[key][0]['config'],'later_config':chosen,'first_source':records[key][0]['source'],'later_source':raw,'meta':str(d['triton_meta'])});continue
    records.setdefault(key,[record]);counts[kind]=counts.get(kind,0)+1
 return {'status':'complete' if not unresolved else 'incomplete_fail_closed','reference':records,'unresolved':unresolved,
  'selection_counts':counts,'source_run':str(root),'binding_sha256':sha((root/'pdl_aot_binding.json').read_bytes()),
  'scope':'All CachingAutotuner objects bound to saved startup wrappers. Native/external/custom Triton non-CachingAutotuner paths are outside this pin.',
  'script_sha256':sha(Path(__file__).read_bytes())}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();x=extract(a.run);a.output.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({'status':x['status'],'identities':len(x['reference']),'unresolved':len(x['unresolved']),'selection_counts':x['selection_counts']}))
