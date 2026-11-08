"""CPU-only exact generated-source -> AOT bundle -> pinned launch binding."""
import ast
import hashlib
import json
import re
from pathlib import Path
import pin_inductor_core as C

def sha(x):return hashlib.sha256(x).hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def literal(node):
 if isinstance(node,ast.Dict):return {literal(k):literal(v) for k,v in zip(node.keys,node.values)}
 if isinstance(node,ast.List):return [literal(v) for v in node.elts]
 if isinstance(node,ast.Tuple):return tuple(literal(v) for v in node.elts)
 if isinstance(node,ast.Attribute) and ast.unparse(node.value) in {'tl','triton.language'}:
  import triton.language as tl
  value=getattr(tl,node.attr,None);C.typed_dtype(value);return value
 if isinstance(node,ast.Attribute):raise ValueError('unsupported metadata attribute: '+ast.unparse(node))
 return ast.literal_eval(node)

def source_identity(source):
 tree=ast.parse(source);fns=[x for x in tree.body if isinstance(x,ast.FunctionDef) and any(ast.unparse(d)=='triton.jit' for d in x.decorator_list)]
 require(len(fns)==1,'nonunique kernel function');fn=fns[0]
 dec=[x for x in fn.decorator_list if isinstance(x,ast.Call) and ast.unparse(x.func).startswith('triton_heuristics.')]
 require(len(dec)==1,'nonunique heuristic');kw={k.arg:k.value for k in dec[0].keywords}
 meta=kw['triton_meta'];fields={ast.literal_eval(k):v for k,v in zip(meta.keys,meta.values)}
 plain={k:literal(v) for k,v in fields.items() if k not in {'device','device_type'}}
 key,evidence=C.identity(ast.unparse(fn),ast.literal_eval(kw['size_hints']) if 'size_hints' in kw else None,
  plain.get('signature'),plain.get('constants'),C.compiler_options(plain))
 return key,evidence,fn.name

def validate_choice(key,metadata,events,reference):
 require(key in events['constructor_identity_keys'],'bundle source bypassed constructor: '+key)
 require(key in reference and len(reference[key])==1,'bundle source missing reference')
 chosen=reference[key][0]['config']
 require(events['launched_configs'].get(key) and all(c==chosen for c in events['launched_configs'][key]),'bundle final singleton config differs')
 for k in ('num_warps','num_ctas','num_stages'):
  require(metadata.get(k)==chosen[k],'bundle compiled '+k+' differs')
 return chosen

def verify_bundle(root,values,wrappers,events,reference,S):
 groups=[v for v in values if isinstance(v,S.Symbol) and v.tag==('torch._inductor.triton_bundler','TritonKernelArtifacts')]
 require(groups,'missing model-bound bundle')
 records=[]
 for group in groups:
  arts={x.state['filename']:x.state['payload'] for x in group.state['artifacts']}
  require(len(arts)==len(group.state['artifacts']),'duplicate bundled filename')
  sources=[(n,p) for n,p in arts.items() if n.endswith('.source')]
  require(len(sources)==1,'missing/ambiguous bundled source');name,payload=sources[0];stem=name[:-7]
  match=re.match(r'#loc = loc\("(/bench-results/[^"\n]+\.py)":',payload.decode())
  require(match is not None,'missing exact source location')
  relative=Path('serving')/match[1].removeprefix('/bench-results/');path=root/relative
  require(path.resolve().is_relative_to((root/'serving/pdl_cache').resolve()),'source escapes isolated cache')
  raw=path.read_bytes();key,evidence,function=source_identity(raw.decode())
  require(function==stem,'bundle symbol/source mismatch')
  for ext in ('.json','.ptx','.cubin','.ttir'):require(stem+ext in arts,'missing compiled artifact '+ext)
  metadata=json.loads(arts[stem+'.json']);require(metadata['name']==function,'compiled metadata symbol mismatch')
  chosen=validate_choice(key,metadata,events,reference)
  has_wait=any(isinstance(c,ast.Call) and ast.unparse(c.func)=='tl.extra.cuda.gdc_wait' for c in ast.walk(ast.parse(raw.decode())))
  if has_wait:
   require(metadata.get('launch_pdl') is True,'PDL source missing compiled launch flag')
   require('griddepcontrol.wait' in arts[stem+'.ptx'].decode(),'PDL PTX wait missing')
  records.append({'identity':key,'compiled_configs':[chosen],'source':str(relative),'source_sha256':sha(raw),
   'launch_pdl':metadata.get('launch_pdl',False),'source_gdc_wait':has_wait,'kernel_hash':group.state['kernel_hash'],'artifacts':{n:sha(p) for n,p in arts.items()},
   'proof_mode':'exact_source_location_bundle_and_guarded_first_launch','metadata_config':{k:metadata[k] for k in ('num_warps','num_ctas','num_stages')},
   'limitation':'Full constexpr config comes from guarded actual first-launch event; bundle directly verifies warps/ctas/stages and exact source/PTX/cubin linkage.'})
 bound={x['identity'] for x in records};wrapper_keys={}
 for digest,source in wrappers.items():
  require(sha(source.encode())==digest,'wrapper source differs from actual AOT digest')
  keys=[]
  for call in ast.walk(ast.parse(source)):
   if isinstance(call,ast.Call) and ast.unparse(call.func)=='async_compile.triton':
    require(len(call.args)>=2,'missing async compiler source');raw=ast.literal_eval(call.args[1]);key,_,_=source_identity(raw)
    require(key in bound,'wrapper kernel lacks exact compiled bundle identity: '+key+' name='+ast.literal_eval(call.args[0])+' bound='+str(sorted(bound)));keys.append(key)
  require(keys,'wrapper has no guarded Triton source');wrapper_keys[digest]=keys
 return records,wrapper_keys
