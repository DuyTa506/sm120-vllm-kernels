import ast,hashlib,json,math,re
from pathlib import Path
from collections import Counter
root=Path.cwd();r=Path('results/qwen_sequential/gdn_integration_2');out=Path('kernels/gdn-norm-fp8/results')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert (r/'exit_code.txt').read_text().strip()=='0'
ready=json.loads((r/'gdn_readiness.json').read_text());assert ready['ready_layers']==ready['expected_layers'] and len(ready['ready_layers'])==30 and not ready['disabled']
verified=json.loads((r/'source_verify.log').read_text());assert verified['status']=='passed' and verified['files']==22

def compilation_config(log):
 text=log.read_text()
 line=next(x for x in text.splitlines() if 'compilation_config={' in x and 'kernel_config=' in x)
 raw=line.split('compilation_config=',1)[1].split(', kernel_config=',1)[0]
 plain=re.sub(r'<([A-Za-z0-9_.]+):[^>]*>',lambda m:repr(m.group(1)),raw)
 return ast.literal_eval(plain)
resolved=compilation_config(r/'docker.log')
cli=json.loads((r/'compilation_config.json').read_text())
assert resolved['splitting_ops']==cli['splitting_ops'] and len(resolved['splitting_ops'])==20
assert resolved['pass_config']['fuse_rope_kvcache_cat_mla'] is False
assert resolved['backend']=='inductor' and resolved['custom_ops']==['none'] and resolved['use_inductor_graph_partition'] is False
baseline_path=Path('results/qwen_sequential/exact_tasks_1/docker.log')
baseline=compilation_config(baseline_path)
normalized={**resolved,'splitting_ops':[x for x in resolved['splitting_ops'] if x!='vllm::gdn_attention_core_norm_fp8_packed_experimental']}
config_equal=normalized==baseline
if not config_equal: print('resolved differences',{k:(baseline.get(k),v) for k,v in normalized.items() if v!=baseline.get(k)})

cache=Path('.cache/vllm/qwen-sequential-shared_exact_gdn/torch_compile_cache/c35103b048/rank_0_0')
graph_path=cache/'backbone/computation_graph.py';source=graph_path.read_text();tree=ast.parse(source)
classes={n.name:n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name.startswith('submod_')}
fn=lambda cls:next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='forward')
target='torch.ops.vllm.gdn_attention_core_norm_fp8_packed_experimental';cutlass='torch.ops._C.cutlass_scaled_mm'
edges=[]
for name,cls in classes.items():
 calls=[x for x in ast.walk(fn(cls)) if isinstance(x,ast.Call) and ast.unparse(x.func)==target]
 if not calls:continue
 assert len(calls)==1
 call=calls[0];q=ast.unparse(call.args[3]);scale=ast.unparse(call.args[4]);consumer=classes['submod_'+str(int(name.split('_')[1])+1)];body=fn(consumer)
 args={x.arg:ast.unparse(x.annotation) if x.annotation else None for x in body.args.args}
 assert 'f8e4m3fn' in args[q] and 'f32' in args[scale]
 aliases={q:q};allowed=[];mm=None
 for stmt in body.body:
  found=[n for n in ast.walk(stmt) if isinstance(n,ast.Call) and ast.unparse(n.func)==cutlass]
  if found:
   assert len(found)==1;mm=found[0];break
  for n in ast.walk(stmt):
   if isinstance(n,ast.Call):
    f=ast.unparse(n.func);allowed.append(f)
    assert f.endswith('.view') or f in ('torch.empty','device'),f
  if isinstance(stmt,ast.AnnAssign) and isinstance(stmt.value,ast.Call) and isinstance(stmt.value.func,ast.Attribute) and stmt.value.func.attr=='view':
   inp=ast.unparse(stmt.value.func.value)
   if inp in aliases:aliases[ast.unparse(stmt.target)]=aliases[inp]
 assert mm is not None and aliases[ast.unparse(mm.args[1])]==q and ast.unparse(mm.args[3])==scale
 weight=ast.unparse(mm.args[2]);layer=int(re.search(r'layers_modules_(\d+)_modules_linear_attn_modules_out_proj',weight).group(1))
 edges.append(dict(layer=layer,producer_submodule=name,producer_line=call.lineno,consumer_submodule=consumer.name,
                   cutlass_line=mm.lineno,q_input=q,scale_input=scale,q_path='views only',pre_gemm_calls=allowed,
                   cutlass_call=ast.unparse(mm),extra_quantization=False))
assert sorted(x['layer'] for x in edges)==ready['expected_layers']
allcalls=Counter(ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call))
assert allcalls[target]==30
old_calls=sum(n for k,n in allcalls.items() if k.startswith('torch.ops.vllm.qwen_gdn_attention_core'))
assert old_calls==0

decode=[]
for c,count in ((1,8),(4,24),(16,96),(64,384)):
 p=r/f'serving/serving_matrix_decode/decode-c{c}.measured.json';d=json.loads(p.read_text())
 assert d['completed']==count and d['failed']==0 and len(d['input_lens'])==len(d['output_lens'])==count
 assert set(d['input_lens'])=={128} and set(d['output_lens'])=={512}
 decode.append(dict(concurrency=c,measured_requests=count,completed=count,failed=0,input_tokens=128,output_tokens=512,
                    source=str(p),sha256=sha(p)))
nll_path=r/'repeated_nll.json';nll=json.loads(nll_path.read_text());assert nll['status']=='passed' and nll['rounds']==1 and len(nll['results'])==10
nll_results=[]
for d in nll['results']:
 assert math.isfinite(d['nll']) and math.isfinite(d['nll_last_half']) and all(math.isfinite(x) for x in d['logprobs'])
 assert d['completion_tokens']==1 and len(d['logprobs'])==d['scored_tokens']==d['prompt_tokens']-1
 nll_results.append({k:d[k] for k in ('prompt_id','round','prompt_tokens','scored_tokens','completion_tokens','nll','nll_last_half')})
assert len({x['cache_salt'] for x in nll['schedule']})==10
logs=(r/'docker.log').read_text().splitlines()
completions=[]
for i,l in enumerate(logs,1):
 if 'Capturing' in l and '100%' in l:
  label=l.split(':',1)[0];m=re.search(r' (\d+)/(\d+) ',l)
  if m:
   item=dict(label=label,completed=int(m.group(1)),total=int(m.group(2)))
   if not completions or completions[-1]['label']!=label or completions[-1]['total']!=item['total']:
    completions.append({**item,'log_line':i})
finishes=[dict(log_line=i,text=l) for i,l in enumerate(logs,1) if 'Graph capturing finished' in l]
jit=[dict(log_line=i,text=l) for i,l in enumerate(logs,1) if 'JIT compilation during inference:' in l]
assert not any('norm_gate_fp8' in x['text'] or 'quant_out' in x['text'] for x in jit)
tracked=[r/'docker.log',r/'config.txt',r/'exit_code.txt',r/'gdn_readiness.json',r/'source_manifest.json',r/'source_verify.log',r/'compilation_config.json',nll_path,
         graph_path,cache/'backbone/cache_key_factors.json',cache/'eagle_head/computation_graph.py',cache/'eagle_head/cache_key_factors.json',baseline_path]
report=dict(status='STATEFUL_INTEGRATION_PASS_SCOPED',run=str(r),variant='shared_exact_gdn',fi_control=0,
            runner_exit_code=0,ready_layers=ready['ready_layers'],source_verification=verified,
            resolved_compilation_config=resolved,baseline_compilation_config_source=str(baseline_path),
            resolved_config_matches_baseline_after_removing_gdn_split=config_equal,
            graph=dict(cache_id='c35103b048',new_gdn_ops=30,old_gdn_ops=0,qa_direct_cutlass_edges=edges,
                       proof_scope='Saved split FX shows each FP8 Q consumed through views by existing CUTLASS with same scale; no re-quantization between producer and GEMM. Stateful CUDA implementation remains opaque.'),
            capture_completions=completions,graph_capture_finished=finishes,inference_jit_warnings=jit,
            decode=dict(cases=decode,total_measured_requests=sum(x['measured_requests'] for x in decode),temperature=0),
            nll=dict(rounds=1,requests=10,finite_all_scored_logprobs=True,unique_cache_salts=10,
                     request_settings=nll['request_settings'],results=nll_results,
                     high_prefill_request_count=sum(x['prompt_tokens']>=1024 for x in nll_results)),
            source_sha256={str(p):sha(p) for p in tracked},
            scope='Successful startup, graph capture, actual decode and scored-prefill smoke. No paired quality/state equivalence or speedup claim; FI control disabled, one NLL round, request-conditioned mixed-state coverage not asserted.')
(out/'integration_gpu_summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(config_equal=config_equal,ops=30,edges=len(edges),decode_requests=report['decode']['total_measured_requests'],nll=10,nll_prompt_tokens=[d['prompt_tokens']for d in nll_results],captures=completions),indent=2))
