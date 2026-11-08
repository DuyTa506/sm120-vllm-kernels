"""Actual V2 worker lifecycle and live graph replay evidence; no fake context."""
import os,json,hashlib,functools
from pathlib import Path
LIVE=False
REPLAYS={}
def register():
 if not os.environ.get('FI_DIRECT_RUNTIME_OUT'):return
 import torch
 from vllm.v1.worker.gpu_worker import Worker
 from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
 from vllm.v1.worker.workspace import current_workspace_manager
 original_graph=ModelCudaGraphManager.run_fullgraph
 @functools.wraps(original_graph)
 def graph(manager,descriptor,*args,**kwargs):
  m=int(descriptor.num_tokens)
  if LIVE:torch.cuda.nvtx.range_push('FI_DIRECT_LIVE_GRAPH_M'+str(m))
  try:return original_graph(manager,descriptor,*args,**kwargs)
  finally:
   if LIVE:torch.cuda.nvtx.range_pop();REPLAYS[m]=REPLAYS.get(m,0)+1
 ModelCudaGraphManager.run_fullgraph=graph
 def report(worker,name):
  from vllm import envs
  runner=worker.model_runner
  assert type(runner).__module__=='vllm.v1.worker.gpu.model_runner'
  spec=runner.vllm_config.speculative_config
  assert spec.method=='mtp' and spec.num_speculative_tokens==2 and spec.num_speculative_tokens_per_batch_size is None
  enabled=envs.VLLM_SM120_FI_DIRECT_SMALL_M
  assert enabled==(os.environ['FI_DIRECT_PROFILE']=='on')
  composition=json.loads((Path(__file__).parent/'generated/composition.json').read_text())
  for relative,sha in composition['generated'].items():
   path=Path(envs.__file__).parent/relative.removeprefix('vllm/')
   assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,relative
  import inspect
  import vllm.v1.worker.workspace as workspace_module
  modules=[workspace_module,inspect.getmodule(type(runner)),inspect.getmodule(ModelCudaGraphManager),inspect.getmodule(type(runner.speculator))]
  runtime_sources={m.__file__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in modules if m is not None and getattr(m,'__file__',None)}
  manager=current_workspace_manager();resources=[]
  for owner,cache in enumerate(manager._persistent_resources):
   for key,value in cache.items():
    if isinstance(key,tuple) and key[0]=='sm120_fi_direct_small_m':
     resources.append(dict(owner=owner,expert=id(key[1]),m=int(key[2]),sanitizer_bytes=value.sanitizer.bytes,
      warmup_pointers_released=value.sanitizer.x is None and 'out' not in value.native.kw))
  if enabled:
   assert resources and all(r['warmup_pointers_released'] for r in resources)
   by_owner={}
   for r in resources:by_owner.setdefault((r['owner'],r['expert']),set()).add(r['m'])
   assert all(shapes=={1,3,4} for shapes in by_owner.values()),'missing per-owner/per-layer M1/M3/M4 warmup coverage'
  else:assert not resources
  factors=envs.compile_factors();assert factors['VLLM_SM120_FI_DIRECT_SMALL_M']==enabled
  torch.cuda.synchronize()
  record=dict(stage=name,pid=os.getpid(),runner=type(runner).__module__,enabled=enabled,
   workspace_locked=manager.is_locked(),resources=resources,live_graph_replays=REPLAYS,
   memory_allocated=torch.cuda.memory_allocated(),memory_reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),
   compile_factors_sha256=hashlib.sha256(repr(sorted(factors.items())).encode()).hexdigest(),
   speculative_config=dict(method=spec.method,num_speculative_tokens=spec.num_speculative_tokens,num_speculative_tokens_per_batch_size=spec.num_speculative_tokens_per_batch_size),speculator_class=type(runner.speculator).__module__+'.'+type(runner.speculator).__qualname__,
   runtime_source_hashes=runtime_sources,source_verified=True)
  out=Path(os.environ['FI_DIRECT_RUNTIME_OUT']);out.mkdir(parents=True,exist_ok=True)
  (out/name).write_text(json.dumps(record,indent=2));return record
 original=Worker.compile_or_warm_up_model
 @functools.wraps(original)
 def warmup(worker,*args,**kwargs):
  global LIVE
  result=original(worker,*args,**kwargs);report(worker,'startup.json');LIVE=True;return result
 Worker.compile_or_warm_up_model=warmup
 def post(worker):
  assert LIVE
  import qwen_fi_control_plugin as fi
  assert fi._owner is not None
  fi._owner.finalize(stage='server_post_workload');fi._owner.write_report()
  record=report(worker,'post_workload.json')
  assert record['live_graph_replays'],'no actual live graph replay'
  return record
 Worker.fi_direct_post_workload=post
