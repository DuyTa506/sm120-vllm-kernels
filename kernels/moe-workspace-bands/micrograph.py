"""Full scalar apply with immutable band wrappers: diagnostic correctness and graph timing."""
import argparse, hashlib, importlib.util, json, os, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT_SHA='d965685b7b29811f78efc46e0b6906e37efe601d12b21a119ca36d610059cc62'
def main():
 p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=30);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 for k in ['CUTE_DSL_CACHE_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','FLASHINFER_WORKSPACE_BASE']:os.environ[k]=str(a.output/k)
 for k,v in dict(VLLM_B12X_FIX_SCALES='1',VLLM_B12X_SHARED_WORKSPACE='1',VLLM_B12X_SCALAR_INPUT_SCALE='1',VLLM_B12X_HYBRID_MIN_TOKENS='1024',VLLM_B12X_ACT_SCALES='static',VLLM_SM120_WORKSPACE_BANDS='1').items():os.environ[k]=v
 sys.path.insert(0,str(HERE.parent/'moe-nvfp4'))
 import torch
 from vllm import envs
 from vllm.v1.worker import workspace as ws
 from moe_common import load_layer_raw,prep_cutlass,run_cutlass,snap
 from test_shared_workspace import make_expert
 from verify_scalar_dispatch import prepare_exact
 from verify_shared_workspace import load_backend,apply,errors,timing_pair,capture
 from workspace_adapter import compose_experts_source,ensure_wrapper,FLAG
 assert FLAG not in envs.environment_variables
 envs.environment_variables[FLAG]=lambda:os.getenv(FLAG,'0')=='1'
 assert torch.cuda.get_device_capability()==(12,0)
 torch.backends.cuda.matmul.allow_tf32=False
 parent=HERE.parent/'moe-nvfp4/scalar/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py'
 isolated=a.output/'candidate_backend.py';isolated.write_text(compose_experts_source(parent.read_text(),PARENT_SHA))
 spec=importlib.util.spec_from_file_location('workspace_bands_candidate',isolated);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 saved=torch.load(a.capture,map_location='cpu',weights_only=True)
 assert saved['input'].shape==(4096,2048) and saved['ids'].shape==(4096,8)
 checkpoint=Path(snap());identity={f:hashlib.sha256((checkpoint/f).read_bytes()).hexdigest() for f in ['config.json','model.safetensors.index.json']}
 assert identity=={'config.json':'46cb11d0ce8048b7efa09694c7725ab8be6f0c02d4b2421c1d89316aae280850','model.safetensors.index.json':'d0ab515cf0dcc4fae3daf192b9c6af1d3a9b0e79a4e699892bc18a3013315fd7'}
 report=dict(status='running',scope='Exact L19 capture prefixes; complete scalar apply, graph span timing; no serving or quality equivalence claim',capture_sha256=hashlib.sha256(a.capture.read_bytes()).hexdigest(),checkpoint_identity=identity,cases=[],transitions=[],gate=dict(finite=True,rel_l2_max=.02,scope='Exploratory screen vs current scalar, not production quality acceptance'))
 def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 def state(w):
  return dict(tile_m=w._dynamic_workspace.tile_m,dynamic_id=id(w._dynamic_workspace),output_ptr=w._moe_output.data_ptr(),fold_ptr=w._folded_w1_alpha.data_ptr() if w._folded_w1_alpha is not None else None,views_id=id(w._weight_views))
 save()
 try:
  raw=load_layer_raw(19);prep=prep_cutlass(raw);ws.init_workspace_manager(torch.device('cuda:0'))
  baseline=prepare_exact(load_backend('scalar'),raw,prep,make_expert);candidate=prepare_exact(module,raw,prep,make_expert)
  before=torch.cuda.memory_allocated();baseline._ensure_wrapper();base_alloc=torch.cuda.memory_allocated()-before
  before=torch.cuda.memory_allocated();candidate._ensure_wrapper(4096);band_alloc=torch.cuda.memory_allocated()-before
  report['scratch_allocation_bytes']=dict(parent=base_alloc,three_bands=band_alloc)
  contexts={}
  # Warm every band (including lazy folded alpha/weight views) before manager lock/capture.
  for m in [4096,2048,1024]:
   x=saved['input'][:m].cuda().contiguous();ids=saved['ids'][:m].cuda().int().contiguous();weights=saved['weights'][:m].cuda().float().contiguous();outputs=[torch.empty_like(x),torch.empty_like(x)]
   calls=[lambda e=e,o=o,x=x,ids=ids,weights=weights:apply(e,prep,x,ids,weights,o) for e,o in zip([baseline,candidate],outputs)]
   for call in calls:call()
   torch.cuda.synchronize()
   w=candidate._ensure_wrapper(m);assert w._dynamic_workspace.tile_m=={1024:32,2048:64,4096:128}[m]
   contexts[m]=dict(x=x,ids=ids,weights=weights,outputs=outputs,calls=calls,state=state(w))
  ws.lock_workspace()
  for m in [1024,2048,4096]:
   c=contexts[m];graphs=[capture(call) for call in c['calls']];c['graphs']=graphs
   case=dict(tokens=m,wrapper_state=c['state'],checks=[]);report['cases'].append(case)
   originals=[c[k].clone() for k in ['x','ids','weights']]
   for name in ['actual','changed_input_routes','restored']:
    if name=='changed_input_routes':
     c['x'].copy_(originals[0]*.37);c['ids'].copy_(originals[1].roll(17,0));c['weights'].copy_(originals[2].roll(17,0)*.75)
    elif name=='restored':
     for k,v in zip(['x','ids','weights'],originals):c[k].copy_(v)
    for graph in graphs:graph.replay()
    torch.cuda.synchronize();parent_out=c['outputs'][0].clone();band_out=c['outputs'][1].clone()
    c['calls'][1]();torch.cuda.synchronize()
    cut=run_cutlass(prep,c['x'],c['ids'],c['weights']);torch.cuda.synchronize()
    row=dict(name=name,band_vs_parent=errors(band_out,parent_out),band_graph_vs_fresh=errors(band_out,c['outputs'][1]),band_vs_cutlass=errors(band_out,cut))
    case['checks'].append(row);save()
    for key in ['band_vs_parent','band_graph_vs_fresh']:assert row[key]['finite'] and row[key]['rel_l2']<=.02,(m,name,key,row[key])
    assert state(candidate._ensure_wrapper(m))==c['state'],'lazy pointer changed after warmup'
   case['timing_passes']=timing_pair(graphs,repeats=a.repeats);save()
  # Revisit all captured graphs after switching bands, preserving their original resources.
  for m in [4096,1024,2048,4096,2048,1024]:
   c=contexts[m];assert state(candidate._ensure_wrapper(m))==c['state']
   for graph in c['graphs']:graph.replay()
   torch.cuda.synchronize();err=errors(c['outputs'][1],c['outputs'][0]);assert err['finite'] and err['rel_l2']<=.02
   report['transitions'].append(dict(tokens=m,state=state(candidate._ensure_wrapper(m)),error=err));save()
  report['status']='passed';report['quality_equivalence']=False;save()
 except BaseException as exc:report['status']='failed';report['exception']=repr(exc);save();raise
if __name__=='__main__':main()
