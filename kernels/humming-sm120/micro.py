"""Hot-layer full expert-call screen; no serving/quality promotion from timings."""
import argparse,hashlib,json,os,statistics,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tokens',type=int,default=1024);p.add_argument('--layer',type=int,default=19);p.add_argument('--layout',choices=['indexed','grouped'],default='indexed');p.add_argument('--rounding',choices=['ru','rn'],default='ru');a=p.parse_args()
assert a.capture.stat().st_size>1000000
for k,v in dict(VLLM_B12X_FIX_SCALES='1',VLLM_B12X_SHARED_WORKSPACE='1',VLLM_B12X_SCALAR_INPUT_SCALE='1',VLLM_B12X_HYBRID_MIN_TOKENS='1024',VLLM_B12X_ACT_SCALES='static',VLLM_HUMMING_MOE_GEMM_TYPE=a.layout).items():os.environ[k]=v
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'moe-nvfp4'))
import torch
from rounding_mode import install
rounding_identity=install(a.rounding)
from moe_common import load_layer_raw,prep_cutlass,route,reference
from verify_shared_workspace import apply,capture,errors,load_backend
from verify_scalar_dispatch import prepare_exact
from test_shared_workspace import make_expert
from vllm.v1.worker import workspace
from prepare import prepare
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
import importlib.metadata,inspect,shutil
import humming
from vllm.model_executor.layers.fused_moe.experts import fused_humming_moe
from vllm.model_executor.layers.quantization.utils.humming import moe as humming_moe
base=Path(__file__).resolve().parent.parent/'moe-nvfp4'
source_paths=[base/n for n in ('moe_common.py','verify_shared_workspace.py','verify_scalar_dispatch.py','test_shared_workspace.py','scalar/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py')]
source_paths += [Path(inspect.getsourcefile(m)) for m in (fused_humming_moe,humming_moe)]
source_paths += list(Path(humming.__file__).parent.rglob('*.py')) + list(Path(humming.__file__).parent.rglob('*.cuh'))
source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
(a.output.parent/'runtime_sources.json').write_text(json.dumps(dict(humming_version=importlib.metadata.version('humming-kernels'),source_sha256=source_hashes),indent=2))
archive=a.output.parent/'helper_snapshot';archive.mkdir(exist_ok=False)
for i,path in enumerate(source_paths[:7]):shutil.copy2(path,archive/(str(i)+'_'+path.name))
workspace.init_workspace_manager(torch.device('cuda:0'))
torch.backends.cuda.matmul.allow_tf32=False
hidden=torch.load(a.capture,map_location='cpu',weights_only=True)['hidden'][a.layer]
assert len(hidden)>=2*a.tokens
raw=load_layer_raw(a.layer);prep=prep_cutlass(raw);baseline=prepare_exact(load_backend('scalar'),raw,prep,make_expert)
layer,expert,make_call=prepare(raw,a.layout,a.layer)
backend_record=dict(expert_class=type(expert).__module__+'.'+type(expert).__qualname__,compute_config=expert.compute_config,input_schemas={k:str(v) for k,v in layer.input_schemas.items()},weight_schemas={k:str(v) for k,v in layer.weight_schemas.items()},layer_configs={k:v.to_str() for k,v in layer.humming_configs.items()},effective_input_scale_2={k:getattr(layer,k+'_input_scale_2').cpu().tolist() for k in ('w13','w2')})
(a.output.parent/'backend.json').write_text(json.dumps(backend_record,indent=2,default=str))
x=hidden[:a.tokens].cuda().contiguous();weights,ids=route(x,raw['gate']);outs=[torch.empty_like(x) for _ in range(3)]
calls=[lambda:apply(baseline,prep,x,ids,weights,outs[0]),lambda:baseline._apply_cutlass(outs[1],x,prep['w13'],prep['w2'],weights,ids,MoEActivation.SILU),make_call(x,ids,weights,outs[2])]
backend_names=['scalar','cutlass','humming']
graphs=[capture(call) for call in calls]
result=dict(regime='hot-layer expert call includes backend route packing, activation quant, GEMMs and reduction; excludes router GEMM/topK and initial weight prep; no promotion',rounding=rounding_identity,backend_names=backend_names,layout=a.layout,layer=a.layer,tokens=a.tokens,capture_sha256=hashlib.sha256(a.capture.read_bytes()).hexdigest(),checks=[],timings=[])
for offset in [0,a.tokens]:
    x.copy_(hidden[offset:offset+a.tokens]);neww,newids=route(x,raw['gate']);weights.copy_(neww);ids.copy_(newids)
    samples=[[] for _ in calls]
    for i in range(8):
        for j,g in enumerate(graphs):g.replay();samples[j].append(outs[j].clone())
    torch.cuda.synchronize()
    checks=dict(offset=offset,self_repeat=[[errors(t,seq[0]) for t in seq[1:]] for seq in samples],cross={f'{backend_names[left]}_vs_{backend_names[right]}':[errors(samples[left][i],samples[right][i]) for i in range(8)] for left,right in [(2,0),(2,1),(0,1)]},graph_eager=[])
    for j,call in enumerate(calls):call();checks['graph_eager'].append(errors(outs[j],samples[j][-1]))
    assert all(torch.isfinite(t).all() for seq in samples for t in seq), 'nonfinite outputs'
    # Independent original-checkpoint FP32 reference, without activation
    # quantization, on EACH disjoint real capture slice. Not a timing region.
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.backends.cuda.matmul.allow_tf32 is False
    checks['reference_precision']=dict(dtype='float32',allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    fp32_reference=reference(x,raw,ids,weights)
    checks['fp32_reference']={backend_names[j]:dict(graph=errors(samples[j][-1],fp32_reference),eager=errors(outs[j],fp32_reference)) for j in range(len(calls))}
    assert bool(torch.isfinite(fp32_reference).all())
    del fp32_reference
    result['checks'].append(checks)
for backend in [0,1,2,2,1,0]:
    times=[]
    for _ in range(40):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();graphs[backend].replay();end.record();end.synchronize();times.append(start.elapsed_time(end)*1000)
    result['timings'].append(dict(backend=backend_names[backend],median_us=statistics.median(times),samples_us=times))
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in source_hashes.items()), 'runtime sources changed'
a.output.write_text(json.dumps(result,indent=2))
# Deliberately diagnostic: no relaxed accuracy gate or automatic policy promotion.
