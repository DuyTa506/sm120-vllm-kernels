"""One fixed baseline call inside NVTX; NCU may rerun this process independently."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
pid=os.getpid();ticks=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
passes=a.output/'passes';passes.mkdir(exist_ok=True);out=passes/f'{pid}_{ticks}';out.mkdir(exist_ok=False)
marker=dict(pid=pid,pgid=os.getpgid(pid),sid=os.getsid(pid),start_ticks=ticks,started=time.time());(out/'process.json').write_text(json.dumps(marker))
for k in ['CUTE_DSL_CACHE_DIR','FLASHINFER_WORKSPACE_BASE','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR']:os.environ[k]=str(a.output/'shared_fixed_cache'/k)
for k,v in dict(VLLM_B12X_FIX_SCALES='1',VLLM_B12X_SHARED_WORKSPACE='1',VLLM_B12X_SCALAR_INPUT_SCALE='1',VLLM_B12X_HYBRID_MIN_TOKENS='1024',VLLM_B12X_ACT_SCALES='static').items():os.environ[k]=v
sys.path.insert(0,str(HERE.parent/'moe-nvfp4'))
import torch
from moe_common import load_layer_raw,prep_cutlass,snap
from test_shared_workspace import make_expert
from verify_scalar_dispatch import prepare_exact
from verify_shared_workspace import load_backend,apply,errors
from vllm.v1.worker import workspace as ws
from protocol import RANGE
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dynamic_kernel as factory,moe_dispatch
assert factory.MoEGatedDynamicKernel.__name__=='MoEGatedDynamicKernel'
assert not moe_dispatch._DYNAMIC_KERNEL_CACHE and moe_dispatch._FORCED_BACKEND is None
native_path=Path(sys.modules[factory.MoEGatedDynamicKernel.__module__].__file__)
native_sha=hashlib.sha256(native_path.read_bytes()).hexdigest()
assert native_sha=='993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445'
original_key=moe_dispatch._dynamic_kernel_cache_key
def fixed_key(*args,**kwargs):return original_key(*args,**kwargs)+('counter_baseline_sha',native_sha)
moe_dispatch._dynamic_kernel_cache_key=fixed_key
capture=HERE.parent/'moe-task-order/results/task_order_4096_1/measurement/a1/l19_m4096_real.pt';data=torch.load(capture,map_location='cpu',weights_only=True)
identity={k:hashlib.sha256((Path(snap())/k).read_bytes()).hexdigest() for k in ['config.json','model.safetensors.index.json']}
assert identity=={'config.json':'46cb11d0ce8048b7efa09694c7725ab8be6f0c02d4b2421c1d89316aae280850','model.safetensors.index.json':'d0ab515cf0dcc4fae3daf192b9c6af1d3a9b0e79a4e699892bc18a3013315fd7'}
report=dict(status='running',native_source_sha256=native_sha,tokens=1024,layer=19,checkpoint_identity=identity,capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),range=RANGE)
def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
save()
try:
 torch.backends.cuda.matmul.allow_tf32=False;assert torch.cuda.get_device_capability()==(12,0)
 raw=load_layer_raw(19);prep=prep_cutlass(raw);ws.init_workspace_manager(torch.device('cuda:0'));e=prepare_exact(load_backend('scalar'),raw,prep,make_expert)
 x=data['input'][:1024].cuda().contiguous();ids=data['ids'][:1024].cuda().int().contiguous();weights=data['weights'][:1024].cuda().float().contiguous();result=torch.empty_like(x)
 def call():apply(e,prep,x,ids,weights,result)
 for _ in range(3):call()
 torch.cuda.synchronize();before=result.clone();w=e._ensure_wrapper();assert w._dynamic_workspace.tile_m==128;ws.lock_workspace()
 # Warmup/timing are explicitly outside the sole selected profiling range.
 start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();call();end.record();end.synchronize();report['outside_range_event_us']=start.elapsed_time(end)*1000
 torch.cuda.nvtx.range_push(RANGE)
 try:call();torch.cuda.synchronize()
 finally:torch.cuda.nvtx.range_pop()
 after=result.clone();call();torch.cuda.synchronize();report['before_after']=errors(before,after);report['after_fresh']=errors(after,result)
 for name in ['before_after','after_fresh']:assert report[name]['finite'] and report[name]['rel_l2']<=.02
 report.update(status='passed',tile_m=128,scope='Counter diagnostic; no speed or production quality claim');save()
except BaseException as exc:report.update(status='failed',exception=repr(exc));save();raise
finally:marker['ended']=time.time();(out/'process.json').write_text(json.dumps(marker))
