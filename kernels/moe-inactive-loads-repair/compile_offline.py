"""Isolated fake-tensor compile; CUDA hidden, fixed recorded188-SM host metadata."""
import ast,hashlib,importlib.util,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
mode=sys.argv[2] if len(sys.argv)>2 else 'repair'
assert mode in ('baseline','repair')
fast_math=len(sys.argv)>3 and sys.argv[3]=='fast'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
os.environ['CUTE_DSL_ARCH']='sm_120a'
output=Path(sys.argv[1]);output.mkdir(exist_ok=False,parents=True)
os.environ['CUTE_DSL_CACHE_DIR']=str(output/'cute_cache')
import torch
assert not torch.cuda.is_initialized()
def forbidden(*a,**k):raise RuntimeError('device query forbidden during offline compile')
torch.cuda._lazy_init=forbidden
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as d,moe_dynamic_kernel as f
import inspect
source=inspect.getsource(d._get_dynamic_kernel)
# These two queries are host scheduling inputs, fixed to previously measured188.
assert source.count('sm_count = get_num_sm(torch.device("cuda"))')==1
source=source.replace('sm_count = get_num_sm(torch.device("cuda"))','sm_count = 188').replace('base_mac = min(get_max_active_clusters(1), sm_count)','base_mac = 188')
name=f.MoEGatedDynamicKernel.__module__.rsplit('.',1)[0]+'.gated_defined_fc2_offline'
spec=importlib.util.spec_from_file_location(name,HERE/'gated_defined_fc2.py');module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
if mode=='repair':f.MoEGatedDynamicKernel=module.MoEGatedDynamicKernelDefinedFC2
namespace=dict(d.__dict__)
def compile_only(mod,name,callback,**kwargs):
 result=callback()
 result.export_to_c(str(output/'compiled.o'), function_name='inactive_defined_fc2')
 return result
namespace['build_and_load_cute_dsl_kernel']=compile_only
exec(compile(source,'<isolated_fake_compile>','exec'),namespace)
record=dict(cuda_visible_devices='',arch='sm_120a',recorded_sms=188,mode=mode,fast_math=fast_math,source_sha256=hashlib.sha256((HERE/'gated_defined_fc2.py' if mode=='repair' else HERE.parent/'moe-inactive-loads/gated_baseline.py').read_bytes()).hexdigest(),status='started')
try:
 namespace['_get_dynamic_kernel'](256,1024,2048,512,8,16384*8,topk_ids_dtype=torch.int32,input_scales_are_reciprocal=True,fast_math=fast_math,activation='silu',share_input_across_experts=True,tile_m=128)
 record['status']='compiled'
except BaseException as exc:record['status']='failed';record['exception']=repr(exc);raise
finally:
 record['torch_cuda_initialized']=torch.cuda.is_initialized();(output/'report.json').write_text(json.dumps(record,indent=2)+'\n');assert not torch.cuda.is_initialized()
