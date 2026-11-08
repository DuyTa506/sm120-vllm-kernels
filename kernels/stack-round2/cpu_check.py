"""Execute extracted *actual* dispatcher bodies with CPU metadata/mocked launches.
No tensor kernels execute. A separate --runtime-env check loads real envs in CUDA-hidden image.
"""
import argparse, ast, hashlib, importlib.util, json, os, sys, types
from pathlib import Path
S=types.SimpleNamespace
MS=(1,16,32,64,65,128,129,1024,4096)
class Tensor:
    def __init__(self,shape,dtype='fp8',stride=None):self.shape=shape;self.dtype=dtype;self.device=S(index=0);self.is_cuda=True;self.ndim=len(shape);self._stride=stride
    def numel(self):
        import math
        return math.prod(self.shape)
    def is_contiguous(self):return True
    def data_ptr(self):return 16
    def stride(self,axis=None):return self._stride if axis is None else self._stride[axis]
    def dim(self):return self.ndim
    @property
    def T(self):return self
    def t(self):return self
    def reshape(self,*args):return self
    def view(self,*args):return self

def extract(path,names,ns):
    tree=ast.parse(path.read_text());nodes=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod),str(path),'exec'),ns)

def dispatch(bundle):
    base=bundle/'vllm/model_executor/kernels/linear/scaled_mm';calls=[]
    def record(owner):
        def fn(*a,**k):calls.append(owner);return Tensor((1,1))
        return fn
    torch=S(bfloat16='bf16',float32='fp32',float8_e4m3fn='fp8',empty=lambda *a,**k:Tensor((1,1)),ops=S(vllm=S(),_C=S(cutlass_scaled_mm=record('CUTLASS'))))
    skinny={'torch':torch,'MAX_M':64,'_load':lambda:S(mm=record('skinnyFP8'))}
    extract(base/'sm120_skinny_fp8.py',['_pick','_sm120_fp8_scaled_mm'],skinny)
    torch.ops.vllm.sm120_fp8_scaled_mm=skinny['_sm120_fp8_scaled_mm']
    dg={'torch':torch,'VERIFIED_M':frozenset((32,33,48,63,64,65,80,96,112,128)),'_READY':{0:({m:None for m in MS},None)},'_DG':S(_C=S(fp8_rowwise_gemm_nt_sm120=record('DG'))),'_audit':lambda *a:None}
    tree=ast.parse((base/'sm120_dg_rowwise_fp8.py').read_text());assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VERIFIED_M' for t in n.targets));exec(compile(ast.Module(body=[assignment],type_ignores=[]),'actual_verified_m','exec'),dg)
    extract(base/'sm120_dg_rowwise_fp8.py',['supported_weight','_eligible','_run'],dg);torch.ops.vllm.sm120_dg_rowwise_fp8_mm=dg['_run']
    ns={'torch':torch,'ops':S(cutlass_scaled_mm=record('CUTLASS'))};extract(base/'cutlass.py',['apply_scaled_mm'],ns)
    rows=[]
    for dg_on,skinny_on in ((False,False),(False,True),(True,False),(True,True)):
      for m in MS:
        a=Tensor((m,2048));b=Tensor((2048,12288),stride=(1,2048));sa=Tensor((m,1),'fp32');sb=Tensor((12288,),'fp32');calls.clear()
        obj=S(use_sm120_dg_rowwise=dg_on,use_sm120_skinny=skinny_on,logical_output_size=12288)
        ns['apply_scaled_mm'](obj,A=a,B=b,As=sa,Bs=sb,out_dtype='bf16',bias=None,output_shape=[m,12288])
        expected='DG' if dg_on and m in dg['VERIFIED_M'] else ('skinnyFP8' if skinny_on and m<=64 else 'CUTLASS')
        assert calls==[expected],(m,dg_on,skinny_on,calls,expected)
        rows.append({'m':m,'dg_flag':dg_on,'skinny_flag':skinny_on,'actual_launches':calls[:]})
    # DG ineligible activation alignment must route exactly once to skinny/CUTLASS.
    a=Tensor((32,2048));a.data_ptr=lambda:1;calls.clear()
    dg['_run'](a,b,Tensor((32,1),'fp32'),sb,True);assert calls==['CUTLASS']
    bf={'torch':torch};extract(bundle/'vllm/model_executor/layers/utils.py',['_use_skinny_bf16_gemm'],bf)
    for m in (1,3,4,16,17,64):
        for n in (64,128,256,512,1024):
            x=Tensor((m,2048),'bf16');w=Tensor((n,2048),'bf16',stride=(2048,1))
            expected=1<=m<=16 and 16<=n<=512 and (m<=3 or n<=128)
            assert bf['_use_skinny_bf16_gemm'](x,w,None)==expected
    # Native plugin contains no diagnostic worker/graph/wrapper instrumentation.
    plugin=(bundle/'plugin/stack_plugin.py').read_text()
    for forbidden in ['compile_or_warm_up_model','run_pw_graph','_ensure_wrapper','nvtx','post_workload','_WORKSPACES']:
        assert forbidden not in plugin,forbidden
    return rows

def runtime_env(bundle):
    import torch
    assert not torch.cuda.is_available(), 'CPU check requires CUDA hidden'
    path=bundle/'vllm/envs.py';spec=importlib.util.spec_from_file_location('stack_actual_envs',path);env=importlib.util.module_from_spec(spec);sys.modules[spec.name]=env;spec.loader.exec_module(env)
    names=['VLLM_SM120_DG_ROWWISE_FP8','VLLM_SM120_SKINNY_FP8','VLLM_SKINNY_BF16_GEMM','VLLM_SM120_TASK_ORDER_LONG_FIRST','VLLM_SM120_STACK_SOURCE_ID','VLLM_SM120_SKINNY_FP8_MAX_M','VLLM_B12X_FIX_SCALES','VLLM_B12X_SHARED_WORKSPACE','VLLM_B12X_SCALAR_INPUT_SCALE','VLLM_SM120_FP8_PREFILL','VLLM_SM120_FP8_PREFILL_NOGLUE','VLLM_FI_KERNEL_PAGES','VLLM_GDN_INTERNAL_CKPT','VLLM_GDN_FUSED_CONV','TORCHINDUCTOR_ENABLE_PDL']
    facts={}
    for name in names:
        os.environ[name]='0';a=env.compile_factors();os.environ[name]='1';b=env.compile_factors();assert name in a and name in b and a[name]!=b[name],name
        facts[name]=[a[name],b[name]];os.environ.pop(name,None)
    import vllm
    sys.modules['vllm.envs']=env;vllm.envs=env
    os.environ['VLLM_SM120_STACK_SOURCE_ID']=json.loads((bundle/'manifest.json').read_text())['source_id']
    sys.path.insert(0,str(bundle/'plugin'))
    import stack_plugin
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dynamic_kernel as factory
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    original=factory.MoEGatedDynamicKernel
    original_key=moe_dispatch._dynamic_kernel_cache_key
    original_static=moe_dispatch._static_kernel_cache_key
    original_micro=moe_dispatch._micro_kernel_cache_key
    kwargs=dict(activation_precision='fp4',quant_mode='nvfp4',E=256,k=2048,n=512,num_topk=8,mac=4,mma_tiler_mn=(128,128),topk_ids_dtype=torch.int64,input_scales_are_reciprocal=True,fast_math=False,activation='silu',swiglu_alpha=1.0,swiglu_beta=0.0,swiglu_limit=None,share_input_across_experts=True)
    base_key=original_key(**kwargs);keys=[];disks=[]
    for flag,profile in [('0','parent'),('1','combined')]:
        factory.MoEGatedDynamicKernel=original;moe_dispatch._dynamic_kernel_cache_key=original_key;stack_plugin._REGISTERED=False
        os.environ['VLLM_SM120_TASK_ORDER_LONG_FIRST']=flag;os.environ['STACK_PROFILE']=profile
        stack_plugin.register()
        key=moe_dispatch._dynamic_kernel_cache_key(**kwargs)
        assert key[:-2]==base_key and key[-2]=='task_order_source_sha256'
        expected=json.loads((bundle/'plugin/native_source_manifest.json').read_text())['candidate_sha256' if flag=='1' else 'baseline_sha256']
        assert key[-1]==expected
        keys.append(key);disks.append(moe_dispatch._disk_kernel_name('dynamic_e256_k2048_n512_t8',key))
        assert moe_dispatch._static_kernel_cache_key is original_static and moe_dispatch._micro_kernel_cache_key is original_micro
        if flag=='0':assert factory.MoEGatedDynamicKernel is original
        else:assert factory.MoEGatedDynamicKernel.__name__=='MoEGatedDynamicKernelLongFirst' and isinstance(factory.MoEGatedDynamicKernel,type)
    assert keys[0]!=keys[1] and disks[0]!=disks[1]
    # Existing native cache must fail closed, not silently load the old artifact.
    factory.MoEGatedDynamicKernel=original;moe_dispatch._dynamic_kernel_cache_key=original_key;stack_plugin._REGISTERED=False
    moe_dispatch._DYNAMIC_KERNEL_CACHE[('cpu_fixture',)]=object()
    try:stack_plugin.register()
    except AssertionError as e:assert 'already populated' in str(e)
    else:raise AssertionError('Populated cache accepted')
    finally:moe_dispatch._DYNAMIC_KERNEL_CACHE.clear()
    os.environ['STACK_PROFILE']='parent'
    try:stack_plugin.register()
    except AssertionError as e:assert 'mismatch' in str(e)
    else:raise AssertionError('Mismatched profile accepted')
    assert not torch.cuda.is_initialized()
    return {'cuda_available':False,'cuda_initialized':False,'actual_compile_factor_toggle_checks':facts,'native_class_off':original.__name__,'native_class_on':'MoEGatedDynamicKernelLongFirst','disk_keys_off_on':disks,'native_cache_preserves_original_tuple':True,'negative_checks':['populated native cache rejected','profile mismatch rejected']}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);p.add_argument('--runtime-env',action='store_true');a=p.parse_args()
    result=runtime_env(a.bundle) if a.runtime_env else {'dispatch':dispatch(a.bundle),'scope':'actual Python dispatch with mocked GPU launches; not tensor correctness or performance'}
    print(json.dumps(result,indent=2))
