"""Direct CPU regression of actual preparation code under vLLM's BF16 default."""
import contextlib,importlib.util,sys,types
from pathlib import Path
from unittest.mock import patch
import torch

# Keep the preparation function real; substitute only external runtime/GPU boundaries.
vllm=types.ModuleType('vllm');vllm.envs=types.SimpleNamespace(VLLM_SM120_DG_ROWWISE_FP8=True)
logger=types.ModuleType('vllm.logger');logger.init_logger=lambda _:types.SimpleNamespace(info=lambda *a:None)
utils=types.ModuleType('vllm.utils.torch_utils');utils.direct_register_custom_op=lambda **kw:None
modules={'vllm':vllm,'vllm.logger':logger,'vllm.utils.torch_utils':utils}
packed=[];calls=[];allocations=[]
def pack(value):
    assert value.dtype==torch.float32,'unit metadata must not inherit BF16 default'
    packed.append(tuple(value.shape));return value

def native(a,b,out,sa,sb):
    assert a[0].dtype==b[0].dtype==torch.float8_e4m3fn
    assert out.dtype==torch.bfloat16 and sa.dtype==sb.dtype==torch.float32
    calls.append(a[0].shape[0])
dg=types.SimpleNamespace(get_pdl=lambda:False,get_mn_major_tma_aligned_packed_ue8m0_tensor=pack,_C=types.SimpleNamespace(fp8_rowwise_gemm_nt_sm120=native))
modules['deep_gemm']=dg
original_default=torch.get_default_dtype()
try:
    with patch.dict(sys.modules,modules):
        spec=importlib.util.spec_from_file_location('dtype_preparation_test',Path(__file__).with_name('sm120_dg_rowwise_fp8.py'))
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        weight=torch.empty((12288,2048),dtype=torch.float8_e4m3fn).T
        scale=torch.ones((12288,1),dtype=torch.float32)
        factories={name:getattr(torch,name) for name in ('ones','zeros','empty')}
        def factory(name):
            def allocate(*a,**kw):
                assert 'dtype' in kw, 'implicit allocation dtype in preparation: '+name
                allocations.append((name,kw['dtype']));kw['device']='cpu'
                return factories[name](*a,**kw)
            return allocate
        torch.set_default_dtype(torch.bfloat16)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(m,'supported_weight',lambda *a:True))
            stack.enter_context(patch.object(torch.cuda,'get_device_capability',lambda *a:(12,0)))
            stack.enter_context(patch.object(torch.cuda,'is_current_stream_capturing',lambda:False))
            stack.enter_context(patch.object(torch.cuda,'device',lambda *a:contextlib.nullcontext()))
            stack.enter_context(patch.object(torch.cuda,'synchronize',lambda *a:None))
            for name in factories:stack.enter_context(patch.object(torch,name,factory(name)))
            assert m.prepare_for_weight(weight,scale)
            assert m.prepare_for_weight(weight,scale) # Cached preparation cannot allocate again.
        assert sorted(calls)==sorted(m.VERIFIED_M) and len(packed)==len(m.VERIFIED_M)+1
        assert len(allocations)==1+4*len(m.VERIFIED_M)
        assert not torch.cuda.is_initialized()
finally:torch.set_default_dtype(original_default)
print('actual preparation path passes BF16 default with explicit metadata/activation/output dtypes; no CUDA initialized')
