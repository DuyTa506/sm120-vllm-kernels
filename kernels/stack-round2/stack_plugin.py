"""Uninstrumented class-only scheduler composition plus fixed DG launch policy."""
import hashlib, importlib.util, json, os, sys
from pathlib import Path
_REGISTERED=False
def register():
    global _REGISTERED
    if _REGISTERED:return
    from vllm import envs
    assert envs.VLLM_SM120_STACK_SOURCE_ID, 'Missing prepared stack source identity'
    for key in ('VLLM_SM120_TASK_ORDER_LONG_FIRST','VLLM_SM120_STACK_SOURCE_ID','VLLM_SM120_DG_ROWWISE_FP8','VLLM_SM120_SKINNY_FP8','VLLM_SKINNY_BF16_GEMM','VLLM_SM120_SKINNY_FP8_MAX_M'):
        assert key in envs.compile_factors(), f'Missing compile factor {key}'
    from fixed_pdl_policy import install_fixed_pdl
    install_fixed_pdl()
    profile=os.environ['STACK_PROFILE']
    assert profile in ('parent','task_order','dense','combined')
    enabled=envs.VLLM_SM120_TASK_ORDER_LONG_FIRST
    assert enabled == (profile in ('task_order','combined')), 'Profile/native policy mismatch'
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dynamic_kernel as factory
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    assert moe_dispatch._FORCED_BACKEND is None, 'Unexpected forced native backend'
    here=Path(__file__).resolve().parent;manifest=json.loads((here/'native_source_manifest.json').read_text())
    original=factory.MoEGatedDynamicKernel
    source=Path(sys.modules[original.__module__].__file__)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['baseline_sha256']
    candidate=here/'gated_long_first.py'
    assert hashlib.sha256(candidate.read_bytes()).hexdigest()==manifest['candidate_sha256']
    # Same tuple extension as serving b3: protects native memory AND disk artifacts.
    assert not moe_dispatch._DYNAMIC_KERNEL_CACHE, 'Native compiler cache already populated before policy registration'
    effective_sha=manifest['candidate_sha256' if enabled else 'baseline_sha256']
    original_dynamic_key=moe_dispatch._dynamic_kernel_cache_key
    def implementation_key(*args,**kwargs):
        base=original_dynamic_key(*args,**kwargs)
        assert isinstance(base,tuple)
        return base+('task_order_source_sha256',effective_sha)
    moe_dispatch._dynamic_kernel_cache_key=implementation_key
    if enabled:
        name=original.__module__.rsplit('.',1)[0]+'.stack_long_first'
        spec=importlib.util.spec_from_file_location(name,candidate);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        factory.MoEGatedDynamicKernel=module.MoEGatedDynamicKernelLongFirst
    _REGISTERED=True
