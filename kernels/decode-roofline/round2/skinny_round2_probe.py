"""Startup-only source/config eligibility, not proof of captured kernel activity."""
import functools
import hashlib
import json
import os
from pathlib import Path


def register():
    if not os.environ.get('SKINNY_ROUND2_OUT'): return
    from vllm.v1.worker.gpu_worker import Worker
    original=Worker.compile_or_warm_up_model
    @functools.wraps(original)
    def warmup(worker,*args,**kwargs):
        result=original(worker,*args,**kwargs)
        from vllm import envs
        from vllm.model_executor.layers import utils
        from vllm.model_executor.kernels.linear.scaled_mm import sm120_skinny_fp8
        runner=worker.model_runner
        assert type(runner).__module__=='vllm.v1.worker.gpu.model_runner'
        cfg=runner.vllm_config.speculative_config
        assert cfg.method=='mtp' and cfg.num_speculative_tokens==2 and cfg.num_speculative_tokens_per_batch_size is None
        bf16=bool(envs.VLLM_SKINNY_BF16_GEMM);fp8=bool(envs.VLLM_SM120_SKINNY_FP8)
        assert (bf16,fp8)=={'off':(False,False),'bf16':(True,False),'fp8':(False,True),'both':(True,True)}[os.environ['SKINNY_PROFILE']]
        assert utils._SKINNY_BF16_GEMM==bf16 and sm120_skinny_fp8.ENABLED==fp8
        selected=utils.dispatch_unquantized_gemm().__name__
        assert selected==('skinny_unquantized_gemm' if bf16 else 'default_unquantized_gemm')
        factors=envs.compile_factors()
        for key in ('VLLM_SKINNY_BF16_GEMM','VLLM_SM120_SKINNY_FP8','VLLM_SM120_SKINNY_FP8_MAX_M'):assert key in factors
        composition=json.loads((Path(__file__).parent/'generated/composition.json').read_text())
        package=Path(envs.__file__).parent
        for name,expected in composition['generated_sha256'].items():
            path=package/name.removeprefix('vllm/')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
        out=Path(os.environ['SKINNY_ROUND2_OUT']);out.mkdir(parents=True,exist_ok=True)
        (out/'runtime.json').write_text(json.dumps(dict(profile=os.environ['SKINNY_PROFILE'],
            bf16=bf16,fp8=fp8,max_m=sm120_skinny_fp8.MAX_M,bf16_dispatch=selected,
            compile_factor_sha256=hashlib.sha256(repr(sorted(factors.items())).encode()).hexdigest(),
            source_verified=True,compile_cache_root=envs.VLLM_CACHE_ROOT),indent=2))
        return result
    Worker.compile_or_warm_up_model=warmup
