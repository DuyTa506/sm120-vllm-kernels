"""Startup-only actual runtime guard; no added scheduler observations."""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
from composition_guards import verify_runtime_sources

_registered=False


def register():
    global _registered
    if _registered or not os.environ.get('GDN_ROUND2_OUT'):
        return
    from vllm.v1.worker.gpu_worker import Worker
    original=Worker.compile_or_warm_up_model
    @functools.wraps(original)
    def warmup(worker,*args,**kwargs):
        from vllm.model_executor.layers.mamba.gdn import gdn_state_replay as replay
        # Capture invokes model-only dummy forwards without acceptance/commit.
        # Their retained per-layer snapshots are not runtime correctness evidence.
        from shadow_startup import suspend_startup_shadow
        with suspend_startup_shadow(replay):
            result=original(worker,*args,**kwargs)
        import torch
        from vllm import envs
        from vllm.model_executor.layers.mamba.gdn import gdn_state_replay as replay
        runner=worker.model_runner
        assert type(runner).__module__=='vllm.v1.worker.gpu.model_runner', 'baseline V2 runner required'
        cfg=runner.vllm_config
        assert cfg.speculative_config.method=='mtp' and cfg.speculative_config.num_speculative_tokens==2
        assert cfg.speculative_config.num_speculative_tokens_per_batch_size is None, 'dynamic MTP not validated'
        assert cfg.parallel_config.pipeline_parallel_size==1
        assert envs.VLLM_GDN_DECODE_KERNEL=='cuda'
        enabled=bool(envs.VLLM_GDN_STATE_REPLAY)
        assert enabled==(os.environ['GDN_ROUND2_MODE']!='off')
        registry=replay._REGISTRY
        layer_count=0
        if enabled:
            assert registry is not None and registry.layers, 'replay never registered'
            assert registry.lib is not None, 'native replay library not prepared'
            for layer in registry.layers:
                assert layer.kv_cache[1].dtype==torch.float32
            layer_count=len(registry.layers)
            assert layer_count==30, f"expected all30 GDN state layers, got {layer_count}"
        from vllm.v1.worker.gpu.model_states import mamba_hybrid
        sources=[Path(inspect.getsourcefile(type(runner))),Path(inspect.getsourcefile(type(runner.speculator))),
                 Path(replay.__file__),Path(mamba_hybrid.__file__),Path(envs.__file__)]
        identity={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        verified_sources=verify_runtime_sources(Path(__file__).parent/'generated/composition.json',
                                               os.environ['GDN_ROUND2_COMPOSITION_SHA'], Path(envs.__file__).parent)
        record=dict(runner=f'{type(runner).__module__}.{type(runner).__qualname__}',
                    proposer=f'{type(runner.speculator).__module__}.{type(runner.speculator).__qualname__}',
                    mtp_depth=2,enabled=enabled,fp32_layers=layer_count,source_sha256=identity,
                    startup_shadow_suspended=True, pending_startup_shadow_records=0,
                    composition_sources_verified=True,composition_source_sha256=verified_sources,
                    composition_sha256=os.environ['GDN_ROUND2_COMPOSITION_SHA'])
        out=Path(os.environ['GDN_ROUND2_OUT']);out.mkdir(parents=True,exist_ok=True)
        from cache_geometry import describe, validate_preempt_geometry, encode_spec_field
        geometry=describe(runner)
        (out/'cache_geometry.json').write_text(json.dumps(geometry,indent=2,default=encode_spec_field))
        if os.environ['GDN_ROUND2_MODE']=='preempt':
            validate_preempt_geometry(geometry)
        (out/'runtime.json').write_text(json.dumps(record,indent=2))
        return result
    Worker.compile_or_warm_up_model=warmup
    _registered=True
