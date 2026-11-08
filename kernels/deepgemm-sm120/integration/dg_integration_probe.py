"""Matched source/audit instrumentation in both DG controls; live graph shape NVTX."""
import functools
import hashlib
import json
import os
from pathlib import Path

_LIVE=False
_REPLAYS={}


def register():
    if not os.environ.get('DG_RUNTIME_OUT'):return
    # Pin the same launch policy as actual-operand micro runs in both profiles.
    # This boundary runs before model load and any candidate JIT/graph capture.
    from fixed_pdl_policy import install_fixed_pdl
    install_fixed_pdl()
    from vllm.v1.worker.gpu_worker import Worker
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
    import torch
    original_graph=ModelCudaGraphManager.run_fullgraph
    @functools.wraps(original_graph)
    def graph_call(manager,descriptor,*args,**kwargs):
        m=int(descriptor.num_tokens) if _LIVE else None
        if m is not None:torch.cuda.nvtx.range_push('DG_FULL_GRAPH_REPLAY_M'+str(m))
        try:result=original_graph(manager,descriptor,*args,**kwargs)
        finally:
            if m is not None:torch.cuda.nvtx.range_pop()
        if m is not None:_REPLAYS[m]=_REPLAYS.get(m,0)+1
        return result
    ModelCudaGraphManager.run_fullgraph=graph_call
    original=Worker.compile_or_warm_up_model
    def report(worker,filename):
        from runtime_audit import audit_model
        from vllm import envs
        assert not envs.VLLM_SM120_SKINNY_FP8 and not envs.VLLM_SKINNY_BF16_GEMM
        runner=worker.model_runner
        assert type(runner).__module__=='vllm.v1.worker.gpu.model_runner'
        cfg=runner.vllm_config.speculative_config
        assert cfg.method=='mtp' and cfg.num_speculative_tokens==2 and cfg.num_speculative_tokens_per_batch_size is None
        record=audit_model(runner.model)
        assert record['flag']==(os.environ['DG_PROFILE']=='on')
        composition=json.loads((Path(__file__).parent/'generated/composition.json').read_text())
        for name,expected in composition['generated_sha256'].items():
            path=Path(envs.__file__).parent/name.removeprefix('vllm/')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
        factors=envs.compile_factors();assert 'VLLM_SM120_DG_ROWWISE_FP8' in factors
        record.update(profile=os.environ['DG_PROFILE'],source_verified=True,
                      compile_factor_sha256=hashlib.sha256(repr(sorted(factors.items())).encode()).hexdigest(),
                      live_full_graph_replays={str(k):v for k,v in sorted(_REPLAYS.items())})
        out=Path(os.environ['DG_RUNTIME_OUT']);out.mkdir(parents=True,exist_ok=True)
        (out/filename).write_text(json.dumps(record,indent=2)+'\n')
        return record
    @functools.wraps(original)
    def warmup(worker,*args,**kwargs):
        global _LIVE
        result=original(worker,*args,**kwargs)
        report(worker,'startup.json');_LIVE=True
        return result
    Worker.compile_or_warm_up_model=warmup
    def post_workload(worker):
        # Explicit RPC while the worker is alive; this is NOT a shutdown hook.
        assert _LIVE, 'Post-workload audit requires completed worker warmup'
        import qwen_fi_control_plugin as fi
        assert fi._owner is not None
        destination=Path(os.environ['DG_RUNTIME_OUT'])/'post_workload.json'
        assert not destination.exists(), 'Refuse to overwrite post-workload evidence'
        fi._owner.finalize(stage='server_post_workload')
        fi._owner.report['server_post_workload_boundary']='explicit_collective_rpc_after_tasks'
        fi._owner.write_report()
        record=report(worker,'post_workload.json')
        return {'stage':'server_post_workload','pid':os.getpid(),
                'enabled_qkvz_modules':record['enabled_qkvz_modules'],
                'deepgemm_pdl':record['deepgemm_pdl']}
    Worker.dg_post_workload_audit=post_workload
    original_shutdown=Worker.shutdown
    @functools.wraps(original_shutdown)
    def shutdown(worker,*args,**kwargs):
        report(worker,'shutdown.json')
        return original_shutdown(worker,*args,**kwargs)
    Worker.shutdown=shutdown
