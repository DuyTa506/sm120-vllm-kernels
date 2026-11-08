"""Native scheduler class-only opt-in; post-workload queue readback is outside timing."""
import functools,hashlib,importlib.util,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
_WORKSPACES={};_CONSTRUCTED=[];_LIVE=False;_REGISTERED=False;_PW_REPLAYS={}

def register():
    global _REGISTERED
    if _REGISTERED or not os.environ.get('TASK_ORDER_OUT'):return
    _REGISTERED=True
    from vllm import envs
    profile=os.environ['TASK_ORDER_PROFILE'];assert profile in ('off','on')
    assert envs.VLLM_SM120_TASK_ORDER_LONG_FIRST == (profile=='on')
    assert 'VLLM_SM120_TASK_ORDER_LONG_FIRST' in envs.compile_factors()
    import torch
    from vllm.v1.worker.gpu_worker import Worker
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dynamic_kernel as factory
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    assert moe_dispatch._FORCED_BACKEND is None, 'Unexpected forced native backend'
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe import FlashInferB12xExperts
    manifest=json.loads((HERE/'native_source_manifest.json').read_text())
    original_source=Path(sys.modules[factory.MoEGatedDynamicKernel.__module__].__file__)
    assert hashlib.sha256(original_source.read_bytes()).hexdigest()==manifest['baseline_sha256']
    candidate=HERE/'gated_long_first.py'
    assert hashlib.sha256(candidate.read_bytes()).hexdigest()==manifest['candidate_sha256']
    # Native on-disk keys otherwise ignore the implementation selected by factory.
    assert not moe_dispatch._DYNAMIC_KERNEL_CACHE, 'Native compiler cache already populated before policy registration'
    effective_sha=manifest['candidate_sha256' if profile=='on' else 'baseline_sha256']
    original_dynamic_key=moe_dispatch._dynamic_kernel_cache_key
    def implementation_key(*args,**kwargs):
        base=original_dynamic_key(*args,**kwargs)
        assert isinstance(base,tuple)
        return base+('task_order_source_sha256',effective_sha)
    moe_dispatch._dynamic_kernel_cache_key=implementation_key
    implementation=factory.MoEGatedDynamicKernel
    if profile=='on':
        name=implementation.__module__.rsplit('.',1)[0]+'.gated_long_first_serving'
        spec=importlib.util.spec_from_file_location(name,candidate);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
        implementation=m.MoEGatedDynamicKernelLongFirst
    def construct(*args,**kwargs):
        obj=implementation(*args,**kwargs);_CONSTRUCTED.append(type(obj).__module__+'.'+type(obj).__name__);return obj
    factory.MoEGatedDynamicKernel=construct
    original_wrapper=FlashInferB12xExperts._ensure_wrapper
    @functools.wraps(original_wrapper)
    def wrapper(instance):
        result=original_wrapper(instance)
        scratch=result._dynamic_workspace
        _WORKSPACES[id(scratch)]=scratch # References only: no GPU reads/copies in inference.
        return result
    FlashInferB12xExperts._ensure_wrapper=wrapper
    def audit(worker,stage,queues=False):
        from vllm import envs
        runner=worker.model_runner
        assert type(runner).__module__=='vllm.v1.worker.gpu.model_runner'
        cfg=runner.vllm_config.speculative_config
        assert cfg.method=='mtp' and cfg.num_speculative_tokens==2 and cfg.num_speculative_tokens_per_batch_size is None
        assert _CONSTRUCTED and all(x.endswith(implementation.__name__) for x in _CONSTRUCTED)
        manager=runner.cudagraph_manager
        descriptors_by_mode={mode.name:[int(d.num_tokens) for d in descs] for mode,descs in manager._capture_descs.items()}
        assert 4096 in descriptors_by_mode.get('PIECEWISE',[]), '4096 prefill is absent from actual worker capture plan'
        checkpoint=Path(runner.vllm_config.model_config.model)
        checkpoint_files=[checkpoint/'config.json',checkpoint/'model.safetensors.index.json']
        assert all(p.is_file() for p in checkpoint_files)
        record=dict(native_cache_source_sha256=effective_sha,registered_long_first=envs.VLLM_SM120_TASK_ORDER_LONG_FIRST,worker_capture_plan=descriptors_by_mode,capture_proof='Plan is not proof; require actual live4096 graph-node cudaGraphLaunch trace gate',live_piecewise_calls={str(k):v for k,v in sorted(_PW_REPLAYS.items())},checkpoint_snapshot=str(checkpoint),checkpoint_identity={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint_files},stage=stage,profile=profile,native_source_manifest=manifest,constructed_classes=_CONSTRUCTED,
                    compile_factors_sha256=hashlib.sha256(repr(sorted(envs.compile_factors().items())).encode()).hexdigest(),
                    active_scalar_source_sha256=hashlib.sha256(Path(sys.modules[FlashInferB12xExperts.__module__].__file__).read_bytes()).hexdigest(),queues=[])
        if queues:
            from order_model import descriptors
            torch.cuda.synchronize()
            seen=set()
            for scratch in _WORKSPACES.values():
                ptr=scratch.task_tail.data_ptr()
                if ptr in seen:continue
                seen.add(ptr)
                counts=scratch.row_counts.cpu().tolist()[:256]
                total=int(scratch.expert_tile_base[256]);tasks=int(scratch.task_tail);split=tasks-total
                # Shared workspace retains the last native call. Ignore other-size owners.
                if sum(counts)!=4096*8:continue
                assert scratch.tile_m==128 and 0<=split<=total
                experts=[];valid=[]
                for e,n in enumerate(counts):
                    for begin in range(0,n,128):experts.append(e);valid.append(min(128,n-begin))
                assert len(experts)==total
                actual=list(zip(scratch.task_expert[:tasks].cpu().tolist(),scratch.task_valid_rows[:tasks].cpu().tolist()))
                expected=descriptors(experts,valid,4,split,profile=='on')
                assert actual==expected,'actual published task queue differs from declared native schedule'
                record['queues'].append(dict(rows=sum(counts),T=total,S=split,tasks=tasks,nontrivial=0<split<total,
                                             row_counts=counts,descriptor_sha256=hashlib.sha256(json.dumps(actual).encode()).hexdigest()))
            assert any(q['nontrivial'] for q in record['queues']),'No actual4096 nonidentity native queue observed'
        out=Path(os.environ['TASK_ORDER_OUT']);out.mkdir(parents=True,exist_ok=True)
        path=out/(stage+'.json');assert not path.exists();path.write_text(json.dumps(record,indent=2)+'\n')
        return record
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
    original_pw=ModelCudaGraphManager.run_pw_graph
    @functools.wraps(original_pw)
    def piecewise(manager,model,model_inputs):
        inputs=model_inputs.get('input_ids')
        if inputs is None:inputs=model_inputs.get('inputs_embeds')
        tokens=int(inputs.shape[0]) if _LIVE and inputs is not None else None
        if tokens is not None:torch.cuda.nvtx.range_push('TASK_ORDER_LIVE_PIECEWISE_M'+str(tokens))
        try:return original_pw(manager,model,model_inputs)
        finally:
            if tokens is not None:
                torch.cuda.nvtx.range_pop();_PW_REPLAYS[tokens]=_PW_REPLAYS.get(tokens,0)+1
    ModelCudaGraphManager.run_pw_graph=piecewise
    original=Worker.compile_or_warm_up_model
    @functools.wraps(original)
    def warmup(worker,*args,**kwargs):
        global _LIVE
        result=original(worker,*args,**kwargs);audit(worker,'startup');_LIVE=True;return result
    Worker.compile_or_warm_up_model=warmup
    def post_workload(worker):
        assert _LIVE
        record=audit(worker,'post_workload',queues=True)
        import qwen_fi_control_plugin as fi
        assert fi._owner is not None
        fi._owner.finalize(stage='server_post_workload')
        fi._owner.report['server_post_workload_boundary']='explicit_collective_rpc_after_tasks';fi._owner.write_report()
        return dict(stage='server_post_workload',profile=profile,queues=len(record['queues']),shutdown_proven=False)
    Worker.task_order_post_workload_audit=post_workload
