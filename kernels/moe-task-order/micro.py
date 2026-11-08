"""Isolated native full-call screen; explicit opt-in class replacement only."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'moe-nvfp4'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=['baseline','long_first'],default='baseline')
    p.add_argument('--layers',type=int,nargs='+',default=[19])
    p.add_argument('--tokens',type=int,nargs='+',default=[1024,2048])
    p.add_argument('--patterns',nargs='+',choices=['real','uniform','eight_hot'],default=['real','uniform','eight_hot'])
    p.add_argument('--repeats',type=int,default=8)
    p.add_argument('--geometry-only',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    assert not args.geometry_only or args.variant=='baseline', 'geometry scan is baseline-only'
    assert args.repeats>=4 and all(1024<=m<=4096 for m in args.tokens)
    args.output.mkdir(parents=True,exist_ok=False)
    for key,child in [('CUTE_DSL_CACHE_DIR','cute'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton'),('FLASHINFER_WORKSPACE_BASE','flashinfer')]:
        os.environ[key]=str(args.output/'compile_cache'/child)
    for k,v in dict(VLLM_B12X_FIX_SCALES='1',VLLM_B12X_SHARED_WORKSPACE='1',VLLM_B12X_SCALAR_INPUT_SCALE='1',VLLM_B12X_HYBRID_MIN_TOKENS='1024',VLLM_B12X_ACT_SCALES='static').items():os.environ[k]=v
    sys.path.insert(0,str(BASE))
    import torch
    from moe_common import H,load_layer_raw,load_hidden,prep_cutlass,route
    from test_shared_workspace import make_expert,wrapper
    from verify_scalar_dispatch import prepare_exact
    from verify_shared_workspace import load_backend,apply,capture,errors
    from vllm.v1.worker import workspace as ws
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dynamic_kernel as factory
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    from order_model import descriptors
    manifest=json.loads((HERE/'source_manifest.json').read_text())
    installed=Path(sys.modules[factory.MoEGatedDynamicKernel.__module__].__file__)
    assert hashlib.sha256(installed.read_bytes()).hexdigest()==manifest['baseline_sha256'],'installed baseline changed'
    assert moe_dispatch._FORCED_BACKEND is None
    if args.variant=='long_first':
        path=HERE/'gated_long_first.py'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['candidate_sha256']
        qualified_name=factory.MoEGatedDynamicKernel.__module__.rsplit('.',1)[0]+'.gated_long_first_experiment'
        spec=importlib.util.spec_from_file_location(qualified_name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        factory.MoEGatedDynamicKernel=module.MoEGatedDynamicKernelLongFirst
    expected_class=factory.MoEGatedDynamicKernel
    constructed=[]
    # Observe constructor identity while leaving the real implementation intact.
    def checked_constructor(*a,**kw):
        instance=expected_class(*a,**kw);constructed.append(type(instance).__name__);return instance
    factory.MoEGatedDynamicKernel=checked_constructor
    ws.init_workspace_manager(torch.device('cuda:0'))
    provenance=dict(variant=args.variant,source_manifest=manifest,installed_source=str(installed),
                    runtime_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [installed,Path(factory.__file__),Path(moe_dispatch.__file__)]},
                    cache_env={key:os.environ[key] for key in ('CUTE_DSL_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','FLASHINFER_WORKSPACE_BASE')},
                    scalar_source_sha256=hashlib.sha256((BASE/'scalar/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py').read_bytes()).hexdigest(),
                    harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    for layer in args.layers:
        raw=load_layer_raw(layer);prep=prep_cutlass(raw)
        expert=prepare_exact(load_backend('scalar'),raw,prep,make_expert)
        assert expert._input_global_scale.numel()==1
        hidden=load_hidden(layer);assert hidden is not None
        for tokens in args.tokens:
            x=hidden[:tokens].contiguous()
            real_weights,real_ids=route(x,raw['gate'])
            for pattern in args.patterns:
                ids=real_ids.clone();weights=real_weights.clone()
                if pattern=='uniform':
                    ids.copy_((torch.arange(tokens*8,device='cuda').reshape(tokens,8)%256).int());weights.fill_(1/8)
                elif pattern=='eight_hot':
                    ids.copy_(torch.arange(8,device='cuda',dtype=torch.int32).expand_as(ids));weights.fill_(1/8)
                output=torch.empty((tokens,H),device='cuda',dtype=torch.bfloat16)
                call=lambda:apply(expert,prep,x,ids,weights,output)
                graph=capture(call)
                assert constructed and all(n==expected_class.__name__ for n in constructed)
                scratch=wrapper(expert)._dynamic_workspace
                torch.cuda.synchronize()
                # Post-call inspection only: no forward host read or added kernel.
                count=int(scratch.task_tail.item());total=int(scratch.expert_tile_base[256].item())
                gate_tiles=512//128;groups=(gate_tiles+1)//2
                split=(count-total)//(groups-1)
                counts=scratch.row_counts.cpu().tolist()
                tiles=[];valid=[]
                for e,rows in enumerate(counts[:256]):
                    for begin in range(0,rows,scratch.tile_m):tiles.append(e);valid.append(min(scratch.tile_m,rows-begin))
                assert len(tiles)==total
                expected=descriptors(tiles,valid,gate_tiles,split,args.variant=='long_first')
                actual=list(zip(scratch.task_expert[:count].cpu().tolist(),scratch.task_valid_rows[:count].cpu().tolist()))
                assert actual==expected,'published descriptor order/payload differs from CPU model'
                applicability=dict(layer=layer,tokens=tokens,pattern=pattern,T=total,S=split,G=groups,task_count=count,
                                   long_tasks=total-split,short_tasks=split*groups,
                                   full_slice_count=gate_tiles,split_slice_count=2,tile_m=scratch.tile_m,row_counts=counts,
                                   caveat='slice counts describe work, not measured per-task duration')
                (args.output/f'applicability_l{layer}_m{tokens}_{pattern}.json').write_text(json.dumps(applicability,indent=2))
                if args.geometry_only:
                    print(json.dumps(applicability),flush=True)
                    del graph,output
                    continue
                if not 0<split<total:
                    raise RuntimeError('no meaningful queue reorder for this input; applicability evidence preserved')
                values=[];repeat_errors=[]
                for _ in range(args.repeats):
                    graph.replay();torch.cuda.synchronize();values.append(output.clone())
                for value in values[1:]:repeat_errors.append(errors(value,values[0]))
                assert all(r["finite"] for r in repeat_errors), "nonfinite repeated output"
                times=[]
                for _ in range(40):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();graph.replay();end.record();end.synchronize();times.append(start.elapsed_time(end)*1000)
                stem=f'l{layer}_m{tokens}_{pattern}'
                torch.save(dict(outputs=[v.cpu() for v in values],input=x.cpu(),ids=ids.cpu(),weights=weights.cpu()),args.output/f'{stem}.pt')
                record=dict(layer=layer,tokens=tokens,pattern=pattern,variant=args.variant,total_tiles=total,split_tiles=split,split_groups=groups,task_count=count,
                            permutation_nontrivial=0<split<total,median_us=statistics.median(times),samples_us=times,baseline_or_candidate_self_repeat=repeat_errors,
                            descriptor_multiset_sha256=hashlib.sha256(json.dumps(sorted(actual)).encode()).hexdigest())
                with (args.output/'measurements.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps({k:v for k,v in record.items() if k not in ('samples_us','baseline_or_candidate_self_repeat')}),flush=True)
                del graph,values,output
    if not args.geometry_only:
        assert all(r['finite'] for r in repeat_errors)

if __name__=='__main__':main()
