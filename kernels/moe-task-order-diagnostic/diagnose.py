"""Single diagnostic variant. No timer; never used for performance conclusions."""
import argparse, hashlib, importlib.util, json, os, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['baseline','long_first'],required=True);p.add_argument('--columns',type=int,choices=[128,2048],default=128);p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    for key in ['CUTE_DSL_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','FLASHINFER_WORKSPACE_BASE']:os.environ[key]=str(args.output/key)
    for k,v in dict(VLLM_B12X_FIX_SCALES='1',VLLM_B12X_SHARED_WORKSPACE='1',VLLM_B12X_SCALAR_INPUT_SCALE='1',VLLM_B12X_HYBRID_MIN_TOKENS='1024',VLLM_B12X_ACT_SCALES='static').items():os.environ[k]=v
    sys.path.insert(0,str(HERE.parent/'moe-nvfp4'))
    import torch
    from moe_common import load_layer_raw,prep_cutlass,snap
    from test_shared_workspace import make_expert,wrapper
    from verify_scalar_dispatch import prepare_exact
    from verify_shared_workspace import load_backend,apply
    from vllm.v1.worker import workspace as ws
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dynamic_kernel as factory
    capture=torch.load(args.capture,map_location='cpu',weights_only=True)
    x,ids,weights=[capture[k].cuda() for k in ('input','ids','weights')]
    assert tuple(x.shape)==(4096,2048) and tuple(ids.shape)==(4096,8)
    counts=torch.bincount(ids.flatten().long(),minlength=256).cpu();tiles=sum((int(c)+127)//128 for c in counts)
    assert tiles==400
    payload=torch.full((tiles*2,128,args.columns),float('nan'),dtype=torch.bfloat16,device='cuda')
    metadata=torch.full((tiles*2,128,4),-1,dtype=torch.int32,device='cuda')
    path=HERE/f'gated_{args.variant}_diagnostic.py'
    manifest=json.loads((HERE/'source_manifest.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest[args.variant]['diagnostic_sha256']
    name=factory.MoEGatedDynamicKernel.__module__.rsplit('.',1)[0]+'.isolated_scatter_diagnostic'
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    cls=getattr(module,'MoEGatedDynamicKernel' if args.variant=='baseline' else 'MoEGatedDynamicKernelLongFirst')
    constructed=[]
    def construct(*a,**kw):
        obj=cls(*a,**kw);assert tuple(obj.tile_shape_mnk[:2])==(128,128);obj.diag_columns=args.columns;obj.diag_payload_ptr=payload.data_ptr();obj.diag_meta_ptr=metadata.data_ptr();constructed.append(obj);return obj
    factory.MoEGatedDynamicKernel=construct
    ws.init_workspace_manager(torch.device('cuda:0'))
    raw=load_layer_raw(19)
    assert tuple(raw['w13'].shape)==(256,1024,1024)
    assert tuple(raw['w2'].shape)==(256,2048,256)
    assert raw['w13'].shape[1]//2//128==4 and (4+1)//2==2
    prep=prep_cutlass(raw);expert=prepare_exact(load_backend('scalar'),raw,prep,make_expert)
    output=torch.empty_like(x)
    # Warm/compile once, then reset all logging sentinels for the observed call.
    apply(expert,prep,x,ids,weights,output);torch.cuda.synchronize()
    assert constructed
    scratch=wrapper(expert)._dynamic_workspace
    assert int(scratch.task_tail)==735 and int(scratch.expert_tile_base[256])==400
    assert int(scratch.task_tail)-int(scratch.expert_tile_base[256])==335
    assert scratch.tile_m==128
    for repeat in range(2):
        payload.fill_(float('nan'));metadata.fill_(-1)
        apply(expert,prep,x,ids,weights,output);torch.cuda.synchronize()
        assert int(scratch.task_tail)==735 and int(scratch.expert_tile_base[256])==400
        result=dict(payload=payload.cpu(),metadata=metadata.cpu(),output=output[:,:args.columns].cpu(),ids=ids.cpu(),weights=weights.cpu(),counts=counts,split=335,tiles=400,variant=args.variant,repeat=repeat,columns=args.columns)
        torch.save(result,args.output/('diagnostic.pt' if repeat==0 else 'diagnostic_repeat1.pt'))
    checkpoint=Path(snap())
    checkpoint_identity={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [checkpoint/'config.json',checkpoint/'model.safetensors.index.json']}
    (args.output/'provenance.json').write_text(json.dumps(dict(source_manifest=manifest,capture=str(args.capture),capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),checkpoint_identity=checkpoint_identity,checkpoint_snapshot=str(checkpoint),repeats=2,columns=args.columns,scratch_bytes=payload.numel()*2+metadata.numel()*4,compiled_objects=len(constructed),timing_claim=False),indent=2))
if __name__=='__main__':main()
