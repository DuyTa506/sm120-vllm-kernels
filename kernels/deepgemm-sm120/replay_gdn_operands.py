"""Validate isolated fused epilogue against captured runtime operands; CPU-safe CLI."""
import argparse
import json
import hashlib
from pathlib import Path
import traceback


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-gpu',action='store_true');ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--skinny',action='store_true')
    ap.add_argument('--ms',default='32,33,63,64,65,96,128,192');ap.add_argument('--iters',type=int,default=30)
    args=ap.parse_args()
    if not args.run_gpu: print('CPU preflight only');return
    assert not args.out.exists()
    import torch
    import deep_gemm
    from common import bench_graph,kernel_names
    from vllm import _custom_ops as ops
    torch.backends.cuda.matmul.allow_tf32=False
    skinny = None
    skinny_manifest = None
    def verify_skinny():
        for name, expected in skinny_manifest['files'].items():
            assert hashlib.sha256((Path('/skinny')/name).read_bytes()).hexdigest() == expected, 'Skinny artifact changed'
    if args.skinny:
        import importlib.util
        skinny_manifest=json.loads(Path('/skinny/manifest.json').read_text())
        assert skinny_manifest['module_name']=='sm120_skinny_fp8'
        assert skinny_manifest['compile_flags']==['-O3','-std=c++17','-gencode=arch=compute_120a,code=sm_120a']
        verify_skinny()
        spec=importlib.util.spec_from_file_location('sm120_skinny_fp8','/skinny/build/sm120_skinny_fp8.so')
        skinny=importlib.util.module_from_spec(spec);spec.loader.exec_module(skinny)
    capture=json.loads(Path('/inputs/results.json').read_text());assert capture['status']=='passed'
    result={'status':'running','capture_snapshot':capture['snapshot'],'capture_prompt_sha256':capture['prompt_tokens_sha256'],
            'skinny_comparison':'unavailable: exact current shipped implementation not mounted; no claim against composed dispatch','cases':[]}
    if args.skinny:
        result['skinny_comparison']='exact existing repository candidate with _pick(32..64)=(2,4), CUTLASS fallback above64; not proof of active CONFIGnext dispatch'
        result['skinny_manifest']=skinny_manifest
    def save():args.out.write_text(json.dumps(result,indent=2)+'\n')
    save()
    try:
        for desc in capture['captures']:
            assert hashlib.sha256((Path('/inputs')/desc['file']).read_bytes()).hexdigest() == desc['file_sha256']
            payload=torch.load(Path('/inputs')/desc['file'],weights_only=False,map_location='cpu')
            ts=payload['tensors'];a_full=ts['a'].cuda();b=ts['b'].cuda();sa_full=ts['sa'].cuda();sb=ts['sb'].cuda()
            assert list(b.stride()) == desc['operands']['b']['stride']
            original=ts['output'].cuda()
            fresh=ops.cutlass_scaled_mm(a_full,b,sa_full,sb,torch.bfloat16)
            assert torch.equal(original,fresh), 'Captured baseline differs from fresh same-shape CUTLASS'
            n,k=b.shape[1],b.shape[0];weight=b.T.contiguous()
            copies=max(2,(256*1024**2+n*k-1)//(n*k));ws=[weight]+[weight.clone() for _ in range(copies-1)]
            bs=deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(torch.ones(n,k//128,device='cuda'))
            for m in map(int,args.ms.split(',')):
                assert 2*m <= a_full.shape[0], 'Need two disjoint actual-input slices'
                a=a_full[:m].clone();sa=sa_full[:m].clone()
                ones=deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(torch.ones(m,k//128,device='cuda'))
                out=torch.empty(m,n,dtype=torch.bfloat16,device='cuda');base=torch.empty_like(out)
                existing_out=torch.empty_like(out)
                def existing(i):
                    if m<=64:
                        skinny.mm(existing_out,a,ws[i],sa.reshape(-1),sb.reshape(-1),2,4)
                    else:
                        torch.ops._C.cutlass_scaled_mm(existing_out,a,ws[i].T,sa,sb,None)
                def fused(i):deep_gemm._C.fp8_rowwise_gemm_nt_sm120((a,ones),(ws[i],bs),out,sa,sb)
                def baseline(i):torch.ops._C.cutlass_scaled_mm(base,a,ws[i].T,sa,sb,None)
                fused(0);baseline(0)
                ref=(a[:16].double()@weight[:64].double().T)*sa[:16].reshape(-1,1).double()*sb.reshape(1,-1)[:,:64].double()
                denom=ref.norm().clamp_min(1e-12)
                err_new=((out[:16,:64].double()-ref).norm()/denom).item()
                err_base=((base[:16,:64].double()-ref).norm()/denom).item()
                rel=((out.float()-base.float()).norm()/base.float().norm().clamp_min(1e-12)).item()
                row={'layer':desc['layer'],'module':desc['module'],'m':m,'n':n,'k':k,'captured_baseline_exact':True,
                     'relative_l2':rel,'bit_mismatches':int((out!=base).sum()),'sample_fp64_fused_l2':err_new,'sample_fp64_cutlass_l2':err_base}
                if args.skinny:
                    existing(0)
                    row['existing_candidate']='skinny_2_4' if m<=64 else 'cutlass_fallback'
                    row['existing_vs_cutlass_relative_l2']=((existing_out.float()-base.float()).norm()/base.float().norm().clamp_min(1e-12)).item()
                    row['existing_vs_fused_bit_mismatches']=int((existing_out!=out).sum())
                    row['sample_fp64_existing_l2']=((existing_out[:16,:64].double()-ref).norm()/denom).item()
                    assert torch.isfinite(existing_out).all() and row['existing_vs_cutlass_relative_l2']<=0.001
                result['cases'].append(row);save()
                assert torch.isfinite(out).all() and torch.isfinite(base).all() and rel <= 0.001
                assert err_new <= max(1e-6,err_base*1.01), 'Sample reference error exceeds baseline screening margin'
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):fused(0)
                a.copy_(a_full[m:2*m]);sa.copy_(sa_full[m:2*m]);graph.replay();replay=out.clone();fused(0)
                assert torch.equal(replay,out),'Captured-input graph mutation mismatch'
                baseline(0)
                changed=((out.float()-base.float()).norm()/base.float().norm().clamp_min(1e-12)).item()
                assert torch.isfinite(out).all() and torch.isfinite(base).all() and changed<=0.001
                row['changed_captured_input_graph_exact']=True;row['changed_relative_l2']=changed
                a.copy_(a_full[:m]);sa.copy_(sa_full[:m])
                times={'cutlass':[],'fused':[]}
                labels=('cutlass','fused','existing_candidate') if args.skinny else ('cutlass','fused')
                if args.skinny:times['existing_candidate']=[]
                for order in (labels,tuple(reversed(labels))):
                    for label in order:
                        fn={'cutlass':baseline,'fused':fused,'existing_candidate':existing}[label]
                        times[label].append(bench_graph([lambda i=i,fn=fn:fn(i) for i in range(copies)],iters=args.iters))
                row['timing_us_median_min_max']=times;save();print(json.dumps(row),flush=True)
                del graph
        if args.skinny:verify_skinny()
        result['status']='passed';save()
    except BaseException:
        result['status']='failed';result['error']=traceback.format_exc();save();raise

if __name__=='__main__':main()
