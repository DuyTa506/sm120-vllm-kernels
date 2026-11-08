"""CPU-safe CLI; --run-gpu explicitly tests same-byte rowwise FP8 DG adapter.
Synthetic operands on actual Qwen shapes; not checkpoint/model quality proof.
"""
import argparse
import json
from pathlib import Path
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-gpu', action='store_true')
    p.add_argument('--fused', action='store_true', help='Use isolated experimental fused-scale binding for correctness/graph checks and add timing')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--ms', default='32,64,1024,4096')
    p.add_argument('--shapes', default='gdn_qkvz,gdn_out')
    p.add_argument('--iters', type=int, default=30)
    args = p.parse_args()
    if not args.run_gpu:
        print('CPU preflight only; --run-gpu required'); return
    assert not args.out.exists(), 'Refusing to overwrite results'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    import torch
    import deep_gemm
    from vllm import _custom_ops as ops
    from common import bench_graph, gpu_state, kernel_names
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(1729)
    shapes = {'gdn_qkvz': (12288, 2048), 'gdn_out': (2048, 4096)}
    record = {'status': 'running', 'fused_candidate': args.fused, 'scope': 'same synthetic FP8 bytes and arbitrary row/channel scales; no vLLM integration', 'timing_contract': 'Both CUTLASS and adapter use preallocated BF16 output; adapter additionally uses preallocated FP32 scratch. Allocation excluded.', 'cases': []}
    def save():
        args.out.write_text(json.dumps(record, indent=2) + '\n')
    save()
    @torch.compile(fullgraph=True)
    def epilogue(acc, sa, sb, out):
        out.copy_((acc * sa * sb.T).to(torch.bfloat16))
    try:
        for name in args.shapes.split(','):
            n, k = shapes[name]
            # More than 128MiB L2: weight copies carry identical data.
            copies = max(2, (256 * 1024**2 + n*k-1)//(n*k))
            b = torch.randn(n, k, device='cuda').to(torch.float8_e4m3fn)
            ws = [b] + [b.clone() for _ in range(copies-1)]
            sb = (0.013 + torch.rand(n, 1, device='cuda') * 0.027).float()
            bs = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(torch.ones(n, k//128, device='cuda'))
            for m in map(int, args.ms.split(',')):
                x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
                a, sa = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
                ones = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(torch.ones(m, k//128, device='cuda'))
                acc = torch.empty(m, n, device='cuda', dtype=torch.float32)
                out = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
                baseline_out = torch.empty_like(out)
                def dg(i):
                    deep_gemm.fp8_fp4_gemm_nt((a, ones), (ws[i], bs), acc, recipe_a=(1,128), recipe_b=(1,128))
                def standalone(i):
                    dg(i); epilogue(acc, sa, sb, out)
                def fused(i):
                    deep_gemm._C.fp8_rowwise_gemm_nt_sm120((a, ones), (ws[i], bs), out, sa, sb)
                adapted = fused if args.fused else standalone
                def baseline(i):
                    torch.ops._C.cutlass_scaled_mm(baseline_out, a, ws[i].T, sa, sb, None)
                    return baseline_out
                adapted(0); ref = baseline(0)
                sample = (a[:16].double() @ b[:64].double().T) * sa[:16].double() * sb[:64].double().T
                denom = sample.norm().clamp_min(1e-12)
                err_a = ((out[:16,:64].double()-sample).norm()/denom).item()
                err_b = ((ref[:16,:64].double()-sample).norm()/denom).item()
                relative = ((out.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-12)).item()
                row = {'shape': name, 'm':m, 'n':n, 'k':k, 'weight_copies':copies,
                       'bit_mismatches':int((out!=ref).sum()), 'relative_l2_vs_cutlass':relative,
                       'sample_fp64_relative_l2_adapter':err_a,'sample_fp64_relative_l2_cutlass':err_b}
                def fused_standalone_check():
                    fused_value = out.clone()
                    standalone(0)
                    comparison = {'bitwise_exact': bool(torch.equal(fused_value,out)),
                                  'bit_mismatches': int((fused_value!=out).sum()),
                                  'relative_l2': ((fused_value.float()-out.float()).norm()/out.float().norm().clamp_min(1e-12)).item()}
                    adapted(0)
                    return comparison
                if args.fused:
                    row['fused_vs_standalone'] = fused_standalone_check()
                record['cases'].append(row); save()
                assert torch.isfinite(out).all() and torch.isfinite(ref).all() and relative <= 0.01
                assert err_a <= max(1e-6, err_b*1.1), 'Adapter increases sampled reference error'
                # Replay must consume changed activation bytes, not frozen output.
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): adapted(0)
                changed_checks = []
                for factor, offset in ((0.37,0.21),(1.7,-0.4),(0.03,0.0)):
                    changed,changed_sa = ops.scaled_fp8_quant(x*factor+offset, use_per_token_if_dynamic=True)
                    a.copy_(changed); sa.copy_(changed_sa); graph.replay(); torch.cuda.synchronize()
                    replay = out.clone(); adapted(0)
                    assert torch.equal(replay,out), 'Changed-input replay differs from eager adapter'
                    changed_ref = baseline(0)
                    changed_l2 = ((out.float()-changed_ref.float()).norm()/changed_ref.float().norm().clamp_min(1e-12)).item()
                    standalone_check = fused_standalone_check() if args.fused else None
                    changed_checks.append({'fused_vs_standalone':standalone_check,'factor':factor,'offset':offset,'relative_l2_vs_cutlass':changed_l2,
                                           'bit_mismatches':int((out!=changed_ref).sum()),'graph_eager_exact':True})
                    assert torch.isfinite(out).all() and torch.isfinite(changed_ref).all() and changed_l2 <= 0.01
                row['changed_input_checks'] = changed_checks
                original_a, original_sa = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
                a.copy_(original_a); sa.copy_(original_sa)
                row['timing_inputs'] = 'original seeded activation and its original token scales'
                row['gpu'] = gpu_state()
                # Interleaved order across two complete graph timing passes.
                timings = {'cutlass':[], 'dg_core_fp32':[], 'dg_adapter':[]}
                labels = ('cutlass','dg_core_fp32','dg_adapter') + (('dg_fused',) if args.fused else ())
                if args.fused: timings['dg_fused'] = []
                for order in (labels, tuple(reversed(labels))):
                    for label in order:
                        fn = {'cutlass':baseline, 'dg_core_fp32':dg, 'dg_adapter':standalone, 'dg_fused':fused}[label]
                        timings[label].append(bench_graph([lambda i=i,fn=fn:fn(i) for i in range(copies)], iters=args.iters))
                row['timing_us_median_min_max'] = timings
                try:
                    row['selected_kernel_names'] = kernel_names(lambda: adapted(0))
                except Exception as exc:
                    row['selected_kernel_names_error'] = str(exc)
                save()
                print(json.dumps(row), flush=True)
                del graph, a, sa, acc, out, ref, replay
            del ws, b, bs, sb
        record['status']='passed'; save()
    except BaseException:
        record['status']='failed'; record['error']=traceback.format_exc(); save(); raise


if __name__ == '__main__':
    main()
