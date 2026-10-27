"""Verify the isolated exact-alpha overlay; GPU mode must be scheduled explicitly.

--preflight checks source scope and constructs actual modular kernels on CPU.
--gpu uses real Qwen weights/hidden states and full FusedMoEKernel.apply calls,
including preparation quantization, expert execution, and finalization.
"""

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
RELATIVE = Path("vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py")
VARIANTS = ("cutlass", "shared", "shared_exact")


def emit(**record):
    print(json.dumps(record), flush=True)


def load_classes():
    os.environ["VLLM_B12X_FIX_SCALES"] = "1"
    os.environ["VLLM_B12X_SHARED_WORKSPACE"] = "1"
    os.environ["VLLM_B12X_HYBRID_MIN_TOKENS"] = "1024"
    os.environ["VLLM_B12X_ACT_SCALES"] = "static"
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import FlashInferExperts
    classes = [FlashInferExperts]
    for variant in VARIANTS[1:]:
        path = HERE / variant / RELATIVE
        spec = importlib.util.spec_from_file_location(f"exact_check_{variant}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        classes.append(module.FlashInferB12xExperts)
    return classes


def config(device, max_tokens):
    import torch
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    return SimpleNamespace(
        in_dtype=torch.bfloat16, device=torch.device(device), num_local_experts=256,
        num_experts=256, experts_per_token=8, hidden_dim=2048,
        intermediate_size_per_partition=512, max_num_tokens=max_tokens,
        activation=MoEActivation.SILU, swiglu_limit=None, swiglu_alpha=None,
        swiglu_beta=None, activation_situ_beta=None, activation_situ_linear_beta=None,
        moe_parallel_config=SimpleNamespace(tp_size=1, tp_rank=0, ep_size=1,
            ep_rank=0, dp_size=1, use_ep=False),
    )


def construct(cls, conf, w13_s2, w2_s2, a13, a2, scales):
    from vllm.model_executor.layers.fused_moe.config import nvfp4_moe_quant_config
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEPModular
    quant = nvfp4_moe_quant_config(
        g1_alphas=w13_s2, g2_alphas=w2_s2, a1_gscale=1.0 / a13,
        a2_gscale=1.0 / a2, w1_scale=scales[0], w2_scale=scales[1],
        is_scale_swizzled=True,
    )
    expert = cls(moe_config=conf, quant_config=quant)
    return expert, FusedMoEKernel(MoEPrepareAndFinalizeNoDPEPModular(), expert)


def preflight():
    from audit_cutlass_scales_cpu import method
    import torch
    base = HERE / "shared" / RELATIVE
    candidate = HERE / "shared_exact" / RELATIVE
    for name in ("apply", "_apply_cutlass", "_ensure_wrapper", "workspace_shapes"):
        assert ast.dump(method(base, "FlashInferB12xExperts", name)) == ast.dump(method(candidate, "FlashInferB12xExperts", name))
    # Only the final CUTLASS-alpha block may change. All b12x-scale assignments
    # and weight/layout handling must remain identical to the running candidate.
    old = method(base, "FlashInferB12xExperts", "_process_weights_fixed_scales")
    new = method(candidate, "FlashInferB12xExperts", "_process_weights_fixed_scales")
    def strip_cutlass_block(node):
        for item in ast.walk(node):
            if isinstance(item, ast.If) and any(
                isinstance(child, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "_cutlass_g1" for t in child.targets)
                for child in item.body
            ):
                item.body = [ast.Pass()]
        return ast.dump(node)
    assert strip_cutlass_block(old) == strip_cutlass_block(new)
    scales = (torch.ones(256), torch.ones(256))
    for cls in load_classes():
        _, kernel = construct(cls, config("cpu", 1024), torch.ones(256), torch.ones(256),
                              torch.ones(256), torch.ones(256), scales)
        assert kernel.is_monolithic is False
    assert not torch.cuda.is_initialized()
    emit(kind="preflight", status="passed", actual_modular_constructors=3,
         forward_methods_unchanged=True, b12x_scale_preparation_unchanged=True,
         cuda_initialized=False)


def gpu(args):
    import torch
    from flashinfer.autotuner import autotune
    from moe_common import H, load_hidden, load_layer_raw, prep_cutlass, route
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.v1.worker import workspace as ws
    classes = load_classes()
    ws.init_workspace_manager(torch.device("cuda:0"))

    def errors(actual, reference):
        a, b = actual.float(), reference.float()
        return {"rel_l2": float((a - b).norm() / b.norm().clamp_min(1e-12)),
                "max_abs": float((a - b).abs().max()),
                "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all())}

    def capture(call):
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = call()
        graph.replay()
        torch.cuda.synchronize()
        return graph, output

    for index in args.layers:
        raw = load_layer_raw(index)
        prep = prep_cutlass(raw)
        hidden = load_hidden(index)
        assert hidden is not None, "captured real Qwen hidden states are required"
        experts, kernels = [], []
        for cls in classes:
            w13_s2, w2_s2 = 1.0 / raw["wgs13"], 1.0 / raw["wgs2"]
            a13, a2 = prep["a13"].repeat(256), prep["a2"].repeat(256)
            expert, kernel = construct(cls, config("cuda", max(args.tokens)), w13_s2, w2_s2,
                                       a13, a2, (prep["s13"], prep["s2"]))
            # Original processed layer scales are retained explicitly. The
            # actual default method folds the aliased g1/g2 parameters in place.
            expert.process_weights_after_loading(SimpleNamespace(
                w13_weight=prep["w13"], w13_weight_scale_2=w13_s2,
                w2_weight_scale_2=w2_s2, w13_input_scale=a13, w2_input_scale=a2))
            experts.append(expert)
            kernels.append(kernel)
            if hasattr(expert, "w1_sf_mma"):
                aliases = [
                    view.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
                    for view, source in ((expert.w1_sf_mma, prep["s13"]),
                                         (expert.w2_sf_mma, prep["s2"]))
                ]
                emit(kind="scale_layout_storage", layer=index,
                     variant=VARIANTS[len(experts) - 1], aliases_existing_scales=aliases)
                assert all(aliases), "MMA-layout scale views unexpectedly allocate copies"
        for attr in ("_input_global_scale", "_fc2_input_scale", "_w1_alpha", "_w2_alpha"):
            assert torch.equal(getattr(experts[1], attr), getattr(experts[2], attr))
        for number in (1, 2):
            expected = getattr(experts[0], f"g{number}_alphas")
            corrected = getattr(experts[2], f"_cutlass_g{number}")
            old = getattr(experts[1], f"_cutlass_g{number}")
            assert torch.equal(corrected, expected), "GPU-loaded exact alpha differs from actual default"
            emit(kind="alpha", layer=index, gemm=number, exact_equal=True,
                 old_different=int(torch.count_nonzero(old != expected)))
        for tokens in args.tokens:
            x = hidden[torch.arange(tokens, device="cuda") % hidden.shape[0]].contiguous()
            weights, ids = route(x, raw["gate"])
            calls = [lambda kernel=kernel: kernel.apply(
                x, prep["w13"], prep["w2"], weights, ids, MoEActivation.SILU,
                256, None, False) for kernel in kernels]
            # All variants see one process's CUTLASS tactic cache. Tune through
            # the actual default implementation so kernel timing cannot silently
            # compare unrelated per-process tactic choices.
            with autotune(True):
                calls[0]()
            captured = [capture(call) for call in calls]
            saved_ids, saved_weights = ids.clone(), weights.clone()
            for pattern in ("real", "eight_hot", "shifted"):
                ids.copy_(saved_ids)
                weights.copy_(saved_weights)
                if pattern == "eight_hot":
                    ids.copy_(torch.arange(8, dtype=torch.int32, device="cuda").expand_as(ids))
                    weights.fill_(1 / 8)
                elif pattern == "shifted":
                    ids.add_(17).remainder_(256)
                samples = []
                for graph, output in captured:
                    values = []
                    for _ in range(args.replays):
                        graph.replay()
                        values.append(output.clone())
                    samples.append(values)
                torch.cuda.synchronize()
                assert all(bool(torch.isfinite(value).all()) for values in samples for value in values), "nonfinite replay output"
                # Below the hybrid threshold compare against actual default;
                # above it compare against unchanged shared b12x execution.
                reference_index = 0 if tokens < 1024 else 1
                reference, exact = samples[reference_index], samples[2]
                stable = [all(torch.equal(values[0], v) for v in values[1:]) for values in (reference, exact)]
                ref_noise = max(errors(v, reference[0])["rel_l2"] for v in reference)
                exact_noise = max(errors(v, exact[0])["rel_l2"] for v in exact)
                error = errors(exact[-1], reference[-1])
                bitwise = torch.equal(exact[-1], reference[-1])
                limit = max(1e-6, 3 * max(ref_noise, exact_noise))
                emit(kind="correctness", layer=index, tokens=tokens, routing=pattern,
                     reference=VARIANTS[reference_index], bitwise=bitwise,
                     observed_repeat_stable=stable, error=error, rel_l2_limit=limit,
                     old_vs_default=errors(samples[1][-1], samples[0][-1]))
                assert error["finite"]
                if all(stable):
                    assert bitwise, "repeat-stable exact candidate differs bitwise"
                else:
                    assert error["rel_l2"] <= limit, "difference exceeds observed reduction noise"
            ids.copy_(saved_ids)
            weights.copy_(saved_weights)
            for order in ((0, 1, 2), (2, 1, 0)):
                timings = [[], [], []]
                for _ in range(args.repeats):
                    for i in order:
                        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        start.record()
                        captured[i][0].replay()
                        end.record()
                        end.synchronize()
                        timings[i].append(start.elapsed_time(end) * 1000)
                emit(kind="timing", layer=index, tokens=tokens, order=order,
                     mode="full_FusedMoEKernel_apply_cuda_graph",
                     median_us={name: statistics.median(times) for name, times in zip(VARIANTS, timings)})
            del captured, samples, calls
        del kernels, experts
        ws.reset_workspace_manager()
        ws.init_workspace_manager(torch.device("cuda:0"))
    emit(kind="complete", status="passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--gpu", action="store_true")
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 3, 4, 12, 128, 512, 1023, 1024])
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 19])
    parser.add_argument("--replays", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if any(n < 1 or n > 16384 for n in args.tokens) or args.replays < 2 or args.repeats < 1:
        parser.error("tokens must be 1..16384; replays>=2 and repeats>=1")
    preflight() if args.preflight else gpu(args)


if __name__ == "__main__":
    main()
