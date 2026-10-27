"""Opt-in scalar dispatch spike; --preflight is CPU-only, --gpu is explicit.

Run in the pinned qwen-prefill-dev image. GPU validation is intentionally a
separate mode so source/guard checks can run while Arm A owns the accelerator.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
BACKEND = HERE / "scalar/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py"


def emit(**record):
    print(json.dumps(record), flush=True)


def prepare_exact(module, raw, prep, make_expert):
    """Retain original processed scales; do not reconstruct them via reciprocal."""
    expert = make_expert(module)
    expert.quant_config = SimpleNamespace(
        w1_scale=prep["s13"], w2_scale=prep["s2"],
        a1_gscale=prep["a1_gscale"], a2_gscale=prep["a2_gscale"],
        g1_alphas=prep["g1"], g2_alphas=prep["g2"],
    )
    expert.moe_config = SimpleNamespace(moe_parallel_config=SimpleNamespace(
        tp_size=1, tp_rank=0, ep_size=1, ep_rank=0, use_ep=False,
    ))
    expert._process_weights_fixed_scales(SimpleNamespace(
        w13_weight=prep["w13"],
        w13_weight_scale_2=1.0 / raw["wgs13"],
        w2_weight_scale_2=1.0 / raw["wgs2"],
        w13_input_scale=prep["a13"].expand(256).contiguous(),
        w2_input_scale=prep["a2"].expand(256).contiguous(),
    ))
    return expert


def preflight():
    import torch

    # Execute only the actual load-time selection method; do not import vLLM,
    # initialize CUDA, construct workspaces, or compile any FlashInfer kernel.
    tree = ast.parse(BACKEND.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "FlashInferB12xExperts")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert "_maybe_use_scalar_input_scale" in methods, "scalar selection not implemented"
    namespace = {"torch": torch, "_B12X_SCALAR_INPUT_SCALE": True,
                 "_B12X_FIX_SCALES": True}
    code = ast.Module(body=[methods["_maybe_use_scalar_input_scale"]], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(BACKEND), "exec"), namespace)
    select = namespace["_maybe_use_scalar_input_scale"]

    def expert(scale=None, **changes):
        obj = SimpleNamespace(
            global_num_experts=256, num_local_experts=256, ep_rank=0,
            topk=8, hidden_dim=2048, intermediate_size_per_partition=512,
            out_dtype=torch.bfloat16, _activation_str="silu", _hybrid_min_tokens=1024,
            _input_global_scale=(torch.full((256,), 0.125) if scale is None else scale),
            moe_config=SimpleNamespace(moe_parallel_config=SimpleNamespace(ep_size=1, use_ep=False)),
        )
        obj.__dict__.update(changes)
        return obj

    cases = 0
    for value in (0.125, 1.0, 2.5):
        obj = expert(torch.full((256,), value))
        original = obj._input_global_scale
        alpha = torch.linspace(0.00001, 0.0001, 256)
        expected = alpha * original
        select(obj)
        assert obj._input_global_scale.shape == (1,)
        assert torch.equal(alpha * obj._input_global_scale, expected)
        assert obj._input_global_scale.data_ptr() != original.data_ptr()
        cases += 1
    nonuniform = torch.ones(256)
    nonuniform[17] = 1.125
    for scale in (nonuniform, torch.zeros(256), torch.full((256,), -1.0),
                  torch.full((256,), float("inf")), torch.full((256,), float("nan")),
                  torch.ones(255), torch.ones(1)):
        obj = expert(scale)
        select(obj)
        assert obj._input_global_scale is scale, "ineligible scale was replaced"
        cases += 1
    for changes in (
        {"global_num_experts": 128}, {"num_local_experts": 128}, {"ep_rank": 1},
        {"topk": 10}, {"hidden_dim": 4096}, {"intermediate_size_per_partition": 1024},
        {"out_dtype": torch.float16}, {"_activation_str": "relu2"},
        {"_hybrid_min_tokens": 0}, {"_hybrid_min_tokens": 3}, {"_hybrid_min_tokens": 1023},
        {"moe_config": SimpleNamespace(moe_parallel_config=SimpleNamespace(ep_size=2, use_ep=True))},
    ):
        obj = expert(**changes)
        original = obj._input_global_scale
        select(obj)
        assert obj._input_global_scale is original, "ineligible configuration was changed"
        cases += 1
    for flag in ("_B12X_SCALAR_INPUT_SCALE", "_B12X_FIX_SCALES"):
        namespace[flag] = False
        obj = expert()
        original = obj._input_global_scale
        select(obj)
        assert obj._input_global_scale is original
        namespace[flag] = True
        cases += 1

    # The entire forward path remains the exact-alpha shared backend implementation.
    base = ast.parse((HERE / "shared_exact/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py").read_text())
    base_cls = next(n for n in base.body if isinstance(n, ast.ClassDef))
    base_methods = {n.name: n for n in base_cls.body if isinstance(n, ast.FunctionDef)}
    for name in ("apply", "_ensure_wrapper", "_apply_cutlass"):
        assert ast.dump(methods[name]) == ast.dump(base_methods[name]), name
    selection_call = methods["_process_weights_fixed_scales"].body.pop()
    assert ast.dump(selection_call) == ast.dump(ast.parse("self._maybe_use_scalar_input_scale()").body[0])
    assert ast.dump(methods["_process_weights_fixed_scales"]) == ast.dump(base_methods["_process_weights_fixed_scales"])
    # Exercise the actual fixture without importing vLLM or allocating GPU data.
    # This float32 value changes under a reciprocal round trip.
    original = torch.tensor(0.007007007021456957, dtype=torch.float32)
    assert not torch.equal(original, 1.0 / (1.0 / original))
    captured = []
    fake = SimpleNamespace(_process_weights_fixed_scales=captured.append)
    prep = dict.fromkeys(("s13", "s2", "g1", "g2", "w13"), torch.ones(256))
    prep.update(a13=original, a2=original * 2,
                a1_gscale=(1.0 / original).repeat(256),
                a2_gscale=(1.0 / (original * 2)).repeat(256))
    prepare_exact(None, {"wgs13": torch.ones(256), "wgs2": torch.ones(256)},
                  prep, lambda _: fake)
    assert torch.equal(captured[0].w13_input_scale, original.repeat(256))
    assert torch.equal(captured[0].w2_input_scale, (original * 2).repeat(256))
    assert not torch.cuda.is_initialized()
    emit(kind="preflight", status="passed", cases=cases,
         baseline="shared_exact", original_input_scales_retained=True,
         forward_methods_unchanged=True, cuda_initialized=False)


def runtime_source_evidence():
    import flashinfer
    root = Path(flashinfer.__file__).resolve().parent / "fused_moe/cute_dsl"
    paths = [root / "b12x_moe.py", root / "blackwell_sm12x/moe_dispatch.py",
             root / "blackwell_sm12x/moe_dynamic_kernel.py",
             root / "blackwell_sm12x/_moe_dynamic/gated.py"]
    emit(kind="runtime_source", files={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in paths})


def gpu(args):
    import torch
    from moe_common import H, load_hidden, load_layer_raw, prep_cutlass, route
    from test_shared_workspace import make_expert, wrapper
    from verify_shared_workspace import apply, capture, errors, load_backend
    from vllm.v1.worker import workspace as ws
    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as dispatch
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dynamic_kernel import _can_use_gated_optimized_kernel

    os.environ["VLLM_B12X_FIX_SCALES"] = "1"
    os.environ["VLLM_B12X_SHARED_WORKSPACE"] = "1"
    os.environ["VLLM_B12X_SCALAR_INPUT_SCALE"] = "1"
    os.environ["VLLM_B12X_HYBRID_MIN_TOKENS"] = "1024"
    os.environ["VLLM_B12X_ACT_SCALES"] = "static"
    runtime_source_evidence()
    assert dispatch._FORCED_BACKEND is None, "do not benchmark a forced dispatch override"
    ws.init_workspace_manager(torch.device("cuda:0"))
    modules = [load_backend("shared_exact"), load_backend("scalar")]
    original_compile = dispatch._get_dynamic_kernel
    for layer in args.layers:
        raw = load_layer_raw(layer)
        base_prep = prep_cutlass(raw)
        hidden = load_hidden(layer)
        assert hidden is not None, "captured Qwen activations are required"
        for scale_case in ("uniform", "nonuniform"):
            prep = dict(base_prep)
            if scale_case == "nonuniform":
                prep["a13"] = prep["a13"].repeat(256)
                prep["a13"][17].div_(1.25)
                prep["a1_gscale"] = 1.0 / prep["a13"]
            experts = [prepare_exact(module, raw, prep, make_expert) for module in modules]
            for name in ("_cutlass_g1", "_cutlass_g2"):
                assert torch.equal(getattr(experts[0], name), getattr(experts[1], name))
            assert experts[0]._input_global_scale.numel() == 256
            assert experts[1]._input_global_scale.numel() == (1 if scale_case == "uniform" else 256)
            for obj in experts:
                assert wrapper(obj)._dynamic_workspace.tile_m == 128
            for tokens in args.tokens:
                x = hidden[torch.arange(tokens, device="cuda") % hidden.shape[0]].contiguous()
                weights, ids = route(x, raw["gate"])
                outputs = [torch.empty((tokens, H), dtype=torch.bfloat16, device="cuda") for _ in experts]
                calls = [lambda i=i: apply(experts[i], prep, x, ids, weights, outputs[i])
                         for i in range(2)]
                if tokens < 1024:
                    with autotune(True):
                        calls[0]()
                graphs = []
                for variant, call in enumerate(calls):
                    observed = []
                    def trace(*pos, **kw):
                        observed.append(bool(kw.get("share_input_across_experts", False)))
                        return original_compile(*pos, **kw)
                    with patch.object(dispatch, "_get_dynamic_kernel", trace):
                        graphs.append(capture(call))
                    expected = variant == 1 and scale_case == "uniform"
                    if tokens >= 1024:
                        assert observed and all(v == expected for v in observed)
                        assert _can_use_gated_optimized_kernel(
                            activation="silu", sf_vec_size=16, mma_tiler_mn=(128, 128),
                            hidden_size=2048, intermediate_size=512, num_topk=8,
                            share_input_across_experts=expected)
                        assert experts[variant]._ensure_wrapper()._folded_w1_alpha.numel() == 256
                    else:
                        assert not observed, "decode crossed into dynamic MoE"
                    emit(kind="dispatch", layer=layer, tokens=tokens, scales=scale_case,
                         variant=variant, shared_input=observed,
                         implementation="MoEGatedDynamicKernel" if tokens >= 1024 else "CUTLASS")
                saved_ids, saved_weights = ids.clone(), weights.clone()
                for pattern in ("real", "eight_hot", "masked_tail", "all_masked", "shifted"):
                    # Negative ids are a b12x mask test, not a CUTLASS contract.
                    if tokens < 1024 and pattern in ("masked_tail", "all_masked"):
                        continue
                    ids.copy_(saved_ids)
                    weights.copy_(saved_weights)
                    if pattern == "eight_hot":
                        hot = torch.arange(8, device="cuda", dtype=torch.int32)
                        if scale_case == "nonuniform":
                            # Guarantee the deliberately changed expert scale
                            # is selected alongside seven unchanged scales.
                            hot[-1] = 17
                        ids.copy_(hot.expand_as(ids))
                        weights.fill_(1 / 8)
                    elif pattern == "masked_tail":
                        ids[-max(1, tokens // 8):].fill_(-1)
                    elif pattern == "all_masked":
                        ids.fill_(-1)
                    elif pattern == "shifted":
                        ids.add_(17).remainder_(256)
                    for replay in range(args.replays):
                        graphs[0].replay()
                        first = outputs[0].clone()
                        graphs[0].replay()
                        noise = errors(outputs[0], first)
                        graphs[1].replay()
                        torch.cuda.synchronize()
                        error = errors(outputs[1], outputs[0])
                        limit = max(0.002, 3 * noise["rel_l2"])
                        emit(kind="correctness", layer=layer, tokens=tokens, scales=scale_case,
                             routing=pattern, replay=replay, error=error, baseline_repeat=noise,
                             rel_l2_limit=limit)
                        assert error["finite"] and error["rel_l2"] <= limit
                        if pattern == "all_masked":
                            assert not torch.count_nonzero(outputs[1])
                ids.copy_(saved_ids)
                weights.copy_(saved_weights)
                for order in ((0, 1), (1, 0)):
                    samples = [[], []]
                    for _ in range(args.repeats):
                        for i in order:
                            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                            start.record()
                            graphs[i].replay()
                            end.record()
                            end.synchronize()
                            samples[i].append(start.elapsed_time(end) * 1000)
                    medians = [statistics.median(s) for s in samples]
                    emit(kind="timing", layer=layer, tokens=tokens, scales=scale_case,
                         mode="full_apply_cuda_graph", order=order,
                         baseline_us=medians[0], scalar_us=medians[1],
                         speedup=medians[0] / medians[1])
                del graphs
            ws.reset_workspace_manager()
            ws.init_workspace_manager(torch.device("cuda:0"))
    emit(kind="complete", status="passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--gpu", action="store_true")
    parser.add_argument("--tokens", nargs="+", type=int, default=[1023, 1024, 8192])
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 19])
    parser.add_argument("--replays", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if any(n < 1 or n > 16384 for n in args.tokens) or args.replays < 1 or args.repeats < 1:
        parser.error("tokens must be 1..16384; replays and repeats must be positive")
    preflight() if args.preflight else gpu(args)


if __name__ == "__main__":
    main()
