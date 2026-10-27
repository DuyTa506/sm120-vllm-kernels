"""Validate/timestamp the candidate with real Qwen layer weights and routing.

python3 verify_shared_workspace.py --tokens 1024 8192 --replays 10
Run in the pinned vLLM image. Reports JSON lines on stdout; no server changes.
Baseline and candidate use identical kernels/scales; only workspace ownership
changes. Timings are interleaved in both orders because the GPU can be shared.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch

from moe_common import H, load_hidden, load_layer_raw, prep_cutlass, route
from test_shared_workspace import make_expert, wrapper
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.v1.worker import workspace as ws


def load_backend(directory):
    path = Path(__file__).parent / directory / "vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py"
    spec = importlib.util.spec_from_file_location(f"b12x_{directory}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(module, raw, prep):
    expert = make_expert(module)
    expert.quant_config = SimpleNamespace(
        w1_scale=prep["s13"], w2_scale=prep["s2"],
        a1_gscale=prep["a1_gscale"], a2_gscale=prep["a2_gscale"],
        g1_alphas=prep["g1"], g2_alphas=prep["g2"],
    )
    expert.moe_config = SimpleNamespace(moe_parallel_config=SimpleNamespace(
        tp_size=1, tp_rank=0, ep_size=1, ep_rank=0,
    ))
    expert._process_weights_fixed_scales(SimpleNamespace(
        w13_weight=prep["w13"],
        w13_weight_scale_2=1.0 / raw["wgs13"],
        w2_weight_scale_2=1.0 / raw["wgs2"],
    ))
    return expert


def apply(expert, prep, x, ids, weights, output):
    expert.apply(
        output, x, prep["w13"], prep["w2"], weights, ids,
        MoEActivation.SILU, 256, None, None, None, None, None, None, False,
    )


def capture(call):
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def errors(a, b):
    a, b = a.float(), b.float()
    return dict(
        finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        rel_l2=float((a - b).norm() / b.norm().clamp_min(1e-12)),
        max_abs=float((a - b).abs().max()),
    )


def timing_pair(graphs, repeats=30):
    passes = []
    for order in ((0, 1), (1, 0)):
        samples = [[], []]
        for _ in range(repeats):
            for i in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[i].replay()
                end.record()
                end.synchronize()
                samples[i].append(start.elapsed_time(end) * 1000)
        passes.append(dict(
            order=list(order),
            baseline_us=statistics.median(samples[0]),
            shared_us=statistics.median(samples[1]),
        ))
    return passes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1023, 1024, 1025, 8192])
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 19])
    parser.add_argument("--replays", type=int, default=10)
    args = parser.parse_args()
    os.environ["VLLM_B12X_FIX_SCALES"] = "1"
    os.environ["VLLM_B12X_SHARED_WORKSPACE"] = "1"
    ws.init_workspace_manager(torch.device("cuda:0"))
    modules = [load_backend("patched"), load_backend("shared")]
    layers = []
    for index in args.layers:
        raw = load_layer_raw(index)
        prep = prep_cutlass(raw)
        hidden = load_hidden(index)
        if hidden is None:
            raise RuntimeError("Captured Qwen hidden states are required")
        experts = [prepare(module, raw, prep) for module in modules]
        layers.append((index, raw, prep, hidden, experts))

    # Measure actual allocation across layers, excluding weights/scale transforms.
    for variant in range(2):
        before = torch.cuda.memory_allocated()
        for _, _, _, _, experts in layers:
            wrapper(experts[variant])
        print(json.dumps(dict(kind="allocation", variant=variant,
                              layers=len(layers),
                              allocated_mib=(torch.cuda.memory_allocated() - before) / 2**20)), flush=True)
    ws.lock_workspace()

    from flashinfer.autotuner import autotune
    for tokens in args.tokens:
        if tokens > 16384 or tokens < 1:
            raise ValueError("tokens must be in [1, 16384]")
        cases = []
        for index, raw, prep, hidden, experts in layers:
            x = hidden[torch.arange(tokens, device="cuda") % hidden.shape[0]].contiguous()
            weights, ids = route(x, raw["gate"])
            outputs = [torch.empty((tokens, H), dtype=torch.bfloat16, device="cuda") for _ in range(2)]
            cases.append((index, prep, x, ids, weights, experts, outputs))
        if tokens < 1024:
            with autotune(True):
                for _, prep, x, ids, weights, experts, outputs in cases:
                    apply(experts[0], prep, x, ids, weights, outputs[0])
        calls = [lambda variant=i: [apply(experts[variant], prep, x, ids, weights, outputs[variant])
                 for _, prep, x, ids, weights, experts, outputs in cases] for i in range(2)]
        graphs = [capture(call) for call in calls]
        print(json.dumps(dict(kind="timing", tokens=tokens, layers=args.layers,
                              passes=timing_pair(graphs))), flush=True)

        # Preserve the captured pointers while mutating routing for every replay.
        saved = [(ids.clone(), weights.clone()) for _, _, _, ids, weights, _, _ in cases]
        for replay in range(args.replays):
            pattern = ("real", "eight_hot", "masked_tail", "all_masked", "shifted")[replay % 5]
            for j, (_, _, _, ids, weights, _, _) in enumerate(cases):
                ids.copy_(saved[j][0])
                weights.copy_(saved[j][1])
                if pattern == "eight_hot":
                    ids.copy_(torch.arange(8, device="cuda", dtype=torch.int32).expand_as(ids))
                    weights.fill_(1 / 8)
                elif pattern == "masked_tail":
                    ids[-max(1, tokens // 8):].fill_(-1)
                elif pattern == "all_masked":
                    ids.fill_(-1)
                elif pattern == "shifted":
                    ids.add_(17).remainder_(256)
            graphs[0].replay()
            baseline_first = [entry[-1][0].clone() for entry in cases]
            graphs[0].replay()
            baseline_noise = [errors(entry[-1][0], first) for entry, first in zip(cases, baseline_first)]
            graphs[1].replay()
            torch.cuda.synchronize()
            for j, (index, _, _, _, _, _, outputs) in enumerate(cases):
                error = errors(outputs[1], outputs[0])
                limit = max(0.002, 3 * baseline_noise[j]["rel_l2"])
                print(json.dumps(dict(kind="correctness", tokens=tokens, layer=index,
                                      replay=replay, routing=pattern, error=error,
                                      baseline_repeat=baseline_noise[j], rel_l2_limit=limit)), flush=True)
                assert error["finite"] and error["rel_l2"] <= limit
                if pattern == "all_masked":
                    assert not torch.count_nonzero(outputs[1])
            # A later call must not overwrite a returned, caller-owned output.
            frozen = [entry[-1][1].clone() for entry in cases]
            calls[0]()
            torch.cuda.synchronize()
            assert all(torch.equal(entry[-1][1], old) for entry, old in zip(cases, frozen))
        del graphs, cases, saved
    print(json.dumps(dict(kind="complete", status="passed")), flush=True)


if __name__ == "__main__":
    main()
