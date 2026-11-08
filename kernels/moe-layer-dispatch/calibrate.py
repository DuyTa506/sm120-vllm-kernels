"""Isolated full-call replay. No active runtime patch is installed by this script."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from policy import choose, digest, select, cache_identity

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "moe-nvfp4"
SOURCE = BASE / "scalar/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", type=Path, default=BASE / "data/moe_inputs.pt")
    p.add_argument("--heldout-capture", type=Path)
    p.add_argument("--layers", nargs="+", type=int, default=[0, 19, 39])
    p.add_argument("--tokens", nargs="+", type=int, default=[1024, 2048])
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--replays", type=int, default=5)
    p.add_argument("--max-rel-l2", type=float, default=0.002)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert min(args.tokens) >= 1024, "below1024 policy remains untouched"
    assert args.repeats >= 10 and args.replays >= 2
    for name, value in dict(VLLM_B12X_FIX_SCALES="1", VLLM_B12X_SHARED_WORKSPACE="1",
                            VLLM_B12X_SCALAR_INPUT_SCALE="1", VLLM_B12X_HYBRID_MIN_TOKENS="1024",
                            VLLM_B12X_ACT_SCALES="static").items():
        os.environ[name] = value
    sys.path.insert(0, str(BASE))
    import torch
    from moe_common import H, load_layer_raw, prep_cutlass, route, snap
    from verify_shared_workspace import apply, capture, errors, load_backend
    from verify_scalar_dispatch import prepare_exact
    from test_shared_workspace import make_expert
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.v1.worker import workspace as ws
    import flashinfer
    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    assert moe_dispatch._FORCED_BACKEND is None
    train = torch.load(args.capture, map_location="cpu", weights_only=True)["hidden"]
    held = (torch.load(args.heldout_capture, map_location="cpu", weights_only=True)["hidden"]
            if args.heldout_capture else train)
    if args.heldout_capture:
        assert digest(args.capture) != digest(args.heldout_capture), "heldout capture duplicates calibration"
    # All input checks precede GPU initialization. Never tile/repeat captured tokens.
    for layer in args.layers:
        available = min(train[layer].shape[0], held[layer].shape[0])
        assert max(args.tokens) <= (available if args.heldout_capture else available // 2)
    provenance = dict(benchmark_regime="hot-cache isolated layer screen; not cold model traversal",
                      helper_sha256={str(path): digest(path) for path in [Path(__file__), HERE / "policy.py"] + [BASE / name for name in ("moe_common.py", "verify_scalar_dispatch.py", "verify_shared_workspace.py", "test_shared_workspace.py")]},
                      source_sha256=digest(SOURCE), source=str(SOURCE),
                      capture_sha256=digest(args.capture), checkpoint=snap(),
                      heldout_capture_sha256=digest(args.heldout_capture) if args.heldout_capture else None,
                      split="independent_capture" if args.heldout_capture else "disjoint_halves_same_prompt",
                      limitation="same-prompt heldout is not independent-prompt generalization" if not args.heldout_capture else None)
    provenance["runtime_sources"] = {str(path): digest(path) for path in (
        Path(flashinfer.__file__).resolve().parent / "fused_moe/cute_dsl/b12x_moe.py",
        Path(moe_dispatch.__file__).resolve(),
    )}
    ws.init_workspace_manager(torch.device("cuda:0"))
    module = load_backend("scalar")
    policy = dict(source_sha256=provenance["source_sha256"], entries={})
    records = []
    args.output.mkdir(parents=True, exist_ok=False)

    def measure(graphs):
        passes = []
        for order in ((0, 1), (1, 0)):
            times = [[], []]
            for _ in range(args.repeats):
                for i in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record(); graphs[i].replay(); end.record(); end.synchronize()
                    times[i].append(start.elapsed_time(end) * 1000)
            passes.append(dict(order=list(order), scalar_us=statistics.median(times[0]),
                               cutlass_us=statistics.median(times[1])))
        return passes

    # Freeze policy globally before any heldout timings are collected.
    for phase in ("calibration", "heldout"):
        if phase == "heldout":
            (args.output / "policy.json").write_text(json.dumps(policy, indent=2))
            frozen_identity = cache_identity(policy)
        for layer in args.layers:
            raw = load_layer_raw(layer)
            prep = prep_cutlass(raw)
            expert = prepare_exact(module, raw, prep, make_expert)
            assert expert._input_global_scale.numel() == 1, "candidate requires actual scalar-scale path"
            source_hidden = train[layer] if phase == "calibration" else held[layer]
            offset = 0 if phase == "calibration" or args.heldout_capture else train[layer].shape[0] // 2
            for tokens in args.tokens:
                x = source_hidden[offset:offset + tokens].to("cuda").contiguous()
                weights, ids = route(x, raw["gate"])
                out = [torch.empty((tokens, H), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
                calls = [lambda: apply(expert, prep, x, ids, weights, out[0]),
                         lambda: expert._apply_cutlass(out[1], x, prep["w13"], prep["w2"], weights, ids, MoEActivation.SILU)]
                # Tactics are calibrated ONLY on calibration inputs and reused on heldout.
                if phase == "calibration":
                    with autotune(True):
                        calls[1]()
                graphs = [capture(call) for call in calls]
                checks = []
                # Change actual activations/routing behind stable graph addresses.
                # Keep extra calibration rows inside its half to prevent leakage.
                split_end = (train[layer].shape[0] // 2
                             if phase == "calibration" and not args.heldout_capture
                             else source_hidden.shape[0])
                replay_offsets = [offset]
                if offset + 2 * tokens <= split_end:
                    replay_offsets.append(offset + tokens)
                for replay_offset in replay_offsets:
                    x.copy_(source_hidden[replay_offset:replay_offset + tokens])
                    new_weights, new_ids = route(x, raw["gate"])
                    weights.copy_(new_weights); ids.copy_(new_ids)
                    for _ in range(args.replays):
                        for graph in graphs:
                            graph.replay()
                        torch.cuda.synchronize()
                        saved = [output.clone() for output in out]
                        check = errors(saved[1], saved[0])
                        check["capture_row_offset"] = replay_offset
                        checks.append(check)
                        # Independently refresh each eager result; cross-backend
                        # graph agreement alone cannot detect stale graph inputs.
                        for backend_index, call in enumerate(calls):
                            call()
                            torch.cuda.synchronize()
                            eager_check = errors(saved[backend_index], out[backend_index])
                            eager_check.update(exact=bool(torch.equal(saved[backend_index], out[backend_index])),
                                               capture_row_offset=replay_offset,
                                               graph_vs_own_eager=backend_index)
                            checks.append(eager_check)
                x.copy_(source_hidden[offset:offset + tokens])
                new_weights, new_ids = route(x, raw["gate"])
                weights.copy_(new_weights); ids.copy_(new_ids)
                row = dict(phase=phase, layer=layer, tokens=tokens,
                           input_sha256=hashlib.sha256(source_hidden[offset:offset + tokens].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                           valid=all(c["finite"] and c["rel_l2"] <= args.max_rel_l2 and c.get("exact", True) for c in checks),
                           errors=checks, passes=measure(graphs))
                if phase == "calibration":
                    policy["entries"][f"{layer}:{tokens}"] = choose(row)
                row["selected"] = select(policy, layer, tokens, digest(SOURCE))
                records.append(row)
                with (args.output / "measurements.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                del graphs, calls, out, x, ids, weights
            del expert, prep, raw
        if phase == "heldout":
            assert cache_identity(policy) == frozen_identity
    summary = dict(provenance=provenance, policy_sha256=cache_identity(policy),
                   accuracy_pass=all(r["valid"] for r in records),
                   status="isolated microbenchmark; integration and task quality gates pending")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    if not summary["accuracy_pass"]:
        raise SystemExit("accuracy failed; see measurements.jsonl; policy is not eligible for integration")


if __name__ == "__main__":
    main()
