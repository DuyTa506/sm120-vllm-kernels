#!/usr/bin/env python3
"""Offline real-inference route capture; GPU use occurs only after explicit invocation.

Reuses prompt token IDs from existing production-recorded agentic trajectories.
No HTTP service, external tools, new trajectory recording, or source-file patching.
"""
import argparse
import hashlib
import json
import lzma
import os
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[2]


def shutdown_owned_engine(llm, out):
    """v0.32 InprocClient.shutdown detaches its finalizer and closes this engine.

    Call only after route export: EngineCore.shutdown frees distributed resources.
    Constructor failure has no complete owned handle; the --rm container is the
    remaining cleanup boundary.
    """
    prior_error = sys.exc_info()[0] is not None
    record = {"api": "llm.llm_engine.engine_core.shutdown",
              "status": "no_complete_handle" if llm is None else "started"}
    try:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
            record["status"] = "completed"
    except BaseException as ex:
        record.update(status="failed", error=f"{type(ex).__name__}: {ex}")
        if not prior_error:
            raise
        traceback.print_exc()
    finally:
        (out / "shutdown.json").write_text(json.dumps(record, indent=2) + "\n")


def select_prompts(path, concurrency, offset):
    with (lzma.open if str(path).endswith(".xz") else open)(path, "rt") as f:
        trajectories = json.load(f)["trajectories"]
    # Interleave task trajectories by turn index; never pad concurrency by duplicating prompts.
    candidates = [(tr["task"], turn, tr["turns"][turn]["prompt_ids"])
                  for turn in range(max(len(tr["turns"]) for tr in trajectories))
                  for tr in trajectories if turn < len(tr["turns"])]
    selected = candidates[offset:offset + concurrency]
    if len(selected) != concurrency:
        raise ValueError("not enough distinct recorded turns for the requested concurrency/offset")
    prompts = [{"prompt_token_ids": list(ids)} for _, _, ids in selected]
    metadata = [{"task": task, "turn": turn, "prompt_tokens": len(ids),
                 "prompt_ids_sha256": hashlib.sha256(
                     json.dumps(ids, separators=(",", ":")).encode()).hexdigest()}
                for task, turn, ids in selected]
    return prompts, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--trajectory", type=Path,
                        default=ROOT / "results/agentic/trajectories_short_production.json.xz")
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--output-tokens", type=int, default=192)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--model", default="unsloth/Qwen3.6-35B-A3B-NVFP4-Fast")
    parser.add_argument("--seed", type=int, default=101)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--target-only", action="store_true")
    group.add_argument("--control", action="store_true", help="Generate matching outputs without any route capture")
    parser.add_argument("--compare-output", type=Path, help="Compare output token IDs with a prior control outputs.json")
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 64 or args.prompt_offset < 0:
        parser.error("concurrency must be 1..64 and prompt offset nonnegative")
    if min(args.output_tokens, args.max_steps, args.max_rows) < 1:
        parser.error("output tokens and capture limits must be positive")
    if args.out.exists() and any(args.out.glob("manifest.json")):
        parser.error("capture output already contains a manifest; use a fresh directory")
    prompts, selected = select_prompts(args.trajectory, args.concurrency, args.prompt_offset)
    if any(len(p["prompt_token_ids"]) + args.output_tokens > 131072 for p in prompts):
        parser.error("selected prompt plus generation exceeds the model context")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_MOE_SKIP_PADDING"] = "1"
    from capture_runtime import V2RouteCapture
    capture = None if args.control else V2RouteCapture(args.max_steps, args.max_rows, not args.target_only)
    metadata = {"instrumentation_only": True, "mode": "control" if args.control else
                "target_only" if args.target_only else "target_and_draft",
                "selected_prompts": selected, "trajectory": str(args.trajectory),
                "concurrency": args.concurrency, "output_tokens": args.output_tokens, "seed": args.seed,
                "argv": sys.argv, "environment": {name: os.environ.get(name) for name in (
                    "VLLM_MOE_SKIP_PADDING", "VLLM_USE_V2_MODEL_RUNNER", "VLLM_SM120_FP8_PREFILL",
                    "VLLM_SM120_FP8_PREFILL_NOGLUE", "VLLM_FI_KERNEL_PAGES",
                    "VLLM_GDN_INTERNAL_CKPT", "VLLM_GDN_FUSED_CONV")}}
    args.out.mkdir(parents=True, exist_ok=True)
    success, error, llm = False, None, None
    try:
        if capture is not None:
            capture.manifest.update(metadata)
            capture.install()
        from vllm import LLM, SamplingParams
        sizes = [int(x) for x in (ROOT / "scripts/cudagraph_sizes.txt").read_text().split()]
        llm = LLM(
            model=args.model, language_model_only=True, max_model_len=131072,
            max_num_seqs=64, max_num_batched_tokens=16384,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_chunked_prefill=True, enable_prefix_caching=True, async_scheduling=True,
            block_size=2176, seed=args.seed,
            speculative_config={"method": "mtp", "num_speculative_tokens": 2},
            compilation_config={"cudagraph_capture_sizes": sizes},
            kernel_config={"moe_backend": "flashinfer_cutlass"},
            aux_output_config={"enable_return_routed_experts": not args.control},
        )
        outputs = llm.generate(prompts, SamplingParams(
            temperature=0, max_tokens=args.output_tokens, ignore_eos=True))
        output = {"selected_prompts": selected,
                  "requests": [{"request_id": o.request_id,
                                "output_token_ids": list(o.outputs[0].token_ids),
                                "finish_reason": o.outputs[0].finish_reason} for o in outputs]}
        (args.out / "outputs.json").write_text(json.dumps(output, indent=2) + "\n")
        if (len(output["requests"]) != args.concurrency
                or len({r["request_id"] for r in output["requests"]}) != args.concurrency):
            raise RuntimeError("incomplete or duplicate request batch in diagnostic run")
        if args.compare_output:
            reference = json.loads(args.compare_output.read_text())
            same = (reference["selected_prompts"] == selected and
                    [x["output_token_ids"] for x in reference["requests"]] ==
                    [x["output_token_ids"] for x in output["requests"]])
            metadata["control_token_ids_match"] = same
            if capture is not None:
                capture.manifest["control_token_ids_match"] = same
            if not same:
                raise RuntimeError("captured run differs from control prompt selection/output token IDs")
        if any(len(r["output_token_ids"]) != args.output_tokens or r["finish_reason"] != "length"
               for r in output["requests"]):
            raise RuntimeError("incomplete generation in diagnostic run")
        success = True
    except BaseException as ex:
        error = f"{type(ex).__name__}: {ex}"
        raise
    finally:
        try:
            if capture is not None:
                try:
                    capture.export(args.out, success=success, error=error)
                except Exception:
                    if error is None:
                        raise
                    traceback.print_exc()
            else:
                metadata.update(success=success, error=error, draft_capture="disabled_control")
                (args.out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        finally:
            shutdown_owned_engine(llm, args.out)
    print(args.out)
    print("Instrumentation only: this run is not a latency/throughput baseline.")


if __name__ == "__main__":
    main()
