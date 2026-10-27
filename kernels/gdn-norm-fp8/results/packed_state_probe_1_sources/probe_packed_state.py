#!/usr/bin/env python3
"""Owned eager layer18 state/Q/S parity probe. No GPU work without --run-gpu.

CPU preflight checks local inputs/APIs and records source readiness. GPU mode
requires the exact baseline next build mounted at installed vLLM paths.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import uuid

from capture_activations import SNAPSHOT, local_prompt
from capture_io import file_sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MOUNTS = ("v1/attention/backends/flashinfer.py", "v1/attention/backends/sm120_fp8_fmha_glue.py",
          "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
          "model_executor/layers/mamba/ops/fused_conv_post_conv.py")
BASELINE_GDN_SHA = "c82a390240c77032414d916ed705780716cd696f90722eb99eb89419f0aa262c"


def preflight(snapshot):
    ids, prompt = local_prompt(snapshot, 8192)
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None: raise RuntimeError("Use the pinned vLLM container")
    package = Path(spec.origin).parent
    source_checks = {}
    for relative in MOUNTS:
        built, installed = ROOT / "build/vllm" / relative, package / relative
        source_checks[relative] = {"build_sha256": file_sha256(built),
                                  "installed_sha256": file_sha256(installed) if installed.exists() else None}
    gdn = MOUNTS[2]
    if source_checks[gdn]["build_sha256"] != BASELINE_GDN_SHA:
        raise RuntimeError("Baseline GDN source changed; review state ownership before updating this probe")
    for relative, needles in {
        "forward_context.py": ("def override_forward_context(", "prev_context = _forward_context"),
        "v1/worker/worker_base.py": ("worker_extension_cls",),
        "entrypoints/llm.py": ("def collective_rpc(",),
        "v1/engine/core_client.py": ("def shutdown(self, timeout:",),
        "v1/attention/backends/gdn_attn.py": ("class GDNAttentionMetadata:", "prefill_state_indices"),
    }.items():
        path = package / relative
        if not all(needle in path.read_text() for needle in needles): raise RuntimeError(f"API changed: {relative}")
        source_checks[relative] = {"installed_sha256": file_sha256(path)}
    files = ["probe_packed_state.py", "packed_state_probe_worker.py", "packed_state_probe_utils.py",
             "fallback_quant_out.py", "run_packed_state_probe.py"]
    files += [str(p.relative_to(HERE)) for p in sorted((HERE / "overlay").rglob("*.py"))]
    return ids, {"status": "CPU_PREFLIGHT_PASS", "prompt": prompt,
                 "sources": source_checks, "probe_sources": {name: file_sha256(HERE / name) for name in files},
                 "runtime_source_ready": all(source_checks[r]["build_sha256"] == source_checks[r]["installed_sha256"] for r in MOUNTS),
                 "torch_imported": "torch" in sys.modules, "vllm_imported": "vllm" in sys.modules,
                 "scope": "No runtime/GPU calls; an unmatched installed source must be mounted before GPU mode"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-gpu", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/packed_state_probe_1")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    args = parser.parse_args()
    if not args.run_gpu and not args.preflight:
        print(json.dumps({"status": "PENDING_GPU", "torch_imported": "torch" in sys.modules,
                          "message": "Use --preflight for CPU checks; only root schedules --run-gpu."}))
        return
    ids, api = preflight(args.snapshot)
    if args.preflight:
        assert not api["torch_imported"] and not api["vllm_imported"]
        print(json.dumps(api, sort_keys=True)); return
    if not api["runtime_source_ready"]: raise RuntimeError("Exact next baseline build mounts required")
    if not 0 < args.gpu_memory_utilization < 1: parser.error("Invalid GPU utilization")
    required_env = {"VLLM_GDN_INTERNAL_CKPT": "1", "VLLM_GDN_FUSED_CONV": "1",
                    "VLLM_SM120_FP8_PREFILL": "1", "VLLM_SM120_FP8_PREFILL_NOGLUE": "1",
                    "VLLM_FI_KERNEL_PAGES": "16", "VLLM_GDN_NORM_FP8_FUSION": "0"}
    if any(os.environ.get(k) != v for k, v in required_env.items()):
        raise RuntimeError("Use the owned runner's original next environment; live fusion must be off")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    run_id = str(uuid.uuid4())
    os.environ["GDN_STATE_PROBE"] = run_id
    os.environ["PYTHONPATH"] = str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", "")
    directory = args.output_dir.resolve(); directory.mkdir(parents=True, exist_ok=False)
    (directory / "request.json").write_text(json.dumps({"run_id": run_id, "layer": 18}) + "\n")
    options = dict(model=str(args.snapshot), tokenizer=str(args.snapshot), language_model_only=True,
                   dtype="bfloat16", tensor_parallel_size=1, enforce_eager=True, compilation_config=0,
                   moe_backend="flashinfer_b12x",
                   enable_prefix_caching=True, enable_chunked_prefill=True, async_scheduling=False,
                   speculative_config={"method": "mtp", "num_speculative_tokens": 2},
                   max_model_len=16384, max_num_batched_tokens=2048, max_num_seqs=2, block_size=2176,
                   gpu_memory_utilization=args.gpu_memory_utilization, distributed_executor_backend="uni",
                   worker_extension_cls="packed_state_probe_worker.GDNStateProbeWorkerExtension", seed=1731)
    manifest = dict(status="incomplete", run_id=run_id, layer=18, api=api, engine_options=options,
                    created_at_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                    scope="Eager private cache/core parity only; no timing, graph or full-model equivalence claim",
                    environment={key: os.environ.get(key) for key in required_env})
    llm = None; installed = False
    def terminate(signum, frame): raise SystemExit(128 + signum)
    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(**options)
        result = llm.collective_rpc("gdn_state_probe_install", timeout=90, args=(str(directory), run_id))
        if len(result) != 1: raise RuntimeError("Expected exactly one worker")
        installed = True; manifest["installation"] = result[0]
        # Short first: decode can overlap the remaining chunks of the long prompt.
        prompts = [ids[:64], ids]
        outputs = llm.generate([{"prompt_token_ids": p} for p in prompts], SamplingParams(
            temperature=0.0, max_tokens=32, min_tokens=32, ignore_eos=True), use_tqdm=False)
        if len(outputs) != 2 or any(not out.finished or list(out.prompt_token_ids) != p
                or len(out.outputs) != 1 or len(out.outputs[0].token_ids) != 32
                for out, p in zip(outputs, prompts)):
            raise RuntimeError("The two exact-length requests did not finish")
        result = llm.collective_rpc("gdn_state_probe_finish", timeout=90)
        installed = False
        if len(result) != 1: raise RuntimeError("Expected exactly one report")
        manifest["worker"] = result[0]
        manifest["output_token_ids"] = [list(out.outputs[0].token_ids) for out in outputs]
        if result[0]["status"] != "complete": raise RuntimeError("Required observed long/mixed/decode coverage missing")
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if llm is not None:
            if installed:
                try: llm.collective_rpc("gdn_state_probe_remove", timeout=30)
                except Exception as exc: manifest["remove_error"] = str(exc)
            try: llm.llm_engine.engine_core.shutdown(timeout=60)
            except Exception as exc:
                manifest["shutdown_error"] = str(exc); manifest["status"] = "cleanup_failed"
        signal.signal(signal.SIGTERM, previous)
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["status"] != "complete": raise RuntimeError("Owned cleanup did not finish")
    print(json.dumps({"status": "STATE_PROBE_COMPLETE", "output_dir": str(directory),
                      "coverage": manifest["worker"]["coverage"], "timing_measured": False}))


if __name__ == "__main__": main()
