#!/usr/bin/env python3
"""Capture actual GDN activations in an owned, standalone eager vLLM process.

Default/--help never import torch or vLLM. --preflight reads local tokenizer
and source files on CPU. Only --run-gpu constructs an LLM or installs hooks.
This instrumentation performs blocking CPU copies and never measures latency.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import uuid

from capture_io import file_sha256

SNAPSHOT = Path("/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/"
                "snapshots/1c3f884bc99aac2524f6d49bcbac8c88401afd66")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-gpu", action="store_true")
    mode.add_argument("--preflight", action="store_true", help="CPU-only local prompt/API checks")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results/activations_8k")
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 18, 38])
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    args = parser.parse_args()
    if args.tokens != 8192 or sorted(args.layers) != [0, 18, 38]:
        parser.error("This capture is scoped to one 8192-token prefill and layers 0,18,38")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    args.layers.sort()
    args.snapshot = args.snapshot.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def local_prompt(snapshot, tokens):
    from tokenizers import Tokenizer
    import pathlib

    tokenizer_path = snapshot / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    sources = [Path(module.__file__) for module in (argparse, json, pathlib)]
    chunks, provenance = [], []
    for path in sources:
        chunks.append(f"\n# Python standard-library source: {path.name}\n" + path.read_text())
        provenance.append({"path": str(path), "sha256": file_sha256(path)})
    text = "Review these Python standard-library excerpts and explain their behavior.\n" + "\n".join(chunks)
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if len(ids) < tokens:
        raise ValueError("Local standard-library text is unexpectedly shorter than the requested prompt")
    ids = ids[:tokens]
    encoded = json.dumps(ids, separators=(",", ":")).encode()
    return ids, {"tokens": len(ids), "token_ids_sha256": hashlib.sha256(encoded).hexdigest(),
                 "tokenizer_sha256": file_sha256(tokenizer_path), "sources": provenance,
                 "format": "raw token IDs, no chat template or added special tokens"}


def api_preflight():
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("Use the pinned vLLM v0.32 container")
    package = Path(spec.origin).parent
    required = {
        "entrypoints/llm.py": ["def collective_rpc(", "compilation_config", "**kwargs"],
        "v1/worker/worker_base.py": ["worker_extension_cls", "worker_class.__bases__"],
        "v1/worker/gpu_model_runner.py": ["def get_model(self)"],
        "v1/engine/core_client.py": ["def shutdown(self, timeout:"],
        "model_executor/layers/layernorm.py": ["class RMSNormGated", "def forward_cuda("],
        "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py": ["def _output_projection(", "self.norm(core_attn_out, z)"],
        "third_party/flash_linear_attention/ops/layernorm_guard.py": ["def rmsnorm_fn("],
    }
    result = {}
    for relative, needles in required.items():
        path = package / relative
        source = path.read_text()
        if not all(needle in source for needle in needles):
            raise RuntimeError(f"Installed API differs from the inspected v0.32 contract: {relative}")
        result[relative] = file_sha256(path)
    here = Path(__file__).resolve().parent
    return {"vllm_version": importlib.metadata.version("vllm"), "source_sha256": result,
            "capture_source_sha256": {name: file_sha256(here / name) for name in
                                      ("capture_activations.py", "capture_worker.py", "capture_io.py")}}


def main():
    args = arguments()
    if not args.run_gpu and not args.preflight:
        print(json.dumps({"status": "PENDING_GPU", "message": "No torch/vLLM import or GPU work. Use --preflight for CPU checks; --run-gpu only when scheduled."}))
        return
    for filename in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
        if not (args.snapshot / filename).is_file():
            raise FileNotFoundError(args.snapshot / filename)
    ids, prompt_info = local_prompt(args.snapshot, args.tokens)
    api = api_preflight()
    if args.preflight:
        assert "torch" not in sys.modules and "vllm" not in sys.modules
        print(json.dumps({"status": "CPU_PREFLIGHT_PASS", "prompt": prompt_info, "api": api,
                          "torch_imported": False, "vllm_imported": False, "gpu_work": False}, sort_keys=True))
        return

    # All GPU-affecting imports and operations are below the explicit gate.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["GDN_ACTIVATION_CAPTURE"] = "1"
    here = str(Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = here + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    request = {"capture_id": str(uuid.uuid4()), "layers": args.layers, "tokens": args.tokens}
    (args.output_dir / "capture_request.json").write_text(json.dumps(request, indent=2) + "\n")
    (args.output_dir / "prompt_token_ids.json").write_text(json.dumps(ids) + "\n")
    engine_options = dict(model=str(args.snapshot), tokenizer=str(args.snapshot),
                          language_model_only=True, dtype="bfloat16", tensor_parallel_size=1,
                          enforce_eager=True, compilation_config=0, enable_prefix_caching=False,
                          enable_chunked_prefill=False, async_scheduling=False, speculative_config=None,
                          max_model_len=16384, max_num_batched_tokens=16384, max_num_seqs=1,
                          gpu_memory_utilization=args.gpu_memory_utilization,
                          distributed_executor_backend="uni",
                          worker_extension_cls="capture_worker.GDNCaptureWorkerExtension", seed=1729)
    manifest = {"schema_version": 1, "status": "incomplete", **request,
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                "snapshot": str(args.snapshot), "model_config_sha256": file_sha256(args.snapshot / "config.json"),
                "weight_index_sha256": file_sha256(args.snapshot / "model.safetensors.index.json"),
                "prompt": prompt_info, "api": api, "engine_options": engine_options,
                "scope": "Actual eager prefill activations; instrumentation only, no timing or compiled-serving equivalence claim.",
                "environment": {key: value for key, value in os.environ.items()
                                if key.startswith(("VLLM_", "GDN_")) or key == "CUDA_VISIBLE_DEVICES"}}
    llm = None
    hooks_installed = False

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    previous_sigterm = signal.signal(signal.SIGTERM, terminate)
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(**engine_options)
        installed = llm.collective_rpc("gdn_capture_install", timeout=60, args=(
            str(args.output_dir), request["capture_id"], args.layers, args.tokens))
        hooks_installed = True
        if len(installed) != 1:
            raise RuntimeError("Capture requires exactly one worker")
        manifest["installation"] = installed[0]
        outputs = llm.generate([{"prompt_token_ids": ids}], SamplingParams(
            temperature=0.0, max_tokens=1, min_tokens=1, ignore_eos=True), use_tqdm=False)
        if (len(outputs) != 1 or list(outputs[0].prompt_token_ids) != ids
                or not outputs[0].finished or len(outputs[0].outputs) != 1
                or len(outputs[0].outputs[0].token_ids) != 1):
            raise RuntimeError("The single 8192-token/one-output-token request did not complete exactly")
        reports = llm.collective_rpc("gdn_capture_finish", timeout=60)
        hooks_installed = False
        if len(reports) != 1 or reports[0]["status"] != "complete":
            raise RuntimeError(f"Incomplete capture: {reports}")
        manifest["captures"] = reports[0]["captures"]
        manifest["worker"] = {key: value for key, value in reports[0].items() if key != "captures"}
        manifest["output_token_ids"] = list(outputs[0].outputs[0].token_ids)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if llm is not None:
            try:
                if hooks_installed:
                    llm.collective_rpc("gdn_capture_remove", timeout=30)
            except Exception as exc:
                manifest["hook_cleanup_error"] = str(exc)
            try:
                # This handle owns only the engine created above; no process-name kills.
                llm.llm_engine.engine_core.shutdown(timeout=60)
            except Exception as exc:
                manifest["shutdown_error"] = str(exc)
                manifest["status"] = "cleanup_failed"
        signal.signal(signal.SIGTERM, previous_sigterm)
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["status"] != "complete":
        raise RuntimeError("Capture did not complete with owned engine cleanup; inspect manifest.json")
    print(json.dumps({"status": "CAPTURE_COMPLETE", "output_dir": str(args.output_dir),
                      "layers": args.layers, "tokens": args.tokens, "timing_measured": False}))


if __name__ == "__main__":
    main()
