"""Mount-only vLLM general plugin; no route hooks or per-lookup serving observer.

The frontend registers lazy callbacks without creating an importer or output.
Only the worker that actually warms the model owns the import and snapshots.
"""
import functools
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path

_registered = False
_owner = None
_shutdown_done = False


@functools.lru_cache(maxsize=1)
def _helpers():
    # Works with python -I and does not trust cwd/PYTHONPATH for a sibling file.
    path = Path(__file__).resolve().with_name("autotune_import.py")
    spec = importlib.util.spec_from_file_location("_qwen_fi_autotune_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the explicit sibling autotune helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def register():
    global _registered
    source = os.environ.get("QWEN_FI_IMPORT_JSON")
    if _registered or not source:
        return
    out = os.environ.get("QWEN_FI_IMPORT_OUT")
    expected = os.environ.get("QWEN_FI_IMPORT_SHA256")
    if not out or not expected:
        raise RuntimeError("FI server import requires explicit OUT and SHA256")
    if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
        raise RuntimeError("FI reference JSON hash does not match the declared immutable source")
    expected_cache = str(Path(out) / "cache/autotune_import")
    # Supply this in the container environment too, before vLLM may cache envs.
    if os.environ.get("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR") != expected_cache:
        raise RuntimeError("FI server cache directory must be the explicit fresh per-run import path")
    warmup = importlib.import_module("vllm.model_executor.warmup.kernel_warmup")
    worker = importlib.import_module("vllm.v1.worker.gpu_worker").Worker
    original_warmup, original_shutdown = warmup.flashinfer_autotune, worker.shutdown

    @functools.wraps(original_warmup)
    def run(runner, *args, **kwargs):
        global _owner
        if _owner is not None:
            raise RuntimeError("FI server plugin supports one model warmup per worker process")
        _owner = _helpers().AutotuneImport(source, out, observe_search=False)
        if _owner.report["source_sha256"] != expected:
            raise RuntimeError("FI reference changed between plugin registration and worker warmup")
        _owner.report.update(server_worker_pid=os.getpid(), server_checkpoint="warmup_started",
                             serving_lookup_observer=False)
        _owner.install(original_warmup=original_warmup)
        try:
            result = warmup.flashinfer_autotune(runner, *args, **kwargs)
            _owner.finalize(stage="server_startup")
            _owner.report["server_checkpoint"] = "warmup_verified"
            _owner.write_report()
            return result
        except BaseException as ex:
            _owner.report.update(server_checkpoint="warmup_failed", error=f"{type(ex).__name__}: {ex}")
            _owner.write_report()
            raise

    @functools.wraps(original_shutdown)
    def shutdown(instance, *args, **kwargs):
        global _shutdown_done
        try:
            if _owner is not None and not _shutdown_done:
                _shutdown_done = True
                try:
                    _owner.finalize(stage="server_shutdown")
                    _owner.report["server_checkpoint"] = "shutdown_verified"
                except BaseException:
                    _owner.report["server_checkpoint"] = "shutdown_verification_failed"
                    raise
                finally:
                    _owner.write_report()
        finally:
            # Cleanup must still happen even when the effective cache drifted.
            result = original_shutdown(instance, *args, **kwargs)
        return result

    warmup.flashinfer_autotune, worker.shutdown = run, shutdown
    _registered = True


def verify_artifacts(out, expected_sha256, require_shutdown=True):
    """CPU post-run gate for the root launcher; never contacts a server/GPU."""
    helpers = _helpers()
    entries_sha256, tactic_entries = helpers.entries_sha256, helpers.tactic_entries
    root = Path(out) / "autotune"
    report = json.loads((root / "verification.json").read_text())
    stage = "server_shutdown" if require_shutdown else "server_startup"
    checkpoint = "shutdown_verified" if require_shutdown else "warmup_verified"
    if report.get("server_checkpoint") != checkpoint and not (
            not require_shutdown and report.get("server_checkpoint") == "shutdown_verified"):
        raise ValueError(f"missing verified {stage} checkpoint")
    if report["source_sha256"] != expected_sha256:
        raise ValueError("source SHA mismatch")
    if hashlib.sha256((root / "source.json").read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("saved source JSON SHA mismatch")
    effective = tactic_entries(json.loads((root / f"{stage}_effective.json").read_text()))
    if entries_sha256(effective) != report["reference_entries_sha256"]:
        raise ValueError("effective tactic table is not exactly the fixed reference (including keys)")
    return {"verified_stage": stage, "source_sha256": expected_sha256,
            "effective_entries_sha256": entries_sha256(effective), "entries": len(effective),
            "inductor_pinned": False, "serving_lookup_observer": False}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CPU verification of server FI import artifacts")
    parser.add_argument("out", type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--startup-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_artifacts(args.out, args.source_sha256, not args.startup_only), indent=2))
