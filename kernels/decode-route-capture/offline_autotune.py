"""Offline-only fixed-reference lifecycle; serving plugin sources stay unchanged.

This diagnostic keeps the imported helper's read-only lookup observer enabled.
It is instrumentation and must never supply latency baseline measurements.
"""
import functools
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path


_spec = importlib.util.spec_from_file_location(
    "_qwen_offline_autotune_helpers", Path(__file__).resolve().with_name("autotune_import.py"))
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load the explicit sibling autotune helper")
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
AutotuneImport = _helpers.AutotuneImport


def verify_offline_artifacts(out, require_shutdown=True):
    """CPU gate; table identity and coverage stay separate from output equality."""
    out = Path(out)
    root = out / "autotune"
    report = json.loads((root / "verification.json").read_text())
    source_bytes = (root / "source.json").read_bytes()
    source = json.loads(source_bytes)
    reference = _helpers.tactic_entries(source)
    digest = _helpers.entries_sha256(reference)
    if (hashlib.sha256(source_bytes).hexdigest() != report["source_sha256"]
            or digest != report["reference_entries_sha256"]):
        raise ValueError("saved source identity differs from the declared reference")
    if report.get("import_mode") != "fixed_reference_replay_v1":
        raise ValueError("offline run lacks explicit fixed-reference replay mode")
    preload = report.get("preload", {})
    if preload.get("returned") is not True or preload.get("boundary") != "before_worker_load_model":
        raise ValueError("offline run lacks successful early preload")
    if (not report.get("load_calls")
            or not all(call.get("returned") is True for call in report["load_calls"])):
        raise ValueError("actual rank cache was not successfully loaded")
    if report.get("offline_startup_boundary") != "after_compile_or_warm_up_model":
        raise ValueError("offline run lacks post-capture startup boundary")
    if require_shutdown:
        if report.get("offline_checkpoint") != "shutdown_verified":
            raise ValueError("offline run lacks verified shutdown checkpoint")
        shutdown = json.loads((out / "shutdown.json").read_text())
        if shutdown.get("status") != "completed" or report.get("offline_worker_shutdown") != "completed":
            raise ValueError("owned engine/worker shutdown did not complete")
    elif report.get("offline_checkpoint") not in ("startup_verified", "shutdown_verified"):
        raise ValueError("offline run lacks verified startup checkpoint")
    stages = ["preloaded", "loaded", "after_warmup", "offline_startup"]
    effective_stages = ["offline_startup"]
    if require_shutdown:
        stages += ["after_generation", "offline_shutdown"]
        effective_stages += ["after_generation", "offline_shutdown"]
    for stage in stages:
        check = report.get("checks", {}).get(stage, {})
        entries = json.loads((root / f"{stage}_tactics.json").read_text())
        if (check.get("entries_sha256") != digest or entries != reference
                or any(check.get(key, ["missing evidence"]) for key in ("missing", "changed", "extra"))):
            raise ValueError(f"offline {stage} tactics differ from the fixed reference")
    for stage in effective_stages:
        effective = json.loads((root / f"{stage}_effective.json").read_text())
        if (_helpers.tactic_entries(effective) != reference
                or effective["_metadata"] != source["_metadata"]):
            raise ValueError(f"offline {stage} effective table differs from reference")
    if report.get("observed_reference_mismatches"):
        raise ValueError("observed FI choices differ from the fixed reference")
    covered = report.get("all_observed_lookups_covered") is True
    if report.get("require_all_observed_keys") and not covered:
        raise ValueError("required observed FI lookup coverage is incomplete")
    return {"status": "verified", "verified_stage": "offline_shutdown" if require_shutdown else "offline_startup",
            "import_mode": report["import_mode"], "source_sha256": report["source_sha256"],
            "effective_entries_sha256": digest, "entries": len(reference),
            "all_observed_lookups_covered": covered, "inductor_pinned": False,
            "instrumentation_only": True}


class OfflineAutotuneControl:
    def __init__(self, source, out, require_coverage=False):
        self.out = Path(out)
        self.imported = AutotuneImport(source, out, require_coverage)
        self.worker = None
        self.shutdown_done = False

    def install(self):
        if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
            raise RuntimeError("offline FI control requires the owned in-process engine")
        if os.environ.get("QWEN_FI_IMPORT_JSON"):
            raise RuntimeError("offline and serving FI lifecycle plugins cannot be combined")
        worker = importlib.import_module("vllm.v1.worker.gpu_worker").Worker
        original_load = worker.load_model
        original_compile = worker.compile_or_warm_up_model
        original_shutdown = worker.shutdown
        owner, imported = self, self.imported
        imported.install(fixed_reference=True)
        imported.report.update(offline_checkpoint="installed_before_model_load",
                               offline_lookup_observer=True)
        imported.write_report()

        @functools.wraps(original_load)
        def load_model(instance, *args, **kwargs):
            if owner.worker is not None:
                raise RuntimeError("offline FI control expects exactly one owned worker load")
            owner.worker = instance
            try:
                imported.preload()
                result = original_load(instance, *args, **kwargs)
                imported.report["offline_checkpoint"] = "model_loaded_with_reference"
                return result
            except BaseException as ex:
                imported.report.update(offline_checkpoint="model_load_failed",
                                       offline_error=f"{type(ex).__name__}: {ex}")
                raise
            finally:
                imported.write_report()

        @functools.wraps(original_compile)
        def compile_or_warm_up_model(instance, *args, **kwargs):
            if owner.worker is not instance:
                raise RuntimeError("offline FI warmup requires the preloaded owned worker")
            try:
                result = original_compile(instance, *args, **kwargs)
                imported.finalize(stage="offline_startup")
                imported.report.update(offline_checkpoint="startup_verified",
                    offline_startup_boundary="after_compile_or_warm_up_model")
                return result
            except BaseException as ex:
                imported.report.update(offline_checkpoint="startup_failed",
                                       offline_error=f"{type(ex).__name__}: {ex}")
                raise
            finally:
                imported.write_report()

        @functools.wraps(original_shutdown)
        def shutdown(instance, *args, **kwargs):
            try:
                if owner.worker is instance and not owner.shutdown_done:
                    owner.shutdown_done = True
                    try:
                        imported.finalize(stage="offline_shutdown")
                        imported.report["offline_checkpoint"] = "shutdown_verified"
                    except BaseException as ex:
                        imported.report.update(offline_checkpoint="shutdown_verification_failed",
                                               offline_error=f"{type(ex).__name__}: {ex}")
                        raise
                    finally:
                        imported.write_report()
            finally:
                # Raw route export occurs in the driver before this boundary.
                try:
                    result = original_shutdown(instance, *args, **kwargs)
                    imported.report["offline_worker_shutdown"] = "completed"
                except BaseException as ex:
                    imported.report.update(offline_worker_shutdown="failed",
                                           offline_shutdown_error=f"{type(ex).__name__}: {ex}")
                    raise
                finally:
                    imported.write_report()
            return result

        worker.load_model = load_model
        worker.compile_or_warm_up_model = compile_or_warm_up_model
        worker.shutdown = shutdown

    def verify(self, require_shutdown=True):
        name = "lifecycle_verification.json" if require_shutdown else "startup_verification.json"
        try:
            result = verify_offline_artifacts(self.out, require_shutdown)
        except BaseException as ex:
            result = {"status": "failed", "error": f"{type(ex).__name__}: {ex}",
                      "instrumentation_only": True}
            (self.out / "autotune" / name).write_text(json.dumps(result, indent=2) + "\n")
            raise
        (self.out / "autotune" / name).write_text(json.dumps(result, indent=2) + "\n")
        return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CPU gate for offline fixed-reference route controls")
    parser.add_argument("out", type=Path)
    parser.add_argument("--startup-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_offline_artifacts(args.out, not args.startup_only), indent=2))
