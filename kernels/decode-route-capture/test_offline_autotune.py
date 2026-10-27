"""CPU-only early offline load, export ordering and lifecycle verification."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from autotune_import import entries_sha256, tactic_entries
from offline_autotune import OfflineAutotuneControl, verify_offline_artifacts
from test_autotune_import import DOCUMENT
import test_autotune_import


class OfflineAutotuneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.source = self.out / "reference.json"
        self.source.write_text(json.dumps(DOCUMENT))
        self.events = []

    def fake_control(self, fail_stage=None, fail_shutdown=False):
        events = self.events
        module = ModuleType("vllm.v1.worker.gpu_worker")

        class Worker:
            def load_model(self):
                events.append("model_load")
                return "loaded"

            def compile_or_warm_up_model(self):
                events.extend(["warmup", "graph_capture"])
                return "compiled"

            def shutdown(self):
                events.append("worker_cleanup")
                if fail_shutdown:
                    raise RuntimeError("worker cleanup failed")
                return "cleaned"

        class Importer:
            def __init__(self, source, out, require_coverage):
                self.report = {}

            def install(self, *, fixed_reference):
                events.append(("install", fixed_reference))

            def preload(self):
                events.append("preload")

            def finalize(self, stage):
                events.append(stage)
                if stage == fail_stage:
                    raise RuntimeError("tactic audit failed")

            def write_report(self):
                pass

        module.Worker = Worker
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict("sys.modules", {module.__name__: module}))
        stack.enter_context(patch.dict(os.environ, {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}))
        stack.enter_context(patch("offline_autotune.AutotuneImport", Importer))
        control = OfflineAutotuneControl(self.source, self.out, True)
        control.install()
        return control, Worker()

    def test_early_load_and_after_capture_snapshot_preserve_native_results(self):
        control, worker = self.fake_control()
        self.assertEqual(worker.load_model(), "loaded")
        self.assertEqual(worker.compile_or_warm_up_model(), "compiled")
        self.events.append("raw_route_export")
        self.assertEqual(worker.shutdown(), "cleaned")
        self.assertEqual(self.events, [("install", True), "preload", "model_load",
            "warmup", "graph_capture", "offline_startup", "raw_route_export",
            "offline_shutdown", "worker_cleanup"])
        self.assertEqual(control.imported.report["offline_checkpoint"], "shutdown_verified")
        self.assertEqual(control.imported.report["offline_worker_shutdown"], "completed")

    def test_tactic_failure_does_not_skip_owned_worker_cleanup(self):
        control, worker = self.fake_control(fail_stage="offline_shutdown")
        worker.load_model()
        worker.compile_or_warm_up_model()
        with self.assertRaisesRegex(RuntimeError, "tactic audit"):
            worker.shutdown()
        self.assertEqual(self.events[-1], "worker_cleanup")
        self.assertEqual(control.imported.report["offline_checkpoint"], "shutdown_verification_failed")

    def test_native_cleanup_failure_cannot_claim_complete_lifecycle(self):
        control, worker = self.fake_control(fail_shutdown=True)
        worker.load_model()
        worker.compile_or_warm_up_model()
        with self.assertRaisesRegex(RuntimeError, "worker cleanup"):
            worker.shutdown()
        self.assertEqual(control.imported.report["offline_worker_shutdown"], "failed")

    def test_actual_importer_and_offline_wrapper_pass_every_boundary_with_cold_policy(self):
        control = OfflineAutotuneControl(self.source, self.out, True)
        fixture = test_autotune_import.AutotuneImportTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        native_run, tuner = fixture.setup_fake_runtime(
            control.imported, cold_policy=True, install=False)
        module = ModuleType("vllm.v1.worker.gpu_worker")

        class Worker:
            def load_model(self):
                self.loaded_with_reference = bool(tuner._file_configs)

            def compile_or_warm_up_model(self):
                return native_run()

            def shutdown(self):
                return "cleaned"

        module.Worker = Worker
        with patch.dict("sys.modules", {module.__name__: module}), \
             patch.dict(os.environ, {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}):
            control.install()
            worker = Worker()
            worker.load_model()
            self.assertTrue(worker.loaded_with_reference)
            self.assertEqual(worker.compile_or_warm_up_model(), (True, 0, 4, None))
            self.assertEqual(control.verify(False)["status"], "verified")
            control.imported.finalize()
            self.assertEqual(worker.shutdown(), "cleaned")
            (self.out / "shutdown.json").write_text(json.dumps({"status": "completed"}))
            self.assertEqual(control.verify()["status"], "verified")
        self.assertEqual(tuner.profiling_cache, {})

    def artifact_fixture(self):
        root = self.out / "autotune"
        root.mkdir()
        raw = self.source.read_bytes()
        (root / "source.json").write_bytes(raw)
        entries = tactic_entries(DOCUMENT)
        report = {"source_sha256": hashlib.sha256(raw).hexdigest(),
            "reference_entries_sha256": entries_sha256(entries),
            "import_mode": "fixed_reference_replay_v1",
            "preload": {"returned": True, "boundary": "before_worker_load_model"},
            "load_calls": [{"returned": True}],
            "offline_startup_boundary": "after_compile_or_warm_up_model",
            "offline_checkpoint": "shutdown_verified", "offline_worker_shutdown": "completed",
            "checks": {}, "require_all_observed_keys": True,
            "all_observed_lookups_covered": True,
            "observed_reference_mismatches": []}
        for stage in ("preloaded", "loaded", "after_warmup", "offline_startup",
                      "after_generation", "offline_shutdown"):
            report["checks"][stage] = {"missing": [], "changed": [], "extra": [],
                                       "entries_sha256": entries_sha256(entries)}
            (root / f"{stage}_tactics.json").write_text(json.dumps(entries))
        for stage in ("offline_startup", "after_generation", "offline_shutdown"):
            (root / f"{stage}_effective.json").write_text(json.dumps(DOCUMENT))
        (root / "verification.json").write_text(json.dumps(report))
        (self.out / "shutdown.json").write_text(json.dumps({"status": "completed"}))
        return root, report

    def test_offline_gate_requires_both_boundaries_and_exact_effective_tables(self):
        root, report = self.artifact_fixture()
        checked = verify_offline_artifacts(self.out)
        self.assertEqual(checked["entries"], 1)
        self.assertEqual(checked["status"], "verified")
        report["offline_checkpoint"] = "startup_verified"
        (root / "verification.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "shutdown checkpoint"):
            verify_offline_artifacts(self.out)

    def test_offline_gate_rejects_tactic_drift_even_with_successful_report(self):
        root, _ = self.artifact_fixture()
        changed = dict(DOCUMENT)
        key = next(k for k in changed if not k.startswith("_"))
        changed[key] = ["Runner", 99]
        (root / "offline_shutdown_effective.json").write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "effective"):
            verify_offline_artifacts(self.out)


if __name__ == "__main__":
    unittest.main()
