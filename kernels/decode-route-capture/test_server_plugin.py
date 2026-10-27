"""CPU-only server lifecycle and metadata discovery checks."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from autotune_import import entries_sha256
import qwen_fi_control_plugin as plugin
from test_autotune_import import DOCUMENT, KEY


class ServerPluginTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.json"
        self.source.write_text(json.dumps(DOCUMENT))
        self.sha = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.out = self.root / "out"
        self.events = []
        environment = patch.dict("os.environ", {
            "QWEN_FI_IMPORT_JSON": str(self.source), "QWEN_FI_IMPORT_OUT": str(self.out),
            "QWEN_FI_IMPORT_SHA256": self.sha,
            "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": str(self.out / "cache/autotune_import")})
        environment.start()
        self.addCleanup(environment.stop)
        self.addCleanup(lambda: setattr(plugin, "_registered", False))
        self.addCleanup(lambda: setattr(plugin, "_owner", None))
        self.addCleanup(lambda: setattr(plugin, "_shutdown_done", False))
        plugin._registered, plugin._owner, plugin._shutdown_done = False, None, False

    def install_fakes(self, fail_shutdown=False):
        events = self.events
        warmup = ModuleType("vllm.model_executor.warmup.kernel_warmup")
        def tune(runner):
            events.append("original_tune")
            return "unchanged_model_output"
        warmup.flashinfer_autotune = tune

        class Worker:
            def shutdown(self):
                events.append("original_shutdown")
                return "cleaned"

        class Importer:
            def __init__(self, source, out, *, observe_search):
                events.append(("create_importer", observe_search))
                self.report = {"source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest()}

            def install(self, *, original_warmup):
                events.append("install_importer")
                def imported(runner):
                    events.append("before_tuning")
                    return original_warmup(runner)
                warmup.flashinfer_autotune = imported

            def finalize(self, *, stage):
                events.append(stage)
                if fail_shutdown and stage == "server_shutdown":
                    raise RuntimeError("effective cache drift")

            def write_report(self):
                pass

        workers = ModuleType("vllm.v1.worker.gpu_worker")
        workers.Worker = Worker
        modules = patch.dict("sys.modules", {warmup.__name__: warmup, workers.__name__: workers})
        modules.start()
        self.addCleanup(modules.stop)
        importer = patch.object(plugin, "_helpers", return_value=SimpleNamespace(AutotuneImport=Importer))
        importer.start()
        self.addCleanup(importer.stop)
        plugin.register()
        return warmup, Worker

    def test_frontend_registration_is_lazy_and_actual_worker_orders_load_before_tuning(self):
        warmup, worker = self.install_fakes()
        self.assertEqual(self.events, [])
        self.assertFalse(self.out.exists())
        self.assertEqual(warmup.flashinfer_autotune(object()), "unchanged_model_output")
        self.assertEqual(self.events[:4], [("create_importer", False), "install_importer",
                                          "before_tuning", "original_tune"])
        self.assertEqual(worker().shutdown(), "cleaned")
        self.assertEqual(self.events[-2:], ["server_shutdown", "original_shutdown"])

    def test_shutdown_verification_error_still_cleans_up_owned_worker(self):
        warmup, worker = self.install_fakes(fail_shutdown=True)
        warmup.flashinfer_autotune(object())
        with self.assertRaisesRegex(RuntimeError, "cache drift"):
            worker().shutdown()
        self.assertEqual(self.events[-1], "original_shutdown")
        self.assertEqual(plugin._owner.report["server_checkpoint"], "shutdown_verification_failed")

    def test_mount_only_entrypoint_is_discoverable_without_installing_a_package(self):
        root = Path(__file__).parent
        entries = [ep for dist in importlib.metadata.distributions(path=[str(root)])
                   for ep in dist.entry_points if ep.group == "vllm.general_plugins"]
        self.assertEqual([(x.name, x.value) for x in entries],
                         [("qwen_fi_control", "qwen_fi_control_plugin:register")])

    def test_post_gate_rejects_missing_shutdown_and_checks_source_effective_hashes(self):
        root = self.out / "autotune"
        root.mkdir(parents=True)
        (root / "source.json").write_bytes(self.source.read_bytes())
        report = {"server_checkpoint": "warmup_verified", "source_sha256": self.sha,
                  "reference_entries_sha256": entries_sha256({KEY: DOCUMENT[KEY]})}
        (root / "verification.json").write_text(json.dumps(report))
        (root / "server_startup_effective.json").write_text(json.dumps(DOCUMENT))
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            plugin.verify_artifacts(self.out, self.sha)
        self.assertEqual(plugin.verify_artifacts(self.out, self.sha, False)["entries"], 1)
        result = subprocess.run([sys.executable, "-I", str(Path(plugin.__file__).resolve()),
                                 str(self.out), "--source-sha256", self.sha, "--startup-only"],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report["server_checkpoint"] = "shutdown_verified"
        (root / "verification.json").write_text(json.dumps(report))
        altered = dict(DOCUMENT, **{KEY: ["Runner", 9]})
        (root / "server_shutdown_effective.json").write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, "effective tactic table"):
            plugin.verify_artifacts(self.out, self.sha)


if __name__ == "__main__":
    unittest.main()
