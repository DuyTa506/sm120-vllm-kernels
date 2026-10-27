"""CPU-only cache API mocks: no CUDA, torch, vLLM or real FlashInfer imports."""
import contextlib
import hashlib
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from autotune_import import AutotuneImport, compiler_config_audit


KEY = "('op', 'Runner', ((1, 2),), ())"
DOCUMENT = {"_metadata": {"flashinfer_version": "test"}, "_generation": "test",
            KEY: ["Runner", 4]}


class Runner:
    def get_cache_key_extras(self, inputs):
        return ()


class AutotuneImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "control.json"
        self.source.write_text(json.dumps(DOCUMENT))

    def setup_fake_runtime(self, imported, *, accept=True, retune=False, missing=False,
                           cold_policy=False, fixed_reference=False, install=True):
        class Tuner:
            def __init__(self):
                self._lock = contextlib.nullcontext()
                self._file_configs, self.profiling_cache = {}, {}
                self.is_tuning_mode = False

            def _winner_cache(self):
                return self.profiling_cache

            def load_configs(self, path):
                self.loaded_path = Path(path)
                if accept:
                    self._file_configs.update({KEY: ("Runner", 4)})
                return accept

            def search_cache(self, *args):
                if cold_policy and self.is_tuning_mode:
                    # Installed FI ignores legacy v1 file entries under cold-L2
                    # tuning, even when their environment metadata matches.
                    self.profiling_cache[Key()] = (9, None)
                    return (True, 0, 9, None)
                return (not missing, 0, 4, None)

        tuner = Tuner()
        class Key:
            file_key, runner_class_name = KEY, "Runner"

        class API:
            get = staticmethod(lambda: tuner)
            _get_cache_key = staticmethod(lambda *args: Key())

        path = imported.cache_root / "hash/autotune_configs.json"
        path.parent.mkdir(parents=True)
        warmup = ModuleType("vllm.model_executor.warmup.kernel_warmup")
        warmup.resolve_flashinfer_autotune_file = lambda runner: path
        fi_utils = ModuleType("vllm.utils.flashinfer")

        @contextlib.contextmanager
        def autotune(tune_mode=True, **kwargs):
            previous = tuner.is_tuning_mode
            tuner.is_tuning_mode = previous or tune_mode
            try:
                yield
            finally:
                tuner.is_tuning_mode = previous

        fi_utils.autotune = autotune

        def run(runner):
            resolved = path.with_name("autotune_configs_dp0_rank0.json")
            tuner.load_configs(str(resolved))
            with fi_utils.autotune(tune_mode=True):
                with fi_utils.autotune(tuning_buckets=(1, 2), round_up=True):
                    result = tuner.search_cache("op", [Runner()], ((1, 2),), object())
            if retune:
                tuner.profiling_cache[Key()] = (9, None)
            return result

        warmup.flashinfer_autotune = run
        mods = {name: ModuleType(name) for name in ["flashinfer", "flashinfer.autotuner",
                "flashinfer.autotuner.autotuner", "vllm", "vllm.distributed",
                "vllm.distributed.parallel_state"]}
        mods["flashinfer.autotuner"].AutoTuner = API
        mods["flashinfer.autotuner.autotuner"]._tactic_to_json = lambda x: x
        mods["vllm.distributed.parallel_state"].get_world_group = lambda: SimpleNamespace(
            world_size=1, rank_in_group=0)
        mods[warmup.__name__] = warmup
        mods[fi_utils.__name__] = fi_utils
        patcher = patch.dict("sys.modules", mods)
        patcher.start()
        self.addCleanup(patcher.stop)
        environment = patch.dict("os.environ", {}, clear=False)
        environment.start()
        self.addCleanup(environment.stop)
        if install:
            imported.install(fixed_reference=fixed_reference)
        runner = SimpleNamespace(vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(
            data_parallel_size=1, data_parallel_rank=0)))
        return lambda: warmup.flashinfer_autotune(runner), tuner

    def test_native_cold_l2_tuning_retunes_imported_legacy_file(self):
        imported = AutotuneImport(self.source, self.root / "run")
        run, _ = self.setup_fake_runtime(imported, cold_policy=True)
        with self.assertRaisesRegex(RuntimeError, "after_warmup"):
            run()
        self.assertEqual(imported.report["checks"]["loaded"]["changed"], [])

    def test_fixed_reference_replays_native_cache_without_policy_retuning(self):
        imported = AutotuneImport(self.source, self.root / "run", observe_search=False)
        run, tuner = self.setup_fake_runtime(imported, cold_policy=True, fixed_reference=True)
        imported.preload()
        self.assertEqual(imported.report["checks"]["preloaded"]["changed"], [])
        self.assertEqual(run(), (True, 0, 4, None))
        self.assertEqual(tuner.profiling_cache, {})
        self.assertFalse(tuner.is_tuning_mode)
        self.assertEqual(imported.finalize()["status"],
                         "reference_tactics_verified_without_lookup_observation")
        import importlib
        fi_utils = importlib.import_module("vllm.utils.flashinfer")
        with fi_utils.autotune(tune_mode=True):
            self.assertTrue(tuner.is_tuning_mode)

    def test_early_preload_refuses_preexisting_winners_without_clearing_them(self):
        imported = AutotuneImport(self.source, self.root / "run")
        _, tuner = self.setup_fake_runtime(imported, fixed_reference=True)
        tuner.profiling_cache[object()] = (9, None)
        with self.assertRaisesRegex(RuntimeError, "before any existing"):
            imported.preload()
        self.assertEqual(len(tuner.profiling_cache), 1)

    def test_fixed_replay_restores_native_autotune_after_failed_verification(self):
        import importlib
        imported = AutotuneImport(self.source, self.root / "run")
        run, tuner = self.setup_fake_runtime(imported, fixed_reference=True, retune=True)
        with self.assertRaisesRegex(RuntimeError, "after_warmup"):
            run()
        fi_utils = importlib.import_module("vllm.utils.flashinfer")
        with fi_utils.autotune(tune_mode=True):
            self.assertTrue(tuner.is_tuning_mode)

    def test_early_preload_rejects_metadata_failure(self):
        imported = AutotuneImport(self.source, self.root / "run")
        self.setup_fake_runtime(imported, fixed_reference=True, accept=False)
        with self.assertRaisesRegex(RuntimeError, "rejected the early"):
            imported.preload()
        self.assertFalse(imported.report["preload"]["returned"])

    def test_real_api_flow_loads_only_json_then_verifies_effective_and_observed_tactics(self):
        original = self.source.read_bytes()
        imported = AutotuneImport(self.source, self.root / "run", True)
        run, tuner = self.setup_fake_runtime(imported)
        self.assertEqual(run(), (True, 0, 4, None))
        report = imported.finalize()
        self.assertEqual(report["status"], "reference_and_observed_tactics_verified")
        self.assertTrue(report["load_calls"][0]["returned"])
        self.assertEqual(tuner.loaded_path.name, "autotune_configs_dp0_rank0.json")
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(list(imported.cache_root.rglob("*json")), [tuner.loaded_path])
        self.assertFalse(report["compiled_graph_cache_imported"])

    def test_rejected_load_does_not_silently_retune_or_pass_final_verification(self):
        imported = AutotuneImport(self.source, self.root / "run")
        run, _ = self.setup_fake_runtime(imported, accept=False)
        with self.assertRaisesRegex(RuntimeError, "successfully load"):
            run()
        with self.assertRaises(RuntimeError):
            imported.finalize()
        self.assertEqual(imported.report["status"], "verification_failed")

    def test_actual_importer_preserves_reference_identity_through_server_gates(self):
        from qwen_fi_control_plugin import verify_artifacts
        expected = hashlib.sha256(self.source.read_bytes()).hexdigest()
        out = self.root / "run"
        imported = AutotuneImport(self.source, out, observe_search=False)
        run, tuner = self.setup_fake_runtime(imported, fixed_reference=True)
        imported.preload()
        self.assertEqual(run(), (True, 0, 4, None))
        self.assertEqual(tuner.search_cache.__name__, "search_cache")
        for stage, checkpoint, require_shutdown in (
                ("server_startup", "warmup_verified", False),
                ("server_shutdown", "shutdown_verified", True)):
            imported.finalize(stage=stage)
            imported.report["server_checkpoint"] = checkpoint
            imported.report["server_startup_boundary"] = "after_compile_or_warm_up_model"
            imported.write_report()
            checked = verify_artifacts(out, expected, require_shutdown)
            self.assertEqual(checked["source_sha256"], expected)
            self.assertEqual(checked["verified_stage"], stage)
            self.assertEqual(checked["entries"], 1)
        self.assertIsInstance(imported.report["runtime_source_sha256"], dict)

    def test_live_winner_overriding_loaded_tactic_is_detected(self):
        imported = AutotuneImport(self.source, self.root / "run")
        run, _ = self.setup_fake_runtime(imported, retune=True)
        with self.assertRaisesRegex(RuntimeError, "after_warmup"):
            run()
        self.assertEqual(imported.report["checks"]["after_warmup"]["changed"][0]["actual"], ["Runner", 9])

    def test_uncovered_lookups_are_explicit_and_strict_mode_fails(self):
        imported = AutotuneImport(self.source, self.root / "run", True)
        run, _ = self.setup_fake_runtime(imported, missing=True)
        run()
        with self.assertRaisesRegex(RuntimeError, "fully covered"):
            imported.finalize()
        self.assertFalse(imported.report["all_observed_lookups_covered"])

    def test_import_does_not_overwrite_an_existing_rank_cache(self):
        imported = AutotuneImport(self.source, self.root / "run")
        run, _ = self.setup_fake_runtime(imported)
        path = imported.cache_root / "hash/autotune_configs_dp0_rank0.json"
        path.write_text("existing")
        with self.assertRaises(FileExistsError):
            run()
        self.assertEqual(path.read_text(), "existing")

    def test_compiler_audit_separates_selected_config_from_timing_metadata(self):
        path = self.root / "kernel.best_config"
        path.write_text(json.dumps({"R0_BLOCK": 1024, "num_warps": 8, "time_taken_ms": 1}))
        first = compiler_config_audit(self.root)["files"][0]
        path.write_text(json.dumps({"R0_BLOCK": 1024, "num_warps": 8, "time_taken_ms": 9}))
        second = compiler_config_audit(self.root)["files"][0]
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["selected_sha256"], second["selected_sha256"])


if __name__ == "__main__":
    unittest.main()
