"""CPU checks of failure/export/shutdown ordering using a fake owned engine."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import capture_runtime
import offline_autotune
import run_capture


class DriverTests(unittest.TestCase):
    def invoke(self, *, generation_error=False, shutdown_error=False, short_batch=False,
               native_aux=False, mismatch=False, autotune=False, tactic_shutdown_error=False):
        events = []
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)

            class Capture:
                def __init__(self, *args):
                    self.manifest = {}

                def install(self):
                    events.append("install")

                def export(self, path, **kwargs):
                    events.append("export")
                    (path / "manifest.json").write_text(json.dumps(dict(self.manifest, **kwargs)))

            def shutdown():
                events.append("shutdown")
                if shutdown_error:
                    raise RuntimeError("shutdown failed")

            def generate(*args):
                events.append("generate")
                if generation_error:
                    raise RuntimeError("generation failed")
                if short_batch:
                    return []
                return [SimpleNamespace(request_id="0", outputs=[SimpleNamespace(
                    token_ids=[7] * 32, finish_reason="length")])]

            llm = SimpleNamespace(generate=generate,
                                  llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=shutdown)))
            def construct(**kwargs):
                self.assertEqual(kwargs["aux_output_config"]["enable_return_routed_experts"], True)
                return llm
            vllm = SimpleNamespace(LLM=construct, SamplingParams=lambda **kwargs: kwargs)
            extra = ["--native-aux-control"] if native_aux else []
            class Control:
                def __init__(self, *args):
                    def finalize():
                        events.append("after_generation_tactics")
                    self.imported = SimpleNamespace(finalize=finalize, report={
                        "status": "reference_and_observed_tactics_verified", "source_sha256": "test",
                        "import_mode": "fixed_reference_replay_v1"})

                def install(self):
                    events.append("install_early_replay")

                def verify(self, require_shutdown=True):
                    events.append("verify_shutdown_tactics" if require_shutdown else "verify_startup_tactics")
                    if require_shutdown and tactic_shutdown_error:
                        raise RuntimeError("shutdown tactic drift")
                    return {"status": "verified"}

            if autotune:
                extra += ["--autotune-json", str(out / "reference_tactics.json"),
                          "--require-autotune-coverage"]
            if mismatch:
                (out / "reference.json").write_text(json.dumps({"selected_prompts": [],
                    "requests": [{"output_token_ids": [8] * 32}]}))
                extra += ["--compare-output", str(out / "reference.json")]
            with patch.dict("sys.modules", {"vllm": vllm}), \
                 patch.dict("os.environ", {}, clear=False), \
                 patch.object(capture_runtime, "V2RouteCapture", Capture), \
                 patch.object(offline_autotune, "OfflineAutotuneControl", Control), \
                 patch.object(run_capture, "select_prompts", return_value=([{"prompt_token_ids": [1]}], [])), \
                 patch("sys.argv", ["run_capture.py", "--out", str(out), "--output-tokens", "32", *extra]), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                error = None
                try:
                    run_capture.main()
                except RuntimeError as ex:
                    error = str(ex)
            manifest = json.loads((out / "manifest.json").read_text())
            shutdown_record = json.loads((out / "shutdown.json").read_text()) if (out / "shutdown.json").exists() else None
        return events, error, manifest, shutdown_record

    def test_exception_preserves_partial_artifacts_before_owned_engine_shutdown(self):
        events, error, manifest, shutdown = self.invoke(generation_error=True)
        self.assertEqual(events, ["install", "generate", "export", "shutdown"])
        self.assertEqual(error, "generation failed")
        self.assertFalse(manifest["success"])
        self.assertEqual(shutdown["status"], "completed")

    def test_cleanup_error_does_not_hide_original_generation_error(self):
        events, error, _, shutdown = self.invoke(generation_error=True, shutdown_error=True)
        self.assertEqual(events[-2:], ["export", "shutdown"])
        self.assertEqual(error, "generation failed")
        self.assertEqual(shutdown["status"], "failed")

    def test_success_exports_before_shutdown(self):
        events, error, manifest, _ = self.invoke()
        self.assertEqual(events[-2:], ["export", "shutdown"])
        self.assertIsNone(error)
        self.assertTrue(manifest["success"])

    def test_incomplete_request_batch_is_not_success(self):
        _, error, manifest, _ = self.invoke(short_batch=True)
        self.assertIsNotNone(error)
        self.assertFalse(manifest["success"])

    def test_native_aux_control_enables_aux_without_probe_install_or_export(self):
        events, error, manifest, _ = self.invoke(native_aux=True)
        self.assertEqual(events, ["generate", "shutdown"])
        self.assertIsNone(error)
        self.assertEqual(manifest["mode"], "native_aux_control")

    def test_exact_mismatch_still_requests_structural_validation(self):
        _, error, manifest, _ = self.invoke(mismatch=True)
        self.assertIn("differs from control", error)
        self.assertFalse(manifest["success"])
        self.assertFalse(manifest["control_token_ids_match"])
        self.assertTrue(manifest["validate_structure"])
        self.assertEqual(manifest["exact_output_status"], "failed")

    def test_offline_replay_gates_before_generation_and_after_raw_export_shutdown(self):
        events, error, manifest, _ = self.invoke(autotune=True)
        self.assertIsNone(error)
        self.assertEqual(events, ["install_early_replay", "install", "verify_startup_tactics",
            "generate", "after_generation_tactics", "export", "shutdown", "verify_shutdown_tactics"])
        self.assertEqual(manifest["autotune_import"]["lifecycle"]["status"], "verified")

    def test_final_tactic_failure_marks_run_failed_after_raw_export(self):
        events, error, manifest, _ = self.invoke(autotune=True, tactic_shutdown_error=True)
        self.assertIn("shutdown tactic drift", error)
        self.assertLess(events.index("export"), events.index("verify_shutdown_tactics"))
        self.assertFalse(manifest["success"])
        self.assertEqual(manifest["autotune_import"]["lifecycle"]["status"], "failed")

    def test_exact_failure_remains_primary_when_final_tactic_audit_also_fails(self):
        _, error, manifest, _ = self.invoke(autotune=True, mismatch=True, tactic_shutdown_error=True)
        self.assertIn("differs from control", error)
        self.assertIn("differs from control", manifest["error"])
        self.assertTrue(manifest["validate_structure"])
        self.assertEqual(manifest["exact_output_status"], "failed")
        self.assertEqual(manifest["autotune_import"]["lifecycle"]["status"], "failed")

    def test_successful_output_with_cleanup_failure_is_not_a_successful_run(self):
        _, error, manifest, _ = self.invoke(shutdown_error=True)
        self.assertIn("shutdown failed", error)
        self.assertFalse(manifest["success"])


if __name__ == "__main__":
    unittest.main()
