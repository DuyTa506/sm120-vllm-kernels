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
import run_capture


class DriverTests(unittest.TestCase):
    def invoke(self, *, generation_error=False, shutdown_error=False, short_batch=False):
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
                    (path / "manifest.json").write_text(json.dumps(kwargs))

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
            vllm = SimpleNamespace(LLM=lambda **kwargs: llm, SamplingParams=lambda **kwargs: kwargs)
            with patch.dict("sys.modules", {"vllm": vllm}), \
                 patch.dict("os.environ", {}, clear=False), \
                 patch.object(capture_runtime, "V2RouteCapture", Capture), \
                 patch.object(run_capture, "select_prompts", return_value=([{"prompt_token_ids": [1]}], [])), \
                 patch("sys.argv", ["run_capture.py", "--out", str(out), "--output-tokens", "32"]), \
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


if __name__ == "__main__":
    unittest.main()
