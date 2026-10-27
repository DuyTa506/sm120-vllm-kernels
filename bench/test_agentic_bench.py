"""CPU-only regression tests; HTTP/SSE is mocked, no server or tools execute."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error


spec = importlib.util.spec_from_file_location(
    "agentic_bench", Path(__file__).with_name("agentic_bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def frame(text="", finish=None, usage=None):
    value = {"id": "cmpl-test", "object": "text_completion", "created": 1,
             "model": "bench", "choices": [{"index": 0, "text": text,
                 "logprobs": None, "finish_reason": finish}]}
    if usage is not None:
        value["usage"] = {"prompt_tokens": 2, "completion_tokens": usage,
                          "total_tokens": 2 + usage}
    return b"data: " + json.dumps(value).encode() + b"\n\n"


def trajectory(task=0, turns=1):
    return {"task": task, "turns": [
        {"prompt_ids": [3, 4], "completion_tokens": 2} for _ in range(turns)]}


class Stream(io.BytesIO):
    """Advance a fake clock at each SSE line instead of sleeping."""
    def __init__(self, data, now=None):
        super().__init__(data)
        self.now = now

    def __next__(self):
        line = super().__next__()
        if self.now is not None:
            self.now[0] += 1
        return line


class ReplayTests(unittest.TestCase):
    def test_empty_metadata_is_not_first_token_and_usage_counts_output(self):
        now, rows, bodies = [0], [], []
        def urlopen(req, timeout):
            bodies.append(json.loads(req.data))
            usage = b'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}}\n\n'
            return Stream(frame() + frame("hi") + frame(finish="length") + usage
                          + b"data: [DONE]\n\n", now)
        with patch.object(bench.urllib.request, "urlopen", side_effect=urlopen), \
                patch.object(bench.time, "time", side_effect=lambda: now[0]), \
                patch.object(bench.time, "perf_counter", side_effect=lambda: now[0]):
            bench.replay_agent(trajectory(), "unchanged-salt", rows)
        self.assertEqual(rows[0]["ttft"], 3)
        self.assertTrue(rows[0]["success"])
        self.assertEqual(rows[0]["completion"], 2)
        self.assertEqual(rows[0]["completion_source"], "usage")
        self.assertEqual(bodies[0]["cache_salt"], "unchanged-salt")
        self.assertEqual(bodies[0]["prompt"], [3, 4])
        self.assertTrue(bodies[0]["stream_options"]["include_usage"])

    def test_truncation_retains_partial_turn_and_raises(self):
        rows = []
        with patch.object(bench.urllib.request, "urlopen", return_value=Stream(frame("hi"))):
            with self.assertRaises(RuntimeError):
                bench.replay_agent(trajectory(), "salt", rows)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["success"])
        self.assertEqual(rows[0]["completion"], 0)
        self.assertEqual(rows[0]["requested_completion"], 2)
        self.assertIsNotNone(rows[0]["ttft"])

    def test_stream_error_and_short_usage_fail_instead_of_counting_requested_tokens(self):
        streams = [
            b'data: {"error":{"message":"engine failed"}}\n\ndata: [DONE]\n\n',
            frame("hi", "stop", 1) + b"data: [DONE]\n\n",
            frame("hi", "error", 2) + b"data: [DONE]\n\n",
            frame("hi", "length", 2),  # EOF even after usage is an incomplete stream.
        ]
        for data in streams:
            with self.subTest(data=data), patch.object(
                    bench.urllib.request, "urlopen", return_value=Stream(data)):
                rows = []
                with self.assertRaises(RuntimeError):
                    bench.replay_agent(trajectory(), "salt", rows)
                self.assertFalse(rows[0]["success"])
                # A failed stream may have server usage; it must still not be successful.
                self.assertLessEqual(rows[0]["completion"], 2)

    def test_length_finish_can_verify_count_without_usage(self):
        rows = []
        with patch.object(bench.urllib.request, "urlopen", return_value=Stream(
                frame("hi", "length") + b"data: [DONE]\n\n")):
            bench.replay_agent(trajectory(), "salt", rows)
        self.assertEqual(rows[0]["completion"], 2)
        self.assertEqual(rows[0]["completion_source"], "length_finish")

    def test_worker_failure_propagates_and_preserves_previous_concurrency(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "traj.json", Path(tmp) / "out.json"
            src.write_text(json.dumps({"trajectories": [trajectory()]}))
            calls = [0]
            def urlopen(req, timeout):
                calls[0] += 1
                if calls[0] == 1:
                    return Stream(frame("hi", "length", 2) + b"data: [DONE]\n\n")
                raise urllib.error.HTTPError(req.full_url, 503, "unavailable", {},
                                             io.BytesIO(b'{"error":"overloaded"}'))
            with patch.object(bench, "metrics", return_value=(0, 0)), \
                    patch.object(bench.urllib.request, "urlopen", side_effect=urlopen):
                with self.assertRaises(RuntimeError):
                    bench.replay(str(src), str(out), [1, 2])
            report = json.loads(out.read_text())
            self.assertTrue(report["1"]["summary"]["valid"])
            failed = report["2"]
            self.assertFalse(failed["summary"]["valid"])
            self.assertEqual(failed["summary"]["expected_turns"], 2)
            self.assertEqual(failed["summary"]["turns"], 0)
            self.assertEqual(failed["summary"]["out_tok_s"], 0)
            self.assertTrue(failed["errors"])
            self.assertIn("503", failed["turns"][0]["error"])

    def test_partial_task_counts_only_its_completed_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "traj.json", Path(tmp) / "out.json"
            src.write_text(json.dumps({"trajectories": [trajectory(turns=2)]}))
            responses = [Stream(frame("hi", "length", 2) + b"data: [DONE]\n\n"),
                         Stream(frame("partial"))]
            with patch.object(bench, "metrics", return_value=(0, 0)), \
                    patch.object(bench.urllib.request, "urlopen", side_effect=responses):
                with self.assertRaises(RuntimeError):
                    bench.replay(str(src), str(out), [1])
            result = json.loads(out.read_text())["1"]
            summary = result["summary"]
            self.assertEqual(len(result["turns"]), 2)
            self.assertEqual(summary["turns"], 1)
            self.assertEqual(summary["failed_turns"], 1)
            self.assertEqual(summary["agents"], 0)
            self.assertEqual(summary["not_run_turns"], 0)
            self.assertAlmostEqual(summary["out_tok_s"] * summary["wall_s"], 2)

    def test_interruption_checkpoints_completed_turns_and_marks_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "traj.json", Path(tmp) / "out.json"
            src.write_text(json.dumps({"trajectories": [trajectory()]}))
            with patch.object(bench, "metrics", side_effect=[(0, 0), KeyboardInterrupt]), \
                    patch.object(bench.urllib.request, "urlopen", return_value=Stream(
                        frame("hi", "length", 2) + b"data: [DONE]\n\n")):
                with self.assertRaises(KeyboardInterrupt):
                    bench.replay(str(src), str(out), [1])
            result = json.loads(out.read_text())["1"]
            self.assertEqual(result["summary"]["turns"], 1)
            self.assertFalse(result["summary"]["valid"])
            self.assertIn("KeyboardInterrupt", result["errors"][0]["error"])


if __name__ == "__main__":
    unittest.main()
