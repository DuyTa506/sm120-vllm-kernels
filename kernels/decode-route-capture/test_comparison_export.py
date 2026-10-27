"""CPU checks: output variation is not equivalence; failures retain route analysis."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from capture_runtime import V2RouteCapture
from compare_outputs import compare_outputs
from test_analysis import GEOMETRY, draft_prefill, target


def output(ids):
    return {"selected_prompts": [{"task": 0}], "requests": [{"output_token_ids": ids}]}


class ComparisonExportTests(unittest.TestCase):
    def test_within_observed_distance_does_not_turn_mismatch_into_equivalence(self):
        result = compare_outputs(output([1, 2, 3]), output([4, 2, 3]), [output([4, 5, 3])])
        self.assertFalse(result["exact_match"])
        self.assertTrue(result["baseline_variation"]["within_observed_token_distance"])
        self.assertEqual(result["baseline_variation"]["max_observed_token_distance"], 2)

    def test_without_repeated_control_no_baseline_variation_claim(self):
        result = compare_outputs(output([1]), output([2]))
        self.assertIsNone(result["baseline_variation"]["within_observed_token_distance"])

    def test_incompatible_baselines_are_rejected(self):
        baseline = dict(output([2]), selected_prompts=[{"task": 1}])
        with self.assertRaises(ValueError):
            compare_outputs(output([1]), output([2]), [baseline])

    def full_capture(self):
        capture = V2RouteCapture(capture_draft=True)
        t = target()
        t["query_start_loc_gpu"] = t["query_start_loc"]
        first = draft_prefill()
        second = dict(target(), phase="draft_decode", query_start_loc=[0, 1],
                      num_real_rows=1, num_rows=1, num_rejected=[0],
                      is_padding=[False], ids=[[[4, 5]]], router_mask_present=[0],
                      router_is_padding=[[False]])
        capture.targets = [t]
        capture.records = [t, first, second]
        capture.manifest["geometry"] = {"target": [GEOMETRY], "draft": [dict(GEOMETRY, format="bf16")]}
        capture.manifest["control_token_ids_match"] = False
        return capture

    def export(self, capture, path):
        class Tensor:
            pass
        fake_torch = SimpleNamespace(Tensor=Tensor, save=lambda value, out: out.write_text("raw saved"))
        with patch.dict("sys.modules", {"torch": fake_torch}):
            capture.export(path, success=False, error="exact output mismatch", validate_structure=True)

    def test_full_draft_structure_validated_while_exact_failure_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.export(self.full_capture(), path)
            manifest = json.loads((path / "manifest.json").read_text())
            self.assertEqual(manifest["gpu_validation"], "failed")
            self.assertEqual(manifest["structural_validation"], "passed")
            self.assertEqual(manifest["error"], "exact output mismatch")
            self.assertTrue((path / "analysis_diagnostic.json").exists())
            self.assertFalse((path / "analysis.json").exists())

    def test_missing_draft_phase_keeps_raw_artifact_and_original_output_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            capture = self.full_capture()
            capture.records.pop()
            with self.assertRaisesRegex(RuntimeError, "draft data incomplete"):
                self.export(capture, path)
            manifest = json.loads((path / "manifest.json").read_text())
            self.assertEqual(manifest["gpu_validation"], "failed")
            self.assertEqual(manifest["structural_validation"], "failed")
            self.assertEqual(manifest["error"], "exact output mismatch")
            self.assertTrue((path / "raw_routes.pt").exists())


if __name__ == "__main__":
    unittest.main()
