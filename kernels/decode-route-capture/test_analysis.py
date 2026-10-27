"""CPU-only tests: no torch, CUDA, HTTP, model loading, or token generation."""
import unittest

from analyze import analyze_capture, analyze_record, expert_bytes, classify_rows


GEOMETRY = {"hidden": 16, "intermediate": 16, "experts": 8, "topk": 2,
            "format": "nvfp4", "layer_id": 19}


def target(step=0):
    return {"step": step, "phase": "target_verify", "ids_dtype": "int32",
            "layer_ids": [19], "request_ids": ["request-A"],
            "query_start_loc": [0, 3], "num_rows": 4, "num_real_rows": 3,
            "num_rejected": [1], "num_sampled": [2], "num_draft_tokens_per_req": [2],
            "is_padding": [False, False, False, True],
            "ids": [[[1, 2]], [[2, 3]], [[3, 4]], [[-1, -1]]]}


def draft_prefill():
    record = target()
    record.update(phase="draft_prefill", router_mask_present=[0],
                  router_is_padding=[[False]] * 4)
    record["ids"][3] = [[4, 7]]
    return record


class AnalysisTests(unittest.TestCase):
    def test_actual_nvfp4_geometry_includes_block_scales(self):
        self.assertEqual(expert_bytes(dict(GEOMETRY, hidden=2048, intermediate=512)), 1769472)
        self.assertEqual(expert_bytes(dict(GEOMETRY, hidden=2048, intermediate=512,
                                         format="bf16")), 6291456)

    def test_rejected_verifier_rows_are_counted_and_padding_is_not(self):
        result = analyze_record(target(), [GEOMETRY])
        layer = result["layers"][0]
        self.assertEqual(layer["unique_experts_executed"], 4)
        self.assertEqual(layer["unique_experts_required"], 3)
        self.assertEqual(layer["weight_payload_bytes"], 1728)
        self.assertEqual(layer["extra_experts_graph_tail"], 0)
        self.assertIsNone(layer["potentially_avoidable_draft_experts"])
        group = layer["requests"][0]
        self.assertEqual(group["union_experts"], 4)
        self.assertEqual(group["adjacent_intersections"], [1, 1])
        self.assertEqual(group["lag2_intersections"], [0])

    def test_rejected_suffix_is_per_request_not_a_global_tail(self):
        record = dict(target(), request_ids=["A", "B"], query_start_loc=[0, 3, 6],
                      num_real_rows=6, num_rows=8, num_rejected=[1, 2],
                      is_padding=[False] * 6 + [True] * 2)
        self.assertEqual(classify_rows(record), [
            "required", "required", "rejected", "required", "rejected", "rejected",
            "graph_padding", "graph_padding"])

    def test_draft_extra_experts_are_unique_against_required_rows(self):
        record = draft_prefill()
        layer = analyze_record(record, [dict(GEOMETRY, format="bf16")])["layers"][0]
        self.assertEqual(layer["unique_experts_executed"], 5)
        self.assertEqual(layer["extra_experts_graph_tail"], 1)
        self.assertEqual(layer["potentially_avoidable_draft_experts"], 2)

    def test_uint8_capture_cannot_disguise_negative_ids_as_expert_255(self):
        record = dict(target(), ids_dtype="uint8")
        with self.assertRaises(ValueError):
            analyze_record(record, [GEOMETRY])

    def test_unmasked_target_padding_is_reported_as_invalid_not_discarded(self):
        record = target()
        record["ids"][3] = [[6, 7]]
        with self.assertRaises(ValueError):
            analyze_record(record, [GEOMETRY])

    def test_capture_pairs_steps_and_reports_acceptance_without_assuming_three_tokens(self):
        next_record = target(1)
        next_record["ids"] = [[[2, 3]], [[3, 4]], [[4, 5]], [[-1, -1]]]
        manifest = {"geometry": {"target": [GEOMETRY], "draft": []},
                    "draft_capture": "disabled_target_only"}
        result = analyze_capture(manifest, [target(), next_record])
        group = result["records"][1]["layers"][0]["requests"][0]
        self.assertEqual(group["previous_capture_intersection"], 3)
        self.assertEqual(group["previous_capture_step_gap"], 1)
        self.assertEqual(result["acceptance"]["proposed_draft_tokens"], 4)
        self.assertEqual(result["acceptance"]["accepted_draft_tokens"], 2)
        self.assertEqual(result["acceptance"]["sampled_output_tokens"], 4)
        self.assertEqual(result["target_weight_payload_bytes"], 3456)
        self.assertEqual(result["target_bytes_per_sampled_output_token"], 864)
        self.assertEqual(result["draft_capture"], "disabled_target_only")

    def test_draft_phase_overlap_is_separate_from_target_weights(self):
        prefill = draft_prefill()
        decode = dict(target(), phase="draft_decode", query_start_loc=[0, 1],
                      num_real_rows=1, num_rows=2, num_rejected=[0],
                      is_padding=[False, True], ids=[[[4, 5]], [[-1, -1]]],
                      router_mask_present=[1], router_is_padding=[[False], [True]])
        manifest = {"geometry": {"target": [GEOMETRY],
                                 "draft": [dict(GEOMETRY, format="bf16")]},
                    "draft_capture": "captured_both_phases"}
        result = analyze_capture(manifest, [target(), prefill, decode])
        pair = result["draft_phase_pairs"][0]
        self.assertEqual(pair["intersection_experts"], 1)
        self.assertEqual(pair["union_experts"], 6)
        self.assertEqual(pair["unique_payload_bytes_both_phases"], 9216)
        self.assertEqual(result["target_weight_payload_bytes"], 1728)
        self.assertEqual(result["steps_missing_draft_pair"], [])

    def test_duplicate_phase_is_not_silently_overwritten(self):
        manifest = {"geometry": {"target": [GEOMETRY], "draft": []},
                    "draft_capture": "disabled_target_only"}
        with self.assertRaises(ValueError):
            analyze_capture(manifest, [target(), target()])

    def test_draft_mask_is_required_and_matches_signed_routes(self):
        for change in ({"router_is_padding": None},
                       {"router_is_padding": [[False], [False], [False], [True]]},
                       {"router_mask_present": [1],
                        "router_is_padding": [[True], [False], [False], [False]]},
                       {"ids": [[[-1, -1]], [[2, 3]], [[3, 4]], [[4, 7]]]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze_record(dict(draft_prefill(), **change), [dict(GEOMETRY, format="bf16")])

    def test_duplicate_topk_ids_cannot_disguise_unwritten_native_buffer(self):
        record = target()
        record["ids"][0] = [[0, 0]]
        with self.assertRaises(ValueError):
            analyze_record(record, [GEOMETRY])

    def test_impossible_acceptance_or_mtp_width_is_rejected(self):
        manifest = {"geometry": {"target": [GEOMETRY], "draft": []},
                    "draft_capture": "disabled_target_only"}
        for change in ({"num_sampled": [3]}, {"num_sampled": [True]},
                       {"num_draft_tokens_per_req": [1]}, {"num_rejected": [0.5]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze_capture(manifest, [dict(target(), **change)])

    def test_full_capture_cannot_claim_complete_with_missing_draft_pair(self):
        manifest = {"geometry": {"target": [GEOMETRY], "draft": [dict(GEOMETRY, format="bf16")]},
                    "draft_capture": "captured_both_phases"}
        with self.assertRaises(ValueError):
            analyze_capture(manifest, [target()])


if __name__ == "__main__":
    unittest.main()
