"""CPU-only isolation regressions; does not import vLLM or invoke CUDA."""
import copy
import json
import os
from dataclasses import dataclass
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from packed_state_probe_utils import clone_tree, private_fixture, private_owner, byte_mismatches


@dataclass
class Checkpoint:
    state_indices: object
    offsets: object


@dataclass
class Metadata:
    spec_state_indices_tensor: object
    non_spec_state_indices_tensor: object
    prefill_state_indices: object
    checkpoint: object


class Owner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.kv_cache = (torch.arange(30).reshape(10, 3), torch.arange(40).reshape(10, 4))
    def _forward_core(self):
        self.kv_cache[0][1].fill_(99)
    def _forward_core_fused_norm_packed(self):
        self._forward_core()


class IsolationTests(unittest.TestCase):
    def test_metadata_tensor_aliases_are_private_and_retained(self):
        tensor = torch.arange(6).reshape(2, 3)
        original = {"a": tensor, "nested": [tensor]}
        copied = clone_tree(original)
        self.assertIs(copied["a"], copied["nested"][0])
        copied["a"].zero_()
        self.assertEqual(int(tensor.sum()), 15)

    def test_null_zero_negative_and_checkpoint_rows_remap(self):
        owner = Owner()
        meta = Metadata(torch.tensor([[7, 9, 0]]), torch.tensor([4, -1]),
                        torch.tensor([4]), Checkpoint(torch.tensor([8, 0]), [16, 0]))
        meta._gdn_ckpt_step = (4, torch.tensor([0, 1]), torch.tensor([0]),
                              torch.ones(2, 3), copy.copy(meta.checkpoint))
        initial = tuple(x.clone() for x in owner.kv_cache)
        caches, copied, rows = private_fixture(owner.kv_cache, meta)
        self.assertEqual(rows, [0, 4, 7, 8, 9])
        self.assertEqual(copied.spec_state_indices_tensor.tolist(), [[2, 4, 0]])
        self.assertEqual(copied.non_spec_state_indices_tensor.tolist(), [1, -1])
        self.assertEqual(copied.checkpoint.state_indices.tolist(), [3, 0])
        self.assertEqual(copied._gdn_ckpt_step[4].state_indices.tolist(), [3, 0])
        copied._gdn_ckpt_step[3].zero_()
        self.assertEqual(float(meta._gdn_ckpt_step[3].sum()), 6)
        proxy = private_owner(owner, caches)
        self.assertIs(proxy._forward_core.__self__, proxy)
        self.assertIs(proxy.weight, owner.weight)
        proxy._forward_core_fused_norm_packed()
        self.assertEqual(int(proxy.kv_cache[0][1, 0]), 99)
        self.assertTrue(all(byte_mismatches(a, b) == 0 for a, b in zip(initial, owner.kv_cache)))

    def test_instance_hook_is_removed_only_on_proxy(self):
        owner = Owner()
        owner._forward_core_fused_norm_packed = lambda: self.fail("live hook entered")
        proxy = private_owner(owner, tuple(x.clone() for x in owner.kv_cache))
        proxy._forward_core_fused_norm_packed()
        self.assertIn("_forward_core_fused_norm_packed", vars(owner))
        self.assertNotIn("_forward_core_fused_norm_packed", vars(proxy))

    def test_known_optional_index_fields_can_be_none(self):
        owner = Owner()
        meta = Metadata(None, torch.tensor([4]), torch.tensor([4]), None)
        _, copied, rows = private_fixture(owner.kv_cache, meta)
        self.assertEqual(rows, [0, 4])
        self.assertIsNone(copied.spec_state_indices_tensor)
        self.assertEqual(copied.prefill_state_indices.tolist(), [1])

    def test_unknown_or_out_of_bounds_metadata_rejected(self):
        owner = Owner()
        for index in [-2, 10]:
            meta = Metadata(None, torch.tensor([index]), None, None)
            with self.assertRaises(ValueError): private_fixture(owner.kv_cache, meta)
        with self.assertRaises(TypeError): clone_tree(object())

    def test_actual_route_labels_and_bounded_case_budget(self):
        from packed_state_probe_worker import case_kind, MAX_CASES
        self.assertEqual(MAX_CASES, 5)
        def metadata(prefills=0, decodes=0, spec=0, actual=2048):
            return SimpleNamespace(num_prefills=prefills, num_decodes=decodes,
                                   num_spec_decodes=spec, num_actual_tokens=actual)
        self.assertEqual(case_kind(metadata(prefills=1)), "long_prefill")
        self.assertEqual(case_kind(metadata(prefills=1, decodes=1)), "mixed_non_spec")
        self.assertEqual(case_kind(metadata(prefills=1, spec=1)), "mixed_spec")
        self.assertEqual(case_kind(metadata(spec=2, actual=6)), "pure_decode")
        self.assertIsNone(case_kind(metadata(prefills=1, actual=31)))

    def test_preflight_container_has_no_gpu_and_readonly_source_mounts(self):
        from run_packed_state_probe import build_command, ROOT, MOUNTS
        args = SimpleNamespace(run_gpu=False, gpu=0, hf_cache=Path("/tmp/hf"),
                               output_dir=ROOT / "kernels/gdn-norm-fp8/results/test_private")
        _, command = build_command(args, "testowner")
        self.assertNotIn("--gpus", command)
        self.assertIn("CUDA_VISIBLE_DEVICES=", command)
        self.assertIn("NVIDIA_VISIBLE_DEVICES=void", command)
        self.assertIn("VLLM_GDN_NORM_FP8_FUSION=0", command)
        self.assertTrue(all(any(path in value and value.endswith(":ro") for value in command) for path in MOUNTS))
        args.run_gpu = True
        _, command = build_command(args, "testowner")
        self.assertIn("--gpus", command)
        self.assertIn("--run-gpu", command)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=", command)

    def test_float_nan_comparison_is_bitwise(self):
        x = torch.tensor([float("nan"), -0.0])
        self.assertEqual(byte_mismatches(x, x.clone()), 0)
        y = x.clone(); y[1] = 0.0
        self.assertGreater(byte_mismatches(x, y), 0)


if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU test requires CUDA hidden"
    assert not torch.cuda.is_initialized()
    program = unittest.main(exit=False)
    assert not torch.cuda.is_initialized()
    print(json.dumps({"status": "CPU_PASS" if program.result.wasSuccessful() else "CPU_FAIL",
                      "tests_run": program.result.testsRun, "cuda_initialized": torch.cuda.is_initialized()}))
    raise SystemExit(0 if program.result.wasSuccessful() else 1)
