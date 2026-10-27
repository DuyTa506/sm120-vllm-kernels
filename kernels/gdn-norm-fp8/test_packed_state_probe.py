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

    def test_declared_baseline_mounts_reject_forced_b12x_for_mtp(self):
        from probe_packed_state import engine_options, validate_fixture_backend, MOUNTS
        options = engine_options(Path("/tmp/snapshot"), 0.78)
        validate_fixture_backend(options, MOUNTS)
        self.assertEqual(options.get("moe_backend", "auto"), "auto")
        options["moe_backend"] = "flashinfer_b12x"
        with self.assertRaisesRegex(ValueError, "unquantized MTP"):
            validate_fixture_backend(options, MOUNTS)

    def test_small_mixed_does_not_consume_required_fused_case(self):
        from packed_state_probe_worker import selected_case_key, coverage_requirements
        self.assertIsNone(selected_case_key("mixed_spec", 31, set()))
        self.assertEqual(selected_case_key("mixed_spec", 1024, set()), "mixed_spec")
        cases = [{"kind": "long_prefill"}, {"kind": "pure_decode"}]
        small = {"kind": "mixed_spec", "metadata": {"num_actual_tokens": 31},
                 "variants": [{"added_padding": p, "route": "original"} for p in (0, 17)]}
        self.assertFalse(coverage_requirements(cases + [small])["complete"])
        large = copy.deepcopy(small); large["metadata"]["num_actual_tokens"] = 1024
        self.assertFalse(coverage_requirements(cases + [large])["complete"])
        for variant in large["variants"]: variant["route"] = "fused"
        self.assertTrue(coverage_requirements(cases + [large])["complete"])

    def test_separate_cold_long_phase_completes_aggregate_coverage(self):
        from probe_packed_state import request_phases
        from packed_state_probe_worker import coverage_requirements
        ids = list(range(8192))
        phases = request_phases(ids, "owned-run")
        self.assertEqual(len(phases), 2)
        self.assertEqual([len(p["prompt_token_ids"]) for p in phases[0]["prompts"]], [64, 8192])
        self.assertEqual(phases[0]["output_tokens"], 32)
        self.assertEqual(len(phases[1]["prompts"]), 1)
        self.assertEqual(phases[1]["prompts"][0]["prompt_token_ids"], ids)
        salt = phases[1]["prompts"][0]["cache_salt"]
        self.assertTrue(salt)
        self.assertNotIn(salt, [p.get("cache_salt") for p in phases[0]["prompts"]])
        self.assertNotEqual(salt, request_phases(ids, "another-run")[1]["prompts"][0]["cache_salt"])
        self.assertEqual(phases[1]["output_tokens"], 1)
        cases = [{"kind": "mixed_spec", "metadata": {"num_actual_tokens": 2048},
                  "variants": [{"added_padding": p, "route": "fused"} for p in (0, 17)]},
                 {"kind": "pure_decode"}]
        self.assertFalse(coverage_requirements(cases)["complete"])
        self.assertTrue(coverage_requirements(cases + [{"kind": "long_prefill"}])["complete"])

    def test_float_nan_comparison_is_bitwise(self):
        x = torch.tensor([float("nan"), -0.0])
        self.assertEqual(byte_mismatches(x, x.clone()), 0)
        y = x.clone(); y[1] = 0.0
        self.assertGreater(byte_mismatches(x, y), 0)


class CleanupTests(unittest.TestCase):
    def run_cleanup(self, replies):
        from run_packed_state_probe import cleanup_owned_container
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            reply = replies.pop(0)
            if isinstance(reply, BaseException): raise reply
            return SimpleNamespace(returncode=reply[0], stdout=reply[1], stderr=reply[2])
        result = cleanup_owned_container("gdn-private-state-owned", "owned", run=run)
        self.assertEqual(replies, [])
        return result, calls

    @staticmethod
    def state(running=True, owner="owned", pid=None):
        return (0, json.dumps([{"Name": "/gdn-private-state-owned", "Id": "exact-owned-id",
                              "Config": {"Labels": {"gdn.state_probe_owner": owner}},
                              "State": {"Running": running, "Restarting": False, "Status": "running" if running else "exited",
                                        "Pid": (123 if running else 0) if pid is None else pid}}]), "")

    def test_absent_needs_no_stop(self):
        result, calls = self.run_cleanup([(1, "[]", "Error: No such object: gdn-private-state-owned")])
        self.assertTrue(result["verified_stopped"])
        self.assertEqual(len(calls), 1)

    def test_stop_success_requires_state_check(self):
        result, calls = self.run_cleanup([self.state(), (0, "owned", ""), self.state(False)])
        self.assertTrue(result["verified_stopped"])
        self.assertEqual([c[1] for c in calls], ["inspect", "stop", "inspect"])
        self.assertEqual(calls[1], ["docker", "stop", "--time", "30", "exact-owned-id"])

    def test_failed_stop_force_removes_only_rechecked_owned_id(self):
        result, calls = self.run_cleanup([self.state(), (1, "", "stop failed"), self.state(),
                                        (0, "exact-owned-id", ""), (1, "[]", "No such object")])
        self.assertTrue(result["verified_stopped"])
        self.assertEqual(calls[3], ["docker", "rm", "-f", "exact-owned-id"])
        self.assertEqual(result["actions"][0]["returncode"], 1)

    def test_ownership_change_refuses_force_remove(self):
        result, calls = self.run_cleanup([self.state(), (1, "", "stop failed"), self.state(owner="foreign")])
        self.assertFalse(result["verified_stopped"])
        self.assertNotIn("rm", [c[1] for c in calls])

    def test_failed_force_remove_preserves_running_failure(self):
        result, calls = self.run_cleanup([self.state(), (1, "", "stop failed"), self.state(),
                                        (1, "", "rm failed"), self.state()])
        self.assertFalse(result["verified_stopped"])
        self.assertEqual(result["status"], "cleanup_failed")

    def test_inspection_failure_does_not_mean_absent(self):
        result, calls = self.run_cleanup([(1, "", "Cannot connect to Docker daemon")])
        self.assertFalse(result["verified_stopped"])
        self.assertEqual(len(calls), 1)

    def test_stop_timeout_is_recorded_and_owned_removal_verified(self):
        import subprocess
        result, calls = self.run_cleanup([self.state(), subprocess.TimeoutExpired("docker stop", 45),
                                        self.state(), (0, "exact-owned-id", ""),
                                        (1, "[]", "No such object")])
        self.assertTrue(result["verified_stopped"])
        self.assertIsNone(result["actions"][0]["returncode"])
        self.assertIn("error", result["actions"][0])

    def test_gpu_lock_remains_held_until_cleanup_recovery(self):
        import fcntl
        import tempfile
        from unittest import mock
        import run_packed_state_probe as runner
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); lock = folder / "gpu.lock"; output = folder / "probe"
            record = folder / "probe.runner.json"
            child = mock.Mock(); child.wait.return_value = 0; child.poll.return_value = 0
            failure = {"container": "owned", "verified_stopped": False,
                       "status": "cleanup_failed", "observations": [], "actions": []}
            def while_waiting(_):
                self.assertFalse(json.loads(record.read_text())["gpu_slot_release_confirmed"])
                with lock.open("a") as other:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mock.patch("sys.argv", ["runner", "--run-gpu", "--lock", str(lock),
                                         "--hf-cache", directory, "--output-dir", str(output)]), \
                 mock.patch.object(runner, "build_command", return_value=("owned", ["docker", "run"])), \
                 mock.patch.object(runner.subprocess, "check_output", return_value=""), \
                 mock.patch.object(runner.subprocess, "Popen", return_value=child), \
                 mock.patch.object(runner, "cleanup_owned_container", return_value=failure), \
                 mock.patch.object(runner, "inspect_owned_container", return_value={"verified_stopped": True, "absent": True}), \
                 mock.patch.object(runner.time, "sleep", side_effect=while_waiting), \
                 mock.patch.object(Path, "mkdir"):
                with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                    runner.main()
            result = json.loads(record.read_text())
            self.assertEqual(result["exit_code"], 1)
            self.assertEqual(result["probe_exit_code"], 0)
            self.assertTrue(result["cleanup_recovery_required"])
            self.assertTrue(result["gpu_slot_release_confirmed"])
            with lock.open("a") as other:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_created_container_is_not_a_verified_terminal_state(self):
        created = list(self.state(False))
        value = json.loads(created[1]); value[0]["State"]["Status"] = "created"
        created[1] = json.dumps(value)
        result, calls = self.run_cleanup([created, (1, "", "not running"), created,
                                        (0, "exact-owned-id", ""), (1, "[]", "No such object")])
        self.assertTrue(result["verified_stopped"])
        self.assertEqual(calls[3], ["docker", "rm", "-f", "exact-owned-id"])

    def test_not_running_with_live_pid_is_unverified(self):
        result, calls = self.run_cleanup([self.state(False, pid=123), (0, "owned", ""),
                                        self.state(False, pid=123), (1, "", "rm failed"), self.state(False, pid=123)])
        self.assertFalse(result["verified_stopped"])


if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU test requires CUDA hidden"
    assert not torch.cuda.is_initialized()
    program = unittest.main(exit=False)
    assert not torch.cuda.is_initialized()
    print(json.dumps({"status": "CPU_PASS" if program.result.wasSuccessful() else "CPU_FAIL",
                      "tests_run": program.result.testsRun, "cuda_initialized": torch.cuda.is_initialized()}))
    raise SystemExit(0 if program.result.wasSuccessful() else 1)
