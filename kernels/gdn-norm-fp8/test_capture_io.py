"""CPU tests for capture artifact integrity and layout reconstruction."""
import os
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

from capture_io import load_capture, restore_layout, tensor_spec, write_capture
from capture_worker import GDNCaptureWorkerExtension


class CaptureIOTest(unittest.TestCase):
    def test_strided_gate_round_trip_preserves_values_and_offset(self):
        # A contiguous saved copy must not erase the production QKVZ slice layout.
        source = torch.arange(3 * 12288, dtype=torch.float32).to(torch.bfloat16)
        z = source.view(3, 12288)[:, 8192:].view(3, 32, 128)
        spec = tensor_spec(z)
        restored = restore_layout(z.contiguous(), spec, device="cpu")
        self.assertEqual(restored.stride(), (12288, 128, 1))
        self.assertEqual(restored.storage_offset(), 8192)
        self.assertTrue(torch.equal(restored, z))
        self.assertFalse(torch.cuda.is_initialized())

    def test_capture_file_integrity_and_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            x = torch.zeros(3, 32, 128, dtype=torch.bfloat16)
            z = torch.ones(3, 12288, dtype=torch.bfloat16)[:, 8192:].view_as(x)
            w = torch.ones(128, dtype=torch.bfloat16)
            entry = write_capture(Path(directory), 18, "model.layers.18.linear_attn.norm",
                                  {"x": x, "z": z, "weight": w, "norm_output": x})
            data = load_capture(Path(directory), entry, expected_tokens=3)
            self.assertTrue(torch.equal(data["z"], z))
            self.assertEqual(entry["specs"]["z"]["stride"], [12288, 128, 1])
            with self.assertRaisesRegex(ValueError, "shape"):
                load_capture(Path(directory), entry, expected_tokens=4)
            artifact = Path(directory) / entry["file"]
            artifact.write_bytes(artifact.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_capture(Path(directory), entry, expected_tokens=3)

    def test_rejects_overlapping_or_unexpected_layout(self):
        z = torch.zeros(3, 32, 128, dtype=torch.bfloat16)
        spec = tensor_spec(z)
        spec["stride"] = [1, 128, 1]
        with self.assertRaisesRegex(ValueError, "stride"):
            restore_layout(z, spec, device="cpu")
        spec["stride"] = [4096, 1, 32]
        with self.assertRaisesRegex(ValueError, "stride"):
            restore_layout(z, spec, device="cpu")


class CaptureHookTest(unittest.TestCase):
    def test_post_initialization_capture_removes_only_owned_hooks(self):
        # This CPU module tests hook plumbing, never the CUDA norm's arithmetic.
        class RMSNormGated(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(128, dtype=torch.bfloat16))
                self.eps, self.group_size, self.norm_before_gate = 1e-6, None, True
                self.activation, self.bias = "silu", None
                self._forward_method = self.forward_cuda

            def forward_cuda(self, x, z):
                return x + z * self.weight

            def forward(self, x, z):
                return self._forward_method(x, z)

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, GDN_ACTIVATION_CAPTURE="1"):
            layer = torch.nn.Module()
            layer.linear_attn = torch.nn.Module()
            layer.linear_attn.norm = RMSNormGated()
            model = torch.nn.Module()
            model.layers = torch.nn.ModuleDict({"18": layer})
            worker = GDNCaptureWorkerExtension()
            worker.model_runner = SimpleNamespace(get_model=lambda: model)
            worker.vllm_config = SimpleNamespace(
                model_config=SimpleNamespace(enforce_eager=True),
                compilation_config=SimpleNamespace(mode=0, custom_ops=["all"]),
                speculative_config=None,
                parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
                scheduler_config=SimpleNamespace(enable_chunked_prefill=False, max_num_seqs=1,
                                                 max_num_batched_tokens=3),
                cache_config=SimpleNamespace(enable_prefix_caching=False))
            norm = layer.linear_attn.norm
            unrelated_calls = []
            unrelated = norm.register_forward_hook(lambda *args: unrelated_calls.append(1))
            x = torch.ones(3, 32, 128, dtype=torch.bfloat16)
            z = torch.ones(3, 12288, dtype=torch.bfloat16)[:, 8192:].view_as(x)
            norm(x, z)  # Startup work before the installation RPC is not captured.
            request = {"capture_id": "cpu-hook-test", "layers": [18], "tokens": 3}
            (Path(directory) / "capture_request.json").write_text(json.dumps(request))
            worker.gdn_capture_install(directory, "cpu-hook-test", [18], 3)
            expected = norm(x, z)
            report = worker.gdn_capture_finish()
            import msgspec
            # Exercise the actual RPC serializer, which rejects TorchVersion
            # even though JSON accepts that str subclass.
            msgspec.msgpack.encode(report)
            self.assertIs(type(report["torch_version"]), str)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["calls"], {"18": 1})
            data = load_capture(Path(directory), report["captures"][0], expected_tokens=3)
            self.assertTrue(torch.equal(data["norm_output"], expected))
            self.assertEqual(report["captures"][0]["specs"]["z"]["storage_offset"], 8192)
            norm(x, z)  # Must not trigger the duplicate-call guard after cleanup.
            self.assertEqual(len(unrelated_calls), 3)
            self.assertEqual(len(norm._forward_hooks), 1)
            self.assertEqual(len(norm._forward_pre_hooks), 0)
            unrelated.remove()
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    assert not torch.cuda.is_initialized()
    unittest.main()
