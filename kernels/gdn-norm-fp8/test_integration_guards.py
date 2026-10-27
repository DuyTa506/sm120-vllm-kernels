"""CPU-only contracts and opaque-op dispatch checks; never launch Triton/vLLM."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent / "overlay/vllm/model_executor/layers/mamba/gdn"


def load_guards():
    spec = importlib.util.spec_from_file_location("gdn_guard_test", ROOT / "gdn_norm_fp8_guards.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_function(name, namespace, source="gdn_norm_fp8_fusion.py"):
    tree = ast.parse((ROOT / source).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), source, "exec"), namespace)
    return namespace[name]


class IntegrationGuardsTest(unittest.TestCase):
    def test_configuration_and_dispatch(self):
        g = load_guards()
        valid = dict.fromkeys(g.CONFIGURATION_KEYS, True)
        self.assertIsNone(g.configuration_rejection(valid))
        for key in valid:
            with self.subTest(key=key):
                self.assertEqual(g.configuration_rejection({**valid, key: False}), key)
        self.assertEqual([g.reference_rows_per_block(n, 188) for n in (3, 16, 31)], [1, 2, 4])
        self.assertEqual(g.packed_route(16384, None, 0), "profile")
        self.assertEqual(g.packed_route(1024, 192, 0), "original")
        self.assertEqual(g.packed_route(16384, 2048, 0), "original")
        self.assertEqual(g.packed_route(1024, 1023, 1), "original")
        self.assertEqual(g.packed_route(1025, 1024, 1), "fused")
        with self.assertRaises(ValueError):
            g.packed_route(31, 32, 1)
        self.assertEqual(g.fallback_warmup_sizes([3, 6, 192], 16384), [1, 2, 3, 6, 192, 16384])

    def test_packed_routes_padding_and_original_state_delegation(self):
        import torch
        g = load_guards()
        events = []
        def fake_fusion(x, z, w, *, out, num_warps, reference_rows_per_block):
            events.append(("fused", x.shape[0], z.stride(), z.storage_offset()))
            self.assertTrue(torch.isfinite(z).all())
            self.assertEqual(num_warps, 8)
            self.assertEqual(reference_rows_per_block, 4)
            out[0].copy_(torch.ones_like(x.flatten(-2)).to(torch.float8_e4m3fn))
            out[1].fill_(0.125)
        namespace = {"torch": torch, "SCALE_FLOOR": g.SCALE_FLOOR,
                     "packed_route": g.packed_route, "reference_rows_per_block": g.reference_rows_per_block,
                     "fused_norm_gate_fp8_explicit": fake_fusion}
        run = extract_function("packed_norm_fp8", namespace)
        def quant(x, *_):
            events.append(("quant", tuple(x.shape)))
            scales = (x.abs().amax(-1, keepdim=True).float() / 448).clamp(min=g.SCALE_FLOOR)
            return (x.float() * scales.reciprocal()).clamp(-448, 448).to(torch.float8_e4m3fn), scales
        make_out = extract_function("make_raw_quant_out", {}, "gdn_norm_fp8_quant_out.py")
        quant_out = make_out(SimpleNamespace(forward_native=quant))
        kernel = SimpleNamespace(_ready=True, _warmup_sizes=(64, 1050),
                                 quant_fp8=quant, _quant_out=quant_out)
        layer = SimpleNamespace(_gdn_norm_fp8_enabled=True, _gdn_norm_fp8_sm_count=188,
                                norm=SimpleNamespace(weight=torch.ones(128, dtype=torch.bfloat16)),
                                out_proj=SimpleNamespace(scheme=SimpleNamespace(fp8_linear=kernel)))
        def prepare(total, actual):
            packed = torch.zeros((total, 12288), dtype=torch.bfloat16)
            packed[actual:, 8192:] = float("nan")
            ba = torch.zeros((total, 64), dtype=torch.bfloat16)
            core = torch.zeros((total, 32, 128), dtype=torch.bfloat16)
            q = torch.empty((total, 4096), dtype=torch.float8_e4m3fn)
            s = torch.empty((total, 1))
            return packed, ba, core, q, s
        layer._warmup_prefill_kernels = lambda x, n: events.append(("warmup", tuple(x.shape), n))
        args = prepare(64, 0)
        run(layer, *args, None)
        self.assertEqual(events, [("warmup", (64, 8192), 0)])
        self.assertTrue((args[3].view(torch.uint8) == 0).all())
        self.assertTrue((args[4] == g.SCALE_FLOOR).all())
        events.clear()
        def original(packed, ba, core):
            events.append(("original",))
            core[:31].fill_(1)
        layer._forward_core_fused_norm_packed = original
        args = prepare(64, 31)
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
            run(layer, *args, SimpleNamespace(num_actual_tokens=31, num_prefills=0))
        self.assertEqual(events, [("original",), ("quant", (64, 4096))])
        self.assertTrue((args[3][31:].view(torch.uint8) == 0).all())
        self.assertTrue((args[4][31:] == g.SCALE_FLOOR).all())
        events.clear()
        layer.split_ba = lambda x: x.split([32, 32], dim=-1)
        def core(**kw):
            events.append(("core",))
            self.assertTrue(kw["b"].is_contiguous() and kw["a"].is_contiguous())
            kw["core_attn_out"][:1025].fill_(1)
        layer._forward_core = core
        args = prepare(1050, 1025)
        run(layer, *args, SimpleNamespace(num_actual_tokens=1025, num_prefills=1))
        self.assertEqual(events, [("core",), ("fused", 1025, (12288, 128, 1), 8192)])
        self.assertTrue((args[3][1025:].view(torch.uint8) == 0).all())
        self.assertTrue((args[4][1025:] == g.SCALE_FLOOR).all())
        self.assertFalse(torch.cuda.is_initialized())

    def test_overlay_uses_the_gpu_validated_quant_out_source_and_options(self):
        tested = ROOT.parents[5] / "fallback_quant_out.py"
        self.assertEqual((ROOT / "gdn_norm_fp8_quant_out.py").read_bytes(), tested.read_bytes())
        tree = ast.parse((ROOT / "gdn_norm_fp8_fusion.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                 and n.func.value.id == "torch" and n.func.attr == "compile"]
        self.assertEqual(len(calls), 1)
        self.assertEqual({k.arg: ast.literal_eval(k.value) for k in calls[0].keywords},
                         {"dynamic": True, "fullgraph": True})
        self.assertEqual(ast.unparse(calls[0].args[0]), "make_raw_quant_out(self.quant_fp8)")

    def test_split_op_fake_schema_and_stable_buffers(self):
        import torch
        source = "qwen_gdn_linear_attn.py"
        namespace = {"torch": torch, "LayerNameType": str}
        fake = extract_function("_fake_gdn_attention_core_norm_fp8_packed_experimental", namespace, source)
        x = torch.empty((3, 12288), dtype=torch.bfloat16)
        b = torch.empty((3, 64), dtype=torch.bfloat16)
        core = torch.empty((3, 32, 128), dtype=torch.bfloat16)
        q = torch.empty((3, 4096), dtype=torch.float8_e4m3fn)
        s = torch.empty((3, 1))
        self.assertIsNone(fake(x, b, core, q, s, "model.layers.0.linear_attn"))
        tree = ast.parse((ROOT / source).read_text())
        registration = next(n.value for n in tree.body if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call) and any(k.arg == "op_name" and isinstance(k.value, ast.Constant)
            and k.value.value == load_guards().OP_NAME for k in n.value.keywords))
        self.assertEqual(ast.literal_eval(next(k.value for k in registration.keywords if k.arg == "mutates_args")),
                         ["core_attn_out", "q", "scale"])
        self.assertFalse(torch.cuda.is_initialized())

    def test_mutating_schema_is_one_dynamic_cpu_graph(self):
        import torch
        # Same mutation/shape contract, with a CPU implementation in a private
        # namespace. This checks Dynamo/fake schema, not CUDA or vLLM execution.
        @torch.library.custom_op("gdn_cpu_schema_test::packed", mutates_args=("core", "q", "scale"))
        def op(x: torch.Tensor, core: torch.Tensor, q: torch.Tensor, scale: torch.Tensor) -> None:
            core.copy_(x)
            q.copy_(x.flatten(-2).to(torch.float8_e4m3fn))
            scale.fill_(1)

        @op.register_fake
        def fake(x, core, q, scale):
            return None

        graphs = []
        def backend(gm, inputs):
            graphs.append(gm)
            return gm.forward

        def forward(x):
            core = torch.zeros_like(x)
            q = torch.empty((x.shape[0], 4096), dtype=torch.float8_e4m3fn)
            scale = torch.empty((x.shape[0], 1), dtype=torch.float32)
            op(x, core, q, scale)
            return q, scale

        compiled = torch.compile(forward, backend=backend, dynamic=True, fullgraph=True)
        for tokens in (3, 31, 1025):
            x = torch.ones((tokens, 32, 128), dtype=torch.bfloat16)
            q, scale = compiled(x)
            self.assertTrue((q.float() == 1).all())
            self.assertTrue((scale == 1).all())
        self.assertEqual(len(graphs), 1)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
