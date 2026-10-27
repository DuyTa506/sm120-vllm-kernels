"""CPU-only contracts and opaque-op dispatch checks; never launch Triton/vLLM."""
import ast
import enum
import importlib.util
from pathlib import Path
from types import MethodType, SimpleNamespace
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


def resolve_installed_split_defaults(base, compilation_source, vllm_source):
    """Run the installed split completion/predicate without platform imports."""
    tree = ast.parse(compilation_source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CompilationConfig")
    enums = [n for n in tree.body if isinstance(n, ast.ClassDef)
             and n.name in ("CompilationMode", "CUDAGraphMode")]
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ("set_splitting_ops_for_v1", "splitting_ops_contain_kv_cache_update")]
    attention = ast.literal_eval(next(n.value for n in cls.body if isinstance(n, ast.AnnAssign)
                                      and isinstance(n.target, ast.Name) and n.target.id == "_attention_ops"))
    mla = next(n for n in ast.parse(vllm_source).body if isinstance(n, ast.FunctionDef)
               and n.name == "enable_rope_kvcache_mla_fusion")
    ns = dict(enum=enum, logger=SimpleNamespace(warning_once=lambda *a:None, info=lambda *a:None))
    exec(compile(ast.Module(body=[*enums, *methods, mla], type_ignores=[]), "installed_config_methods", "exec"), ns)
    passes = dict(fuse_attn_quant=False, fuse_rope_kvcache=False, fuse_qk_norm_rope_kvcache=False)
    passes.update(base.get("pass_config", {}))
    cfg = SimpleNamespace(mode=ns["CompilationMode"].VLLM_COMPILE,
                          cudagraph_mode=ns["CUDAGraphMode"].FULL_AND_PIECEWISE,
                          use_inductor_graph_partition=False,
                          splitting_ops=base.get("splitting_ops"),
                          pass_config=SimpleNamespace(**passes), _attention_ops=attention)
    cfg.splitting_ops_contain_kv_cache_update = MethodType(ns["splitting_ops_contain_kv_cache_update"], cfg)
    if "fuse_rope_kvcache_cat_mla" not in passes:
        cfg.pass_config.fuse_rope_kvcache_cat_mla = ns["enable_rope_kvcache_mla_fusion"](SimpleNamespace(compilation_config=cfg))
    ns["set_splitting_ops_for_v1"](cfg, "allgather_reducescatter", 1)
    return dict(splitting_ops=cfg.splitting_ops, pass_config=vars(cfg.pass_config))


class IntegrationGuardsTest(unittest.TestCase):
    def test_candidate_only_adds_gdn_to_actual_completed_baseline_splits(self):
        from integration_config import OP, add_split
        source_root = Path("/usr/local/lib/python3.12/dist-packages/vllm/config")
        compilation = (source_root / "compilation.py").read_text()
        vllm = (source_root / "vllm.py").read_text()
        baseline = resolve_installed_split_defaults({}, compilation, vllm)
        self.assertEqual(len(baseline["splitting_ops"]), 19)
        self.assertEqual(baseline["splitting_ops"][-2:],
                         ["vllm::unified_kv_cache_update", "vllm::unified_mla_kv_cache_update"])
        self.assertIs(baseline["pass_config"]["fuse_rope_kvcache_cat_mla"], False)
        generated, _ = add_split({}, compilation)
        actual = resolve_installed_split_defaults(generated, compilation, vllm)
        self.assertEqual([op for op in actual["splitting_ops"] if op != OP], baseline["splitting_ops"])
        self.assertEqual(actual["pass_config"], baseline["pass_config"])
        custom_base = {"backend": "inductor", "debug_dump_path": "/tmp/gdn-test",
                       "splitting_ops": [*baseline["splitting_ops"], "test::existing"],
                       "pass_config": {"fuse_norm_quant": False}}
        kept, _ = add_split(custom_base, compilation)
        self.assertEqual(kept["splitting_ops"], [*custom_base["splitting_ops"], OP])
        self.assertEqual(kept["debug_dump_path"], "/tmp/gdn-test")
        self.assertIs(kept["pass_config"]["fuse_norm_quant"], False)
        self.assertNotIn("fuse_rope_kvcache_cat_mla", custom_base["pass_config"])
        for invalid in ({"splitting_ops": []}, {"use_inductor_graph_partition": True},
                        {"pass_config": {"fuse_rope_kvcache_cat_mla": True}}):
            with self.assertRaises(ValueError):
                add_split(invalid, compilation)

    def test_warmup_covers_installed_mtp_full_capture_descriptors(self):
        from audit_capture_sizes import installed_capture_descriptors
        g = load_guards()
        # The integration failure occurs on the second descending FULL profile
        # descriptor: configured 184 is rounded to 186 for 3-token MTP queries.
        sizes = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88,
                 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184,
                 192, 200, 256, 1024, 16384]
        descriptors, _ = installed_capture_descriptors(sizes, 16384, 64, 3)
        self.assertEqual(descriptors["FULL"][:2], [192, 186])
        warmed = g.fallback_warmup_sizes(sizes, 16384, decode_query_len=3,
                                         max_num_seqs=64, max_capture_size=16384)
        self.assertTrue(set(descriptors["FULL"]).issubset(warmed),
                        f"Unwarmed actual FULL descriptors: {sorted(set(descriptors['FULL'])-set(warmed))}")
        self.assertTrue(set(descriptors["PIECEWISE"]).issubset(warmed))
        # Round-up respects both request and maximum-token bounds.
        self.assertEqual(g.fallback_warmup_sizes([1, 2, 4, 8], 16,
                         decode_query_len=3, max_num_seqs=2, max_capture_size=8),
                         [1, 2, 3, 4, 6, 8, 16])
        with self.assertRaises(ValueError):
            g.fallback_warmup_sizes([1, 2], 16, decode_query_len=3)

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
        layer = SimpleNamespace(prefix="test.linear_attn", _gdn_norm_fp8_enabled=True, _gdn_norm_fp8_sm_count=188,
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
        kernel._warmup_sizes = (3,)
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, r"tokens=64, actual_tokens=31.*warmed_tokens=\(3,\)"):
                run(layer, *args, SimpleNamespace(num_actual_tokens=31, num_prefills=0))
        kernel._warmup_sizes = (64, 1050)
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
