"""CPU-only compile of installed FLA layouts and isolated explicit candidate."""
import ast
from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType
from unittest.mock import patch

from capture_io import file_sha256


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TRITON_DUMP_PTXAS_LOG"] = "1"
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    import torch

    assert not torch.cuda.is_initialized()
    root = Path(__file__).resolve().parent
    output = root / "results/explicit_layout_offline"
    output.mkdir(parents=True, exist_ok=True)
    attempts = []

    def forbidden(name):
        def fail(*args, **kwargs):
            attempts.append(name)
            raise RuntimeError(f"Forbidden runtime access: {name}")
        return fail

    class ForbiddenDriver:
        def __getattr__(self, name):
            return forbidden("triton.driver." + name)()

    report = {"status": "pending", "scope": "Offline compilation only; GPU equivalence untested",
              "cuda_visible_devices": "", "cuda_initialized_before": False,
              "runtime_access_attempts": attempts, "kernel_launches": 0, "variants": []}
    try:
        with tempfile.TemporaryDirectory(prefix="gdn-explicit-offline-") as cache, ExitStack() as stack:
            os.environ["TRITON_CACHE_DIR"] = cache
            for name in ("_lazy_init", "init", "is_available", "device_count", "current_device",
                         "get_device_properties", "get_device_capability", "get_device_name",
                         "current_stream", "default_stream", "set_device", "synchronize"):
                stack.enter_context(patch.object(torch.cuda, name, forbidden("torch.cuda." + name)))
            import triton
            import triton.language as tl
            from triton.backends.compiler import GPUTarget
            from triton.compiler import ASTSource
            from triton.experimental.gluon._runtime import GluonASTSource
            from triton.runtime.driver import driver
            driver.set_active(ForbiddenDriver())
            from fused_explicit import _norm_gate_fp8_explicit

            package = Path(importlib.util.find_spec("vllm").origin).parent
            baseline_path = package / "third_party/flash_linear_attention/ops/layernorm_guard.py"
            function = next(node for node in ast.parse(baseline_path.read_text()).body
                            if isinstance(node, ast.FunctionDef) and node.name == "layer_norm_fwd_kernel")
            module = ModuleType("gdn_fla_offline_source")
            module.__file__ = str(baseline_path)
            module.__dict__.update(triton=triton, tl=tl)
            sys.modules[module.__name__] = module
            exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                         str(baseline_path), "exec"), module.__dict__)
            baseline_fn = module.layer_norm_fwd_kernel.fn  # unwrap only the heuristics wrapper
            report.update(torch_version=torch.__version__, triton_version=triton.__version__,
                          baseline_source_sha256=file_sha256(baseline_path),
                          candidate_source_sha256=file_sha256(root / "fused_explicit.py"))

            def compile_one(name, source, warps, **labels):
                log = io.StringIO()
                with redirect_stdout(log):
                    compiled = triton.compile(source, target=GPUTarget("cuda", 120, 32),
                                              options={"num_warps": warps, "enable_fp_fusion": True})
                assert compiled.module is None and compiled.function is None and compiled._run is None
                assert not torch.cuda.is_initialized()
                for stage in ("ttir", "ttgir", "llir", "ptx"):
                    if stage in compiled.asm:
                        (output / f"{name}.{stage}").write_text(compiled.asm[stage])
                meta = compiled.metadata._asdict()
                (output / f"{name}.metadata.json").write_text(json.dumps(meta, indent=2, default=vars) + "\n")
                assembler_log = log.getvalue()
                (output / f"{name}.ptxas.txt").write_text(assembler_log)
                def count(pattern):
                    match = re.search(pattern, assembler_log)
                    return int(match.group(1)) if match else None
                item = {"name": name, "num_warps": warps, **labels,
                        "registers": count(r"Used (\d+) registers"), "shared_bytes": meta["shared"],
                        "spill_store_bytes": count(r"(\d+) bytes spill stores"),
                        "spill_load_bytes": count(r"(\d+) bytes spill loads"),
                        "layouts": [line for line in compiled.asm["ttgir"].splitlines()
                                    if line.startswith("#") and "#ttg.blocked" in line]}
                report["variants"].append(item)
                print(json.dumps(item), flush=True)

            signature = {"X": "*bf16", "Y": "*bf16", "W": "*bf16", "B": "constexpr",
                         "Z": "*bf16", "Mean": "constexpr", "Rstd": "*fp32",
                         "stride_x_row": "i32", "stride_y_row": "i32", "stride_z_row": "i32",
                         "M": "i32", "N": "constexpr", "eps": "fp32", "BLOCK_N": "constexpr",
                         "ROWS_PER_BLOCK": "constexpr", "HAS_BIAS": "constexpr", "HAS_Z": "constexpr",
                         "NORM_BEFORE_GATE": "constexpr", "IS_RMS_NORM": "constexpr", "ACTIVATION": "constexpr"}
            for rows in (1, 2, 4):
                constants = dict(B=None, Mean=None, N=128, BLOCK_N=128, ROWS_PER_BLOCK=rows,
                                 HAS_BIAS=False, HAS_Z=True, NORM_BEFORE_GATE=True,
                                 IS_RMS_NORM=True, ACTIVATION="silu")
                source = ASTSource(baseline_fn, signature=signature, constexprs=constants,
                                   attrs={(i,): [["tt.divisibility", 16]] for i in (0, 1, 2, 4, 6, 7, 8, 9, 10)})
                compile_one(f"fla_rows{rows}", source, 1, rows_per_block=rows)
            for values in (4, 8):
                for warps in (4, 8, 16):
                    source = GluonASTSource(_norm_gate_fp8_explicit,
                        signature={"X": "*bf16", "Z": "*bf16", "W": "*bf16", "Q": "*fp8e4nv", "S": "*fp32",
                                   "stride_x_token": "constexpr", "stride_z_token": "constexpr", "EPS": "constexpr",
                                   "VALUES_PER_LANE": "constexpr", "WARPS": "constexpr"},
                        constexprs=dict(stride_x_token=4096, stride_z_token=12288, EPS=1e-6,
                                        VALUES_PER_LANE=values, WARPS=warps),
                        attrs={(i,): [["tt.divisibility", 16]] for i in range(5)})
                    compile_one(f"explicit_values{values}_warps{warps}", source, warps, values_per_lane=values)
            report["status"] = "compiled_offline"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["cuda_initialized_after"] = torch.cuda.is_initialized()
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    main()
