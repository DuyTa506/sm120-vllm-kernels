#!/usr/bin/env python3
"""Compile the unchanged Triton prototype for SM120 without a CUDA context.

Run inside the pinned qwen-prefill-dev Python environment. CUDA is hidden;
runtime-driver and torch device-query entrypoints fail immediately. The
script extracts only the JIT function from fused.py, avoiding vLLM imports.
No kernel handle is initialized and no kernel is launched.
"""

import argparse
import ast
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import traceback
from types import ModuleType
from unittest.mock import patch


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=here / "fused.py")
    parser.add_argument("--output-dir", type=Path, default=here / "results/offline_sm120")
    parser.add_argument("--warps", type=int, nargs="+", choices=(4, 8, 16), default=[4, 8, 16])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TRITON_DUMP_PTXAS_LOG"] = "1"
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    import torch

    assert not torch.cuda.is_initialized(), "CUDA was already initialized"
    attempts = []

    def forbidden(name):
        def fail(*args, **kwargs):
            attempts.append(name)
            raise RuntimeError(f"Offline compilation attempted runtime CUDA access: {name}")
        return fail

    class ForbiddenDriver:
        def __getattr__(self, name):
            return forbidden("triton.driver." + name)()

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pending", "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "cuda_visible_devices": "", "cuda_initialized_before": False,
        "target": {"backend": "cuda", "arch": 120, "warp_size": 32},
        "specialization": {"stride_x_token": 4096, "stride_z_token": 12288, "EPS": 1e-6,
                           "pointer_alignment_bytes": 16},
        "kernel_launches": 0, "device_queries": 0, "runtime_access_attempts": attempts,
        "variants": [],
        "scope": "Offline code generation only; numerical correctness, launch feasibility, occupancy and speed remain unverified.",
    }
    try:
        with tempfile.TemporaryDirectory(prefix="gdn-offline-triton-") as cache, ExitStack() as stack:
            os.environ["TRITON_CACHE_DIR"] = cache
            for name in ("_lazy_init", "init", "is_available", "device_count", "current_device",
                         "get_device_properties", "get_device_capability", "get_device_name",
                         "current_stream", "default_stream", "set_device", "synchronize"):
                stack.enter_context(patch.object(torch.cuda, name, forbidden("torch.cuda." + name)))
            import triton
            import triton.language as tl
            from triton.backends.compiler import GPUTarget
            from triton.compiler import ASTSource
            from triton.runtime.driver import driver

            driver.set_active(ForbiddenDriver())
            report["torch_version"] = torch.__version__
            report["triton_version"] = triton.__version__
            tree = ast.parse(args.source.read_text(), filename=str(args.source))
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_norm_gate_fp8")
            module = ModuleType("gdn_offline_jit_source")
            module.__file__ = str(args.source)
            module.__dict__.update(triton=triton, tl=tl)
            sys.modules[module.__name__] = module
            exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                         str(args.source), "exec"), module.__dict__)
            source = ASTSource(
                module._norm_gate_fp8,
                signature={"X": "*bf16", "Z": "*bf16", "W": "*bf16", "Q": "*fp8e4nv", "S": "*fp32",
                           "stride_x_token": "constexpr", "stride_z_token": "constexpr", "EPS": "constexpr"},
                constexprs={"stride_x_token": 4096, "stride_z_token": 12288, "EPS": 1e-6},
                attrs={(i,): [["tt.divisibility", 16]] for i in range(5)},
            )
            for warps in args.warps:
                log = io.StringIO()
                with redirect_stdout(log):
                    compiled = triton.compile(source, target=GPUTarget("cuda", 120, 32),
                                              options={"num_warps": warps, "enable_fp_fusion": True})
                assert compiled.module is None and compiled.function is None and compiled._run is None
                assert not torch.cuda.is_initialized()
                prefix = args.output_dir / f"warps{warps}"
                for stage in ("ttir", "ttgir", "llir", "ptx"):
                    prefix.with_suffix("." + stage).write_text(compiled.asm[stage])
                metadata = compiled.metadata._asdict()
                prefix.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, default=vars) + "\n")
                assembler_log = log.getvalue()
                prefix.with_suffix(".ptxas.txt").write_text(assembler_log)
                ptx, ttir, ttgir = (compiled.asm[s] for s in ("ptx", "ttir", "ttgir"))

                def number(pattern):
                    match = re.search(pattern, assembler_log)
                    return int(match.group(1)) if match else None

                variant = {
                    "num_warps": warps, "registers_per_thread": number(r"Used (\d+) registers"),
                    "stack_frame_bytes": number(r"(\d+) bytes stack frame"),
                    "spill_store_bytes": number(r"(\d+) bytes spill stores"),
                    "spill_load_bytes": number(r"(\d+) bytes spill loads"),
                    "shared_memory_bytes": metadata.get("shared"),
                    "global_scratch_bytes": metadata.get("global_scratch_size"),
                    "ptx_target": next((line.strip() for line in ptx.splitlines() if line.startswith(".target")), None),
                    "ptx_bf16_conversion_instructions": sorted(set(re.findall(r"\bcvt\.[^\s;]*bf16[^\s;]*", ptx))),
                    "ptx_fp8_conversion_instructions": sorted(set(re.findall(r"\bcvt\.[^\s;]*e4m3[^\s;]*", ptx))),
                    "ttir_rounding_lines": [line.strip() for line in ttir.splitlines()
                                            if ("arith.truncf" in line or "arith.extf" in line)
                                            and ("bf16" in line or "f8E4M3FN" in line)],
                    "ttgir_layouts": [line for line in ttgir.splitlines() if line.startswith("#") and "#ttg." in line],
                    "ttgir_reductions": [line.strip() for line in ttgir.splitlines() if '"tt.reduce"' in line or "tt.reduce" in line],
                    "cuda_initialized": torch.cuda.is_initialized(), "kernel_handles_initialized": False,
                }
                report["variants"].append(variant)
                print(json.dumps({"warps": warps, "registers": variant["registers_per_thread"],
                                  "shared": variant["shared_memory_bytes"],
                                  "spill_stores": variant["spill_store_bytes"],
                                  "spill_loads": variant["spill_load_bytes"], "cuda_initialized": False}), flush=True)
            report["status"] = "compiled_offline"
    except Exception as exc:
        report["status"] = "pending_compile_failure"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["cuda_initialized_after"] = torch.cuda.is_initialized()
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, default=vars) + "\n")
        assert not torch.cuda.is_initialized(), "Offline compiler initialized CUDA"


if __name__ == "__main__":
    main()
