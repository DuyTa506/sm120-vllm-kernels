"""Audit real checkpoint scale arithmetic on CPU; never initialize CUDA.

Uses the installed FlashInferExperts.process_weights_after_loading method as
reference and extracts the actual candidate/shared alpha assignment ASTs.
Run inside the pinned image, with CUDA_VISIBLE_DEVICES= for additional isolation.
"""

import argparse
import ast
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
RELATIVE = Path("vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py")


def method(path, class_name, method_name):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def alpha_code(path):
    node = method(path, "FlashInferB12xExperts", "_process_weights_fixed_scales")
    assignments = [item for item in ast.walk(node) if isinstance(item, ast.Assign)
                   and any(isinstance(target, ast.Attribute) and target.attr in ("_cutlass_g1", "_cutlass_g2") for target in item.targets)]
    assert len(assignments) == 2
    return compile(ast.fix_missing_locations(ast.Module(body=assignments, type_ignores=[])), str(path), "exec")


def compare(actual, reference):
    ulp = (actual.view(torch.int32).to(torch.int64) - reference.view(torch.int32).to(torch.int64)).abs()
    return {"count": actual.numel(), "different": int((ulp != 0).sum()),
            "max_ulp": int(ulp.max()),
            "max_relative": float(((actual - reference).abs() / reference.abs()).max())}


def audit(snapshot=None, candidate=None):
    torch.set_num_threads(1)
    snapshot = snapshot or next(Path("/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots").iterdir())
    candidate = candidate or HERE / "shared_exact" / RELATIVE
    baseline = HERE / "shared" / RELATIVE
    vllm_root = Path(importlib.util.find_spec("vllm").origin).parent
    reference_path = vllm_root / "model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py"
    reference_method = method(reference_path, "FlashInferExperts", "process_weights_after_loading")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[reference_method], type_ignores=[]), str(reference_path), "exec"), namespace)
    reference_process = namespace["process_weights_after_loading"]
    codes = {"shared": alpha_code(baseline), "exact": alpha_code(candidate)}
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefixes = sorted({name.split(".experts.")[0] for name in weight_map
                       if ".experts." in name and name.endswith(".weight_global_scale")})
    rows = []
    with ExitStack() as stack:
        handles = {}
        def get(name):
            shard = weight_map[name]
            if shard not in handles:
                handles[shard] = stack.enter_context(safe_open(snapshot / shard, framework="pt", device="cpu"))
            return handles[shard].get_tensor(name).float().reshape(())
        for prefix in prefixes:
            experts = sorted({int(name.split(".experts.")[1].split(".")[0]) for name in weight_map
                              if name.startswith(prefix + ".experts.") and name.endswith(".weight_global_scale")})
            def scales(projection, kind):
                return torch.stack([get(f"{prefix}.experts.{e}.{projection}.{kind}_global_scale") for e in experts])
            w1, w3, w2 = [scales(projection, "weight") for projection in ("gate_proj", "up_proj", "down_proj")]
            a1, a3, a2_raw = [scales(projection, "input") for projection in ("gate_proj", "up_proj", "down_proj")]
            w13_s2, w2_s2 = 1.0 / w1, 1.0 / w2
            # Matches loader reciprocal, shared activation max over gate/up
            # and all experts, repeat(E), then quant-config reciprocal.
            a13 = (1.0 / torch.stack([a1, a3], dim=1)).max().repeat(len(experts))
            a2 = (1.0 / a2_raw).max().repeat(len(experts))
            layer = SimpleNamespace(w13_weight_scale_2=w13_s2.clone(), w2_weight_scale_2=w2_s2.clone(),
                                    w13_input_scale=a13, w2_input_scale=a2)
            ref = SimpleNamespace(quant_config=SimpleNamespace(use_nvfp4_w4a4=True))
            reference_process(ref, layer)
            reference = (layer.w13_weight_scale_2.clone(), layer.w2_weight_scale_2.clone())
            outputs = {}
            for name, code in codes.items():
                result = SimpleNamespace()
                exec(code, {"torch": torch}, {"self": result, "layer": layer,
                     "w13_s2": w13_s2, "w2_s2": w2_s2, "a1g": 1.0 / a13, "a2g": 1.0 / a2})
                outputs[name] = (result._cutlass_g1, result._cutlass_g2)
            row = {"layer": prefix, "experts": len(experts),
                   "gate_up_weight_scale_equal": bool(torch.equal(w1, w3)),
                   "gate_up_input_scale_equal": bool(torch.equal(a1, a3))}
            for index, label in enumerate(("g1", "g2")):
                row[label] = {name: compare(values[index], reference[index]) for name, values in outputs.items()}
            rows.append(row)
    summary = {label: {name: {
        "count": sum(row[label][name]["count"] for row in rows),
        "different": sum(row[label][name]["different"] for row in rows),
        "layers_with_difference": sum(row[label][name]["different"] > 0 for row in rows),
        "max_ulp": max(row[label][name]["max_ulp"] for row in rows),
        "max_relative": max(row[label][name]["max_relative"] for row in rows),
    } for name in codes} for label in ("g1", "g2")}
    return {"checkpoint": str(snapshot), "device": "cpu", "cuda_initialized": torch.cuda.is_initialized(),
            "layers": len(rows), "reference": str(reference_path),
            "sources_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (reference_path, baseline, candidate)},
            "summary": summary, "per_layer": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    result = audit(args.snapshot, args.candidate)
    print(json.dumps(result, indent=2))
    assert not result["cuda_initialized"]
    assert all(result["summary"][label]["exact"]["different"] == 0 for label in ("g1", "g2")), "candidate alpha differs from actual default CUTLASS processing"


if __name__ == "__main__":
    main()
