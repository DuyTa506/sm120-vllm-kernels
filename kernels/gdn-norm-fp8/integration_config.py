#!/usr/bin/env python3
"""CPU-only: add the experimental split op before constructing VllmConfig.

Use the emitted JSON as the candidate's --compilation-config. No imports of
Torch/vLLM and no edits to installed sources, running servers or defaults.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

OP = "vllm::gdn_attention_core_norm_fp8_packed_experimental"


def add_split(base, source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CompilationConfig")
    item = next(n for n in cls.body if isinstance(n, ast.AnnAssign)
                and isinstance(n.target, ast.Name) and n.target.id == "_attention_ops")
    standard = ast.literal_eval(item.value)
    # _attention_ops is not the completed baseline. With default splitting_ops
    # and no Inductor graph partition, vLLM later appends both KV-update ops.
    # Supplying our explicit list bypasses that completion, so preserve it here.
    contains_kv = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                       and n.name == "splitting_ops_contain_kv_cache_update")
    kv_assignment = next(n for n in contains_kv.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "kv_cache_update_ops"
                                 for t in n.targets))
    standard = [*standard, *ast.literal_eval(kv_assignment.value)]
    config = dict(base)
    if config.get("use_inductor_graph_partition") not in (None, False):
        raise ValueError("GDN experiment requires the ordinary Dynamo split configuration")
    supplied = config.get("splitting_ops")
    if supplied is not None and not set(standard).issubset(supplied):
        raise ValueError("Explicit base splitting_ops must preserve all19 completed baseline splits")
    splits = list(standard if supplied is None else supplied)
    if OP not in splits:
        splits.append(OP)
    config["splitting_ops"] = splits
    passes = dict(config.get("pass_config") or {})
    for name in ("fuse_attn_quant", "fuse_rope_kvcache", "fuse_qk_norm_rope_kvcache",
                 "fuse_rope_kvcache_cat_mla"):
        if passes.get(name) not in (None, False):
            raise ValueError(f"GDN experiment requires baseline {name}=False")
    # The normal default predicate returns False because KV updates are split.
    # Pin this derived default so CLI and resolved engine records are explicit.
    passes["fuse_rope_kvcache_cat_mla"] = False
    config["pass_config"] = passes
    return config, standard


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-json", default="{}", help="existing compilation-config JSON, preserved")
    p.add_argument("--source", type=Path, default=Path("/usr/local/lib/python3.12/dist-packages/vllm/config/compilation.py"))
    p.add_argument("--manifest", type=Path)
    args = p.parse_args()
    source = args.source.read_text()
    result, standard = add_split(json.loads(args.base_json), source)
    print(json.dumps(result, separators=(",", ":")))
    if args.manifest:
        args.manifest.write_text(json.dumps({"source": str(args.source),
            "sha256": hashlib.sha256(source.encode()).hexdigest(), "standard_splits": standard,
            "baseline_split_count": len(standard), "candidate_split_count": len(result["splitting_ops"]),
            "candidate": result, "cuda_imported": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
