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
    config = dict(base)
    splits = list(config.get("splitting_ops") or standard)
    # Preserve all explicit entries and all baseline stateful split operations.
    for name in [*standard, OP]:
        if name not in splits:
            splits.append(name)
    config["splitting_ops"] = splits
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
            "candidate": result, "cuda_imported": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
