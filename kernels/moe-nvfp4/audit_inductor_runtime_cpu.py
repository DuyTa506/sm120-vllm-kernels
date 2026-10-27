#!/usr/bin/env python3
"""Read matched-3 Inductor artifacts; never unpickle, import Torch, or execute kernels.

Run inside the existing dev container with CUDA_VISIBLE_DEVICES= because model
cache files are owned by root. Only this script's JSON output is written.
"""
import argparse
import ast
import hashlib
import json
import pickletools
import re
import struct
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def elf_sections(data):
    assert data[:6] == b"\x7fELF\x02\x01", "Expected little-endian ELF64 CUDA binary"
    offset = struct.unpack_from("<Q", data, 40)[0]
    stride, count, string_index = struct.unpack_from("<HHH", data, 58)
    headers = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * stride)
               for i in range(count)]
    names_header = headers[string_index]
    names = data[names_header[4]:names_header[4] + names_header[5]]
    return {names[h[0]:].split(b"\0", 1)[0].decode(): {
        "flags": h[2], "size": h[5], "sha256": digest(data[h[4]:h[4] + h[5]])}
        for h in headers}


def function_body(source):
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    return ast.dump(ast.Module(body=fn.body, type_ignores=[]), include_attributes=False)


def nested_opcodes(data, ancestry=()):
    # genops interprets syntax only; GLOBAL/REDUCE/BUILD are never executed.
    ops = list(pickletools.genops(data))
    yield ancestry, data, ops
    for op, arg, pos in ops:
        if isinstance(arg, bytes) and arg.startswith(b"\x80\x04"):
            yield from nested_opcodes(arg, ancestry + (pos,))


def string_tokens(ops):
    """Resolve memoized literal strings only; retain every opcode as evidence."""
    memo = {}
    previous = None
    result = []
    for op, arg, pos in ops:
        value = arg
        if op.name in {"BINGET", "LONG_BINGET", "GET"}:
            value = memo.get(int(arg))
        if op.name == "MEMOIZE":
            memo[len(memo)] = previous if isinstance(previous, str) else None
        elif op.name in {"BINPUT", "LONG_BINPUT", "PUT"}:
            memo[int(arg)] = previous if isinstance(previous, str) else None
        result.append((op.name, value, pos))
        previous = value
    return result


def next_integer(tokens, label):
    indexes = [i for i, (_, value, _) in enumerate(tokens) if value == label]
    if not indexes:
        return None
    i = indexes[0] + 1
    while tokens[i][0] == "MEMOIZE":
        i += 1
    name, value, _ = tokens[i]
    assert name.startswith("BININT") and isinstance(value, int), (label, tokens[i])
    return value


def runtime_records(model, names, wrapper_hashes):
    records = []
    model_bytes = model.read_bytes()
    for ancestry, blob, ops in nested_opcodes(model_bytes):
        wrappers = [arg for _, arg, _ in ops if isinstance(arg, str)
                    and arg.startswith('r"""\nCompile-time auto-tuning block:')]
        wrapper_digests = {digest(value.encode()) for value in wrappers}
        if not wrapper_hashes.intersection(wrapper_digests):
            continue
        tokens = string_tokens(ops)
        # Literal names identify individual kernel records. Later memo references
        # can refer to metadata/launchers and must not create duplicate records.
        for i, (op, arg, pos) in enumerate(ops):
            if not isinstance(arg, str) or arg not in names:
                continue
            starts = [j for j in range(i + 1, len(tokens))
                      if tokens[j][1] == "compile_results"]
            assert starts, (model, ancestry, arg)
            start = starts[0]
            end = next(j for j in range(start + 1, len(tokens))
                       if tokens[j][1] == "launchers")
            region = tokens[start:end]
            compiled_results = []
            for h, (_, value, _) in enumerate(region):
                if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                    continue
                config_start = next(j for j in range(h + 1, len(region)) if region[j][1] == "kwargs")
                config_end = next(j for j in range(config_start + 1, len(region))
                                  if region[j][1] == "compile_meta")
                conf = region[config_start:config_end]
                config = {key: next_integer(conf, key)
                          for key in ("XBLOCK", "R0_BLOCK", "num_warps", "num_stages")}
                config = {key: value for key, value in config.items() if value is not None}
                assert config["num_warps"] == next_integer(region[h:config_start], "num_warps")
                assert "XBLOCK" in config
                compiled_results.append({"compiled_hash": value, "config": config,
                    "shared_bytes": next_integer(region[h:config_start], "shared")})
            assert compiled_results
            # Each linked blob stores its wrapper source as a string. Match its
            # full content hash to the separately saved .py file below.
            records.append({
                "model_sha256": digest(model_bytes), "nested_byte_offsets": ancestry,
                "nested_pickle_sha256": digest(blob), "kernel_opcode_offset": pos,
                "kernel_name": arg, "compiled_results": compiled_results,
                "selection_status": ("one_serialized_candidate" if len(compiled_results) == 1
                                     else "multiple_serialized_candidates_runtime_unresolved"),
                "wrapper_source_sha256": sorted(wrapper_digests),
                "opcode_evidence": [(name, value, offset) for name, value, offset in region
                                    if isinstance(value, (str, int))],
            })
    return records


def audit(root):
    report_path = root / "results/qwen_sequential/scalar_vs_exact_matched_3.greedy.json"
    report = json.loads(report_path.read_text())
    changes = report["runtime_changes"]["inductor_config_changes"]["changed"]
    assert len(changes) == 4
    output = {"schema_version": 1, "mode": "CPU_file_and_pickle_opcode_inspection_only",
              "script_sha256": digest(Path(__file__).read_bytes()),
              "report_sha256": digest(report_path.read_bytes()), "entries": {},
              "limitations": ["Saved model artifacts, not per-request runtime tracing.",
                              "Only kernels linked to the four changed tuning entries are audited.",
                              "No claim of global determinism or model quality equivalence."]}
    for side, variant in [("baseline", "shared_exact"), ("candidate", "scalar")]:
        cache = root / f".cache/vllm/qwen-sequential-{variant}"
        pyfiles = list((cache / "torch_compile_cache/torch_aot_compile").rglob("*.py"))
        texts = {p: p.read_text() for p in pyfiles}
        bodies = {}
        for p, source in texts.items():
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and "def triton_" in node.value:
                    try:
                        body = function_body(node.value)
                    except (SyntaxError, StopIteration):
                        continue
                    embedded = ast.parse(node.value)
                    name = next(n.name for n in embedded.body if isinstance(n, ast.FunctionDef))
                    bodies.setdefault(body, []).append({"path": str(p.relative_to(cache)),
                        "line": node.lineno, "kernel_name": name, "sha256": digest(source.encode())})
        target_names = set()
        for key, change in changes.items():
            recorded, = change[side]
            best_path = cache / recorded["path"]
            assert digest(best_path.read_bytes()) == recorded["sha256"], best_path
            generated, = best_path.parent.glob("*.py")
            source = texts[generated]
            matches = bodies.get(function_body(source), [])
            target_names.update(m["kernel_name"] for m in matches)
            output["entries"].setdefault(key, {})[side] = {
                "best_config_path": str(best_path.relative_to(root)),
                "best_config_sha256": recorded["sha256"],
                "best_config": json.loads(best_path.read_text()),
                "generated_path": str(generated.relative_to(root)),
                "generated_sha256": digest(source.encode()), "generated_source": source,
                "matching_runtime_wrappers": matches,
            }
        all_records = []
        wrapper_hashes = {match["sha256"] for entry in output["entries"].values()
                          for match in entry[side]["matching_runtime_wrappers"]}
        for model in cache.glob("torch_compile_cache/torch_aot_compile/*/rank_0_0/model"):
            recs = runtime_records(model, target_names, wrapper_hashes)
            for rec in recs:
                rec["model_path"] = str(model.relative_to(root))
            all_records.extend(recs)
        metadata = {}
        for path in cache.rglob("*.json"):
            if path.name.startswith("__grp__"):
                continue
            data = json.loads(path.read_text())
            if "hash" in data and "name" in data:
                binary = path.with_suffix(".cubin")
                metadata[data["hash"]] = {"metadata_path": str(path.relative_to(root)),
                    "metadata_sha256": digest(path.read_bytes()), "name": data["name"],
                    "cubin_sha256": digest(binary.read_bytes()), "num_warps": data["num_warps"],
                    "elf_sections": elf_sections(binary.read_bytes())}
        for key, entry in output["entries"].items():
            value = entry[side]
            wrappers = {m["sha256"] for m in value["matching_runtime_wrappers"]}
            names = {m["kernel_name"] for m in value["matching_runtime_wrappers"]}
            recs = [r for r in all_records if r["kernel_name"] in names
                    and wrappers.intersection(r["wrapper_source_sha256"])]
            for rec in recs:
                for compiled in rec["compiled_results"]:
                    compiled["binary_metadata"] = metadata[compiled["compiled_hash"]]
                    assert compiled["config"]["num_warps"] == compiled["binary_metadata"]["num_warps"]
            value["serialized_runtime_records"] = recs
            value["all_matching_wrappers_resolved"] = wrappers == {
                sha for r in recs for sha in r["wrapper_source_sha256"] if sha in wrappers}
            assert value["all_matching_wrappers_resolved"]
    for entry in output["entries"].values():
        assert entry["baseline"]["generated_sha256"] == entry["candidate"]["generated_sha256"]
        def candidates(side):
            return {r["compiled_hash"]: r for record in entry[side]["serialized_runtime_records"]
                    for r in record["compiled_results"]}
        base, candidate = candidates("baseline"), candidates("candidate")
        entry["serialized_compiled_config_sets_equal"] = (
            {k: v["config"] for k, v in base.items()} ==
            {k: v["config"] for k, v in candidate.items()})
        entry["binary_comparison"] = {}
        for key in base.keys() & candidate.keys():
            a, b = base[key]["binary_metadata"], candidate[key]["binary_metadata"]
            sa, sb = a["elf_sections"], b["elf_sections"]
            entry["binary_comparison"][key] = {
                "complete_cubin_bytes_equal": a["cubin_sha256"] == b["cubin_sha256"],
                "executable_sections_equal": (
                    {n: v for n, v in sa.items() if v["flags"] & 4} ==
                    {n: v for n, v in sb.items() if v["flags"] & 4}),
                "different_sections": sorted(n for n in sa.keys() | sb.keys()
                                             if sa.get(n) != sb.get(n)),
            }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "pass", "entries": len(result["entries"]),
        "runtime_records_each_side": {k: len(v["baseline"]["serialized_runtime_records"])
                                      for k, v in result["entries"].items()},
        "all_linked_compiled_config_sets_equal": all(
            v["serialized_compiled_config_sets_equal"] for v in result["entries"].values()),
        "all_candidate_executable_sections_equal": all(
            c["executable_sections_equal"] for v in result["entries"].values()
            for c in v["binary_comparison"].values()),
        "final_in_memory_selection_fully_resolved": False}, indent=2))
