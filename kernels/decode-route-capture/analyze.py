"""Analyze signed raw routes without loading torch or touching a GPU."""
import argparse
import gzip
import json
from pathlib import Path


def expert_bytes(geometry):
    """Full gate/up/down weight payload, including NVFP4 block scales."""
    elements = 3 * geometry["hidden"] * geometry["intermediate"]
    if geometry["format"] == "nvfp4":
        if elements % 16:
            raise ValueError("NVFP4 geometry is not block aligned")
        theoretical = elements // 2 + elements // 16
    elif geometry["format"] == "bf16":
        theoretical = elements * 2
    else:
        raise ValueError(f"unsupported weight format: {geometry['format']}")
    return geometry.get("stored_payload_bytes_per_expert", theoretical)


def classify_rows(record):
    n, real = record["num_rows"], record["num_real_rows"]
    starts, rejected = record["query_start_loc"], record["num_rejected"]
    if (any(type(x) is not int for x in [n, real, *starts, *rejected])
            or not 0 <= real <= n or not starts or starts[0] != 0 or starts[-1] != real
            or len(starts) != len(record["request_ids"]) + 1
            or len(rejected) != len(record["request_ids"])
            or len(set(record["request_ids"])) != len(record["request_ids"])):
        raise ValueError("inconsistent row/request boundaries")
    labels = ["graph_padding"] * n
    for start, end, count in zip(starts[:-1], starts[1:], rejected):
        if not start < end or not 0 <= count < end - start:
            raise ValueError("invalid request interval or rejected suffix")
        labels[start:end - count] = ["required"] * (end - start - count)
        labels[end - count:end] = ["rejected"] * count
    padding = record.get("is_padding")
    if padding is not None and padding != [i >= real for i in range(n)]:
        raise ValueError("unexpected non-tail graph padding; this probe excludes adaptive verification")
    return labels


def analyze_record(record, geometry):
    if record["phase"] not in ("target_verify", "draft_prefill", "draft_decode"):
        raise ValueError("unknown capture phase")
    if record["ids_dtype"] != "int32":
        raise ValueError("raw route IDs must be signed int32; uint8 loses the -1 sentinel")
    labels = classify_rows(record)
    ids, layer_ids = record["ids"], record["layer_ids"]
    if len(ids) != record["num_rows"] or any(len(row) != len(layer_ids) for row in ids):
        raise ValueError("route tensor shape does not match metadata")
    geometries = {g["layer_id"]: g for g in geometry}
    if len(set(layer_ids)) != len(layer_ids) or set(layer_ids) != set(geometries):
        raise ValueError("captured layers do not match live geometry")
    draft = record["phase"].startswith("draft_")
    if draft:
        present, mask = record.get("router_mask_present"), record.get("router_is_padding")
        if (not isinstance(present, list) or len(present) != len(layer_ids)
                or any(type(x) is not int or x not in (0, 1) for x in present)
                or not isinstance(mask, list) or len(mask) != len(ids)
                or any(len(row) != len(layer_ids) or any(type(x) is not bool for x in row)
                       for row in mask)):
            raise ValueError("missing or malformed actual draft router mask metadata")
    layers = []
    for slot, layer_id in enumerate(layer_ids):
        geom = geometries[layer_id]
        row_sets = []
        for row_index, row in enumerate(ids):
            values = row[slot]
            if len(values) != geom["topk"] or any(
                    not isinstance(x, int) or isinstance(x, bool)
                    or not -1 <= x < geom["experts"] for x in values):
                raise ValueError("invalid expert ID/top-k width or unwritten capture buffer")
            if record["phase"] == "target_verify":
                if labels[row_index] == "graph_padding" and any(x != -1 for x in values):
                    raise ValueError("target padding unexpectedly activates experts")
                if labels[row_index] != "graph_padding" and any(x < 0 for x in values):
                    raise ValueError("real target row contains invalid expert IDs")
            else:
                masked = mask[row_index][slot]
                if masked and (not present[slot] or labels[row_index] == "required"):
                    raise ValueError("draft mask is absent or masks a required row")
                if masked and any(x != -1 for x in values):
                    raise ValueError("masked draft row unexpectedly activates experts")
                if not masked and any(x < 0 for x in values):
                    raise ValueError("unmasked draft row contains invalid expert IDs")
            valid = [x for x in values if x >= 0]
            if len(set(valid)) != len(valid):
                raise ValueError("duplicate top-k expert ID; possible unwritten capture buffer")
            row_sets.append({x for x in values if x >= 0})
        selected = lambda label: set().union(
            *(row_sets[i] for i, kind in enumerate(labels) if kind == label))
        required, rejected, tail = (selected(x) for x in ("required", "rejected", "graph_padding"))
        real = required | rejected
        all_experts = real | tail
        requests = []
        for request_id, start, end in zip(record["request_ids"], record["query_start_loc"][:-1],
                                          record["query_start_loc"][1:]):
            sets = row_sets[start:end]
            union = set().union(*sets)
            requests.append({
                "request_id": request_id, "rows": list(range(start, end)),
                "full_mtp2_group": record["phase"] == "target_verify" and len(sets) == 3,
                "union_experts": len(union), "expert_ids": sorted(union),
                "adjacent_intersections": [len(a & b) for a, b in zip(sets, sets[1:])],
                "lag2_intersections": [len(a & b) for a, b in zip(sets, sets[2:])],
            })
        layers.append({
            "layer_id": layer_id, "expert_ids": sorted(all_experts),
            "unique_experts_executed": len(all_experts),
            "unique_experts_nonpadding": len(real), "unique_experts_required": len(required),
            "unique_experts_rejected": len(rejected), "unique_experts_graph_tail": len(tail),
            "extra_experts_graph_tail": len(tail - real),
            "potentially_avoidable_draft_experts": (
                len((rejected | tail) - required) if record["phase"].startswith("draft_") else None),
            "bytes_per_expert": expert_bytes(geom),
            "weight_payload_bytes": len(all_experts) * expert_bytes(geom), "requests": requests,
            "router_mask_present": bool(present[slot]) if draft else True,
            "router_masked_rows": (sum(row[slot] for row in mask) if draft else
                                   labels.count("graph_padding")),
        })
    return {"step": record["step"], "phase": record["phase"],
            "num_rows": record["num_rows"], "num_real_rows": record["num_real_rows"],
            "row_classes": labels, "layers": layers}


def analyze_capture(manifest, records):
    results, previous, by_step, seen = [], {}, {}, set()
    acceptance = {"proposed_draft_tokens": 0, "accepted_draft_tokens": 0,
                  "rejected_draft_tokens": 0, "sampled_output_tokens": 0}
    target_bytes = 0
    for record in sorted(records, key=lambda r: (r["step"], r["phase"])):
        if type(record["step"]) is not int or record["step"] < 0:
            raise ValueError("invalid target step ID")
        key = record["step"], record["phase"]
        if key in seen:
            raise ValueError("duplicate route capture for a step/phase")
        seen.add(key)
        scope = "target" if record["phase"] == "target_verify" else "draft"
        result = analyze_record(record, manifest["geometry"][scope])
        for layer in result["layers"]:
            for group in layer["requests"]:
                key = (record["phase"], layer["layer_id"], group["request_id"])
                current = set(group["expert_ids"])
                if key in previous:
                    step, experts = previous[key]
                    group["previous_capture_intersection"] = len(current & experts)
                    group["previous_capture_step_gap"] = record["step"] - step
                previous[key] = record["step"], current
        results.append(result)
        by_step.setdefault(record["step"], {})[record["phase"]] = result
        if scope == "target":
            proposed = record["num_draft_tokens_per_req"]
            rejected, sampled = record["num_rejected"], record["num_sampled"]
            if len(proposed) != len(rejected) or len(sampled) != len(proposed):
                raise ValueError("missing acceptance metadata")
            if any(type(p) is not int or not 0 <= p <= 2 or not 0 <= r <= p
                   for p, r in zip(proposed, rejected)):
                raise ValueError("rejections exceed scheduled draft tokens")
            if any(type(n) is not int or n < 1 for n in sampled):
                raise ValueError("invalid sampler output count")
            starts = record["query_start_loc"]
            if any(end - start != p + 1 or n + r != p + 1
                   for start, end, p, n, r in zip(starts[:-1], starts[1:], proposed, sampled, rejected)):
                raise ValueError("pure-decode query width or acceptance count disagrees with MTP metadata")
            acceptance["proposed_draft_tokens"] += sum(proposed)
            acceptance["rejected_draft_tokens"] += sum(rejected)
            acceptance["accepted_draft_tokens"] += sum(proposed) - sum(rejected)
            acceptance["sampled_output_tokens"] += sum(sampled)
            target_bytes += sum(layer["weight_payload_bytes"] for layer in result["layers"])
    draft_pairs, missing = [], []
    for step, phases in sorted(by_step.items()):
        if "target_verify" not in phases:
            raise ValueError("draft capture has no matching target step")
        if not all(phase in phases for phase in ("draft_prefill", "draft_decode")):
            missing.append(step)
            continue
        first = {l["layer_id"]: set(l["expert_ids"]) for l in phases["draft_prefill"]["layers"]}
        for layer in phases["draft_decode"]["layers"]:
            a, b = first[layer["layer_id"]], set(layer["expert_ids"])
            draft_pairs.append({"step": step, "layer_id": layer["layer_id"],
                                "union_experts": len(a | b), "intersection_experts": len(a & b),
                                "unique_payload_bytes_both_phases": len(a | b) * layer["bytes_per_expert"]})
    sampled = acceptance["sampled_output_tokens"]
    proposed = acceptance["proposed_draft_tokens"]
    if missing and manifest["draft_capture"] == "captured_both_phases":
        raise ValueError("manifest claims complete draft capture but a phase is missing")
    acceptance["accepted_fraction"] = acceptance["accepted_draft_tokens"] / proposed if proposed else None
    return {"schema_version": 1, "instrumentation_only": True,
            "draft_capture": manifest["draft_capture"], "steps_missing_draft_pair": missing,
            "acceptance": acceptance, "target_weight_payload_bytes": target_bytes,
            "target_bytes_per_sampled_output_token": target_bytes / sampled if sampled else None,
            "records": results, "draft_phase_pairs": draft_pairs,
            "interpretation": "Unique weight+block-scale payload, not measured DRAM traffic. "
                "Rejected target verification rows are included. Sampler output counts may differ "
                "from client-emitted tokens at request termination. Cache residency is unmeasured."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.capture_dir / "manifest.json").read_text())
    with gzip.open(args.capture_dir / "records.jsonl.gz", "rt") as f:
        records = [json.loads(line) for line in f]
    result = analyze_capture(manifest, records)
    path = args.out or args.capture_dir / "analysis.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("records", "draft_phase_pairs")}, indent=2))
    print(path)


if __name__ == "__main__":
    main()
