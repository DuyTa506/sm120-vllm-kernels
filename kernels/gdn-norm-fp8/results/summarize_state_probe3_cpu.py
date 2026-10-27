#!/usr/bin/env python3
"""Reproduce probe3 evidence validation and source preservation; CPU stdlib only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
RUN = RESULTS / "packed_state_probe_3"
manifest = json.loads((RUN / "manifest.json").read_text())
worker = json.loads((RUN / "worker_report.json").read_text())
runner = json.loads((RESULTS / "packed_state_probe_3.runner.json").read_text())
assert manifest["status"] == worker["status"] == "complete"
assert runner["exit_code"] == 0 and runner["cleanup"]["verified_stopped"]
assert runner["gpu_slot_release_confirmed"] and not (RESULTS / "packed_state_probe_3_gpu_after.txt").read_text().strip()
assert all(worker["coverage_requirements"].values())
assert len(worker["cases"]) == 3
cases = []
for case in worker["cases"]:
    assert len(case["variants"]) == 2
    assert all(value == 0 for value in case["live_cache_byte_mismatches"])
    for variant in case["variants"]:
        mismatch = variant["mismatch"]
        assert mismatch["fp8_bytes"] == mismatch["scale_bytes"] == 0
        assert all(value == 0 for value in mismatch["cache_bytes"])
        assert variant["padding_invariant"] and variant["tail_zero_floor"]
    cases.append({"kind": case["kind"], "metadata": case["metadata"],
                  "variants": case["variants"], "live_cache_byte_mismatches": case["live_cache_byte_mismatches"]})
assert [p["name"] for p in manifest["phases"]] == ["mixed_batch", "isolated_long"]
assert manifest["phases"][0]["prompt_lengths"] == [64, 8192]
assert manifest["phases"][1]["prompt_lengths"] == [8192]
salt = manifest["phases"][1]["cache_salts"][0]
assert salt and salt not in manifest["phases"][0]["cache_salts"]
assert [len(ids) for ids in manifest["output_token_ids"]] == [32, 32, 1]

snapshot = RESULTS / "packed_state_probe_3_sources"
snapshot.mkdir(exist_ok=True)
source_inventory = {}
for relative, expected in manifest["api"]["probe_sources"].items():
    original = HERE / relative
    assert original.resolve().is_relative_to(HERE)
    assert hashlib.sha256(original.read_bytes()).hexdigest() == expected, relative
    target = snapshot / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert hashlib.sha256(target.read_bytes()).hexdigest() == expected
    else:
        shutil.copy2(original, target)
    source_inventory[relative] = expected
(snapshot / "verified_sources.json").write_text(json.dumps(source_inventory, indent=2) + "\n")
report = {"status": "PASS_SCOPED_STATE_PARITY", "exit_code": 0,
          "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
          "scope": "Layer18 actual eager MTP2 owner/private compact caches, baseline auto MoE activations, exact original packed core plus explicitly compiled native QuantFP8 oracle",
          "checked_cases": len(cases), "checked_padding_variants": sum(len(c["variants"]) for c in cases),
          "cases": cases, "observed": worker["observed"], "calls": worker["calls"],
          "coverage_requirements": worker["coverage_requirements"],
          "checkpoint_observed": worker["checkpoint_observed"], "mixed_non_spec_observed": "mixed_non_spec" in worker["observed"],
          "phases": manifest["phases"], "cleanup": runner["cleanup"],
          "gpu_compute_after": "", "source_snapshot": str(snapshot.relative_to(HERE)),
          "verified_probe_source_count": len(source_inventory), "source_sha256": source_inventory,
          "limitations": ["No checkpoint-bearing metadata or non-speculative mixed case observed",
                          "No CUDA graph replay in this eager probe; separate integration smoke covers compiled execution",
                          "Private compact caches do not reproduce live allocator identity or async scheduler behavior",
                          "No full-model paired output/NLL equivalence or performance measurement"]}
(RESULTS / "packed_state_probe_3_summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"status": report["status"], "cases": len(cases), "variants": 6,
                  "verified_sources": len(source_inventory), "checkpoint_observed": worker["checkpoint_observed"]}))
