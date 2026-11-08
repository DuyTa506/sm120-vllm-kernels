"""Fail closed on missing shadow coverage, errors, or unexercised preemption."""
import argparse
import json
import math
from pathlib import Path
import re


def metric(path, name):
    values = [float(line.split()[-1]) for line in path.read_text().splitlines()
              if line.startswith(name + "{") or line.startswith(name + " ")]
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError(f"Missing or invalid counter {name} in {path}")
    return sum(values)


def evaluate(directory, mode):
    text = (directory / "coverage.log").read_text()
    rows = [tuple(map(int, match)) for match in re.findall(
        r"ROUND2 rows=(\d+) bad=(\d+) boundary_rows=(\d+) boundary_bad=(\d+) armed_bad=(\d+) out_bad=(\d+)", text)]
    preemptions = metric(directory / "metrics_after.txt", "vllm:num_preemptions_total") - metric(directory / "metrics_before.txt", "vllm:num_preemptions_total")
    if preemptions < 0:
        raise ValueError("Preemption counter decreased; observations are not comparable")
    result = dict(mode=mode, checked_rows=sum(r[0] for r in rows), boundary_rows=sum(r[2] for r in rows),
                  preemptions=preemptions, numerical_bad=any(r[1] or r[3] or r[4] or r[5] for r in rows),
                  fatal="Traceback (most recent call last)" in text or "round2 shadow mismatch" in text)
    runtime_path=directory / "serving/round2_runtime/runtime.json"
    runtime=json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    result["composition_sha256"]=runtime.get("composition_sha256")
    result["passed"] = bool(result["composition_sha256"] and runtime.get("fp32_layers",0)==30 and runtime.get("composition_sources_verified") is True and result["checked_rows"] and not result["numerical_bad"] and not result["fatal"]
                            and (result["boundary_rows"] > 0 if mode == "shadow" else preemptions > 0))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("mode", choices=["shadow", "preempt"])
    args = parser.parse_args()
    result = evaluate(args.directory, args.mode)
    (args.directory / "coverage_gate.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit("required current-stack shadow/preemption coverage missing or failed")
