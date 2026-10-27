#!/usr/bin/env python3
"""Apply experimental scalar/GDN patches only in disposable CPU-only trees."""
import argparse
import ast
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


MOE = "vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py"
ORACLE = "vllm/model_executor/layers/fused_moe/oracle/unquantized.py"
GDN = "vllm/model_executor/layers/mamba/gdn/"
GDN_FILES = [GDN + name for name in (
    "qwen_gdn_linear_attn.py", "gdn_norm_fp8_fusion.py", "gdn_norm_fp8_guards.py",
    "gdn_norm_fp8_kernel.py", "gdn_norm_fp8_quant_out.py")]
SCALAR_PATCH = "kernels/moe-nvfp4/06-moe-b12x-scalar-input-scale.patch"
GDN_PATCH = "kernels/gdn-norm-fp8/01-gdn-norm-fp8-experimental.patch"
SERIES = ["kernels/moe-nvfp4/" + name for name in (
    "03-moe-b12x-scale-fix-and-hybrid.patch", "04-moe-b12x-shared-workspace.patch",
    "05-moe-cutlass-exact-alphas.patch", "06-moe-b12x-scalar-input-scale.patch")]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def apply(tree, patch):
    command = ["patch", "--batch", "--forward", "--fuzz=0", "-p1",
               "-d", str(tree), "-i", str(patch)]
    records = []
    for dry_run in (True, False):
        result = subprocess.run(command + (["--dry-run"] if dry_run else []),
                                capture_output=True, text=True)
        require(result.returncode == 0, result.stdout + result.stderr)
        require(not any(word in (result.stdout + result.stderr).lower()
                        for word in ("offset", "fuzz", "failed", "reversed")),
                "Patch did not apply exactly: " + result.stdout + result.stderr)
        records.append({"dry_run": dry_run, "exit_code": result.returncode,
                        "stdout": result.stdout, "stderr": result.stderr})
    return records


def verify(root):
    source_files = {MOE: "kernels/moe-nvfp4/scalar/" + MOE,
                    **{p: "kernels/gdn-norm-fp8/overlay/" + p for p in GDN_FILES}}
    expected_bytes = {p: (root / source).read_bytes() for p, source in source_files.items()}
    baseline_files = {MOE: "kernels/moe-nvfp4/shared_exact/" + MOE,
                      GDN_FILES[0]: "build/" + GDN_FILES[0]}
    baseline_bytes = {p: (root / source).read_bytes() for p, source in baseline_files.items()}
    protected = {**{s: sha((root / s).read_bytes()) for s in source_files.values()},
                 **{s: sha((root / s).read_bytes()) for s in baseline_files.values()}}
    manifests = {}
    for name in ("scalar_matched_3", "exact_matched_3", "gdn_integration_2", "scalar_gdn_matched_3"):
        path = "results/qwen_sequential/" + name + "/source_manifest.json"
        data = (root / path).read_bytes()
        manifests[name] = {"path": path, "sha256": sha(data), "files": json.loads(data)["files"]}
    require(sha(baseline_bytes[MOE]) == manifests["exact_matched_3"]["files"][baseline_files[MOE]],
            "shared_exact baseline differs from measured source")
    overlay_manifest = json.loads((root / "kernels/gdn-norm-fp8/overlay_manifest.json").read_text())
    require(sha(baseline_bytes[GDN_FILES[0]]) == overlay_manifest["baseline_sha256"],
            "GDN baseline differs from overlay's declared next build")
    for path, source in source_files.items():
        name = "scalar_matched_3" if path == MOE else "gdn_integration_2"
        require(sha(expected_bytes[path]) == manifests[name]["files"][source],
                "Overlay differs from measured source: " + source)
        require(sha(expected_bytes[path]) == manifests["scalar_gdn_matched_3"]["files"][source],
                "Overlay differs from the composed run's startup source capture: " + source)
    for patch, targets in ((SCALAR_PATCH, [MOE]), (GDN_PATCH, GDN_FILES)):
        actual = [line.removeprefix("+++ b/") for line in (root / patch).read_text().splitlines()
                  if line.startswith("+++ b/")]
        require(actual == targets, "Unexpected patch targets: " + patch)

    def write(tree, contents):
        for path, data in contents.items():
            target = tree / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    def check(tree, contents):
        result = {}
        for path, expected in contents.items():
            actual = (tree / path).read_bytes()
            require(actual == expected, "Patched bytes differ: " + path)
            ast.parse(actual, filename=path)
            result[path] = sha(actual)
        return result

    scenarios = []
    # Prove the existing MoE sequence leads to the required scalar parent/output.
    with tempfile.TemporaryDirectory(prefix="qwen-patch-series-") as directory:
        tree = Path(directory)
        write(tree, {p: (root / "kernels/moe-nvfp4/orig" / p).read_bytes() for p in (MOE, ORACLE)})
        steps = []
        for patch, variant in zip(SERIES, ("patched", "shared", "shared_exact", "scalar")):
            logs = apply(tree, root / patch)
            target = (root / "kernels/moe-nvfp4" / variant / MOE).read_bytes()
            files = check(tree, {MOE: target, ORACLE: (root / "kernels/moe-nvfp4/patched" / ORACLE).read_bytes()})
            steps.append({"patch": patch, "matched_variant": variant, "files": files, "checks": logs})
        scenarios.append({"name": "moe_03_04_05_06_from_saved_orig", "steps": steps})

    # The two new patches touch disjoint paths and must commute on pinned parents.
    for order in ((SCALAR_PATCH, GDN_PATCH), (GDN_PATCH, SCALAR_PATCH)):
        with tempfile.TemporaryDirectory(prefix="qwen-patch-composition-") as directory:
            tree = Path(directory)
            write(tree, baseline_bytes)
            steps = [{"patch": patch, "checks": apply(tree, root / patch)} for patch in order]
            files = check(tree, expected_bytes)
            require({str(p.relative_to(tree)) for p in tree.rglob("*") if p.is_file()} == set(expected_bytes),
                    "Unexpected file produced in composed temporary tree")
            scenarios.append({"name": "then".join(Path(p).name for p in order),
                              "steps": steps, "files": files, "all_six_byte_identical": True})
    require(all(sha((root / path).read_bytes()) == expected for path, expected in protected.items()),
            "Protected baseline or overlay source changed during verification")
    return {
        "schema_version": 1, "status": "passed", "scope": "CPU patch reproduction, not runtime validation",
        "verifier_sha256": sha(Path(__file__).read_bytes()),
        "patch_tool": subprocess.check_output(["patch", "--version"], text=True).splitlines()[0],
        "patches": {p: sha((root / p).read_bytes()) for p in [*SERIES, GDN_PATCH]},
        "baselines": {p: {"source": s, "sha256": sha(baseline_bytes[p])} for p, s in baseline_files.items()},
        "outputs": {p: {"source": s, "sha256": sha(expected_bytes[p])} for p, s in source_files.items()},
        "measured_source_manifests": {name: {k: v for k, v in data.items() if k != "files"}
                                      for name, data in manifests.items()},
        "composed_run_manifest_scope": "startup source identity only; no run completion or quality claim",
        "required_configuration_helper": {"path": "kernels/gdn-norm-fp8/integration_config.py",
            "sha256": sha((root / "kernels/gdn-norm-fp8/integration_config.py").read_bytes())},
        "scenarios": scenarios, "protected_sources_unchanged": True,
        "gpu_used": False, "container_inference_used": False,
        "limitations": ["Artifacts are experimental, not selected or recommended.",
                        "Frozen-source identity does not establish combined serving quality or performance.",
                        "Runtime flags, splitting configuration, pinned dependencies, and all prerequisites remain required."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root.resolve())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "scenarios": len(result["scenarios"]),
                      "output_files": len(result["outputs"]), "protected_sources_unchanged": True}))
