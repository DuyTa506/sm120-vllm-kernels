"""CPU-only frozen exact-shape policy. Unknown shapes retain scalar baseline."""
import hashlib
import json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose(calibration, minimum_gain=0.05):
    """Promote only if CUTLASS wins in BOTH paired timing orders."""
    return "cutlass" if calibration["valid"] and all(
        p["cutlass_us"] <= (1 - minimum_gain) * p["scalar_us"]
        for p in calibration["passes"]
    ) and len(calibration["passes"]) >= 2 else "scalar"


def select(policy, layer, tokens, source_sha256):
    if policy["source_sha256"] != source_sha256:
        raise ValueError("backend changed: recalibrate before integration")
    if tokens < 1024:
        return "existing"
    return policy["entries"].get(f"{layer}:{tokens}", "scalar")


def cache_identity(policy):
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
