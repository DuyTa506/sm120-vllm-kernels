"""Verify this recorded raw capture on CPU; never initializes CUDA."""
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

root = Path(__file__).resolve().parent
assert not torch.cuda.is_initialized()
raw = torch.load(root / "raw_routes.pt", map_location="cpu", weights_only=True)
with gzip.open(root / "records.jsonl.gz", "rt") as stream:
    rows = [json.loads(line) for line in stream]


def lists(value):
    if isinstance(value, torch.Tensor):
        assert value.device.type == "cpu"
        return value.tolist()
    if isinstance(value, dict):
        return {key: lists(item) for key, item in value.items()}
    if isinstance(value, list):
        return [lists(item) for item in value]
    return value


assert lists(raw) == rows, "authoritative tensor capture differs from JSON view"
assert len(raw) == 192
phases = collections.Counter()
shapes = collections.defaultdict(set)
for record in raw:
    assert record["ids"].dtype == torch.int32
    assert record["ids_dtype"] == "int32"
    phases[record["phase"]] += 1
    shapes[record["phase"]].add(tuple(record["ids"].shape))
    if record["phase"] == "target_verify":
        assert lists(record["query_start_loc_gpu"]) == record["query_start_loc"]
        for name in ("num_sampled", "num_rejected"):
            assert lists(record["proposal_" + name]) == lists(record[name])
assert phases == {"target_verify": 64, "draft_prefill": 64, "draft_decode": 64}
assert not torch.cuda.is_initialized()
result = {
    "status": "passed",
    "scope": "CPU raw-tensor/JSON integrity only; GPU/output equivalence remains failed",
    "cuda_initialized": False,
    "tensor_load": {"map_location": "cpu", "weights_only": True},
    "records": len(raw),
    "phase_counts": dict(phases),
    "phase_shapes": {key: sorted(value) for key, value in shapes.items()},
    "all_route_tensors_signed_int32": True,
    "all_tensor_and_metadata_values_match_json": True,
    "target_query_and_acceptance_metadata_match": True,
    "sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
               for name in ("raw_routes.pt", "records.jsonl.gz", "verify_raw_cpu.py")},
}
print(json.dumps(result, indent=2))
