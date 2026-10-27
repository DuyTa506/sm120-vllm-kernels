"""CPU-serializable activation artifacts; importing this module is GPU-safe."""
import hashlib
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_spec(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "storage_offset": tensor.storage_offset(), "dtype": str(tensor.dtype)}


def write_capture(directory, layer, module_name, tensors, *, specs=None, norm_attributes=None):
    import torch

    specs = specs or {key: tensor_spec(value) for key, value in tensors.items()}
    destination = Path(directory) / f"layer{layer}.pt"
    if destination.exists():
        raise FileExistsError(destination)
    payload = {key: value.detach().to("cpu").contiguous().clone() for key, value in tensors.items()}
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return {"layer": layer, "module_name": module_name, "file": destination.name,
            "sha256": file_sha256(destination), "specs": specs,
            "norm_attributes": norm_attributes}


def load_capture(directory, entry, *, expected_tokens):
    import torch

    filename = entry["file"]
    if filename != f"layer{entry['layer']}.pt":
        raise ValueError("Unexpected capture filename")
    path = Path(directory) / filename
    if file_sha256(path) != entry["sha256"]:
        raise ValueError(f"Capture checksum mismatch: {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    shapes = {"x": (expected_tokens, 32, 128), "z": (expected_tokens, 32, 128),
              "weight": (128,), "norm_output": (expected_tokens, 32, 128)}
    if set(data) != set(shapes) or set(entry["specs"]) != set(shapes):
        raise ValueError("Capture must contain x, z, weight and norm_output")
    for key, shape in shapes.items():
        value, spec = data[key], entry["specs"][key]
        if tuple(value.shape) != shape or tuple(spec["shape"]) != shape:
            raise ValueError(f"Capture {key} shape does not match {shape}")
        if value.dtype != torch.bfloat16 or spec["dtype"] != "torch.bfloat16":
            raise ValueError(f"Capture {key} must be BF16")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Capture {key} has non-finite values")
    return data


def restore_layout(saved, spec, *, device):
    """Copy values into fresh storage with the original view's stride/offset."""
    import torch

    shape, stride = tuple(spec["shape"]), tuple(spec["stride"])
    offset = spec["storage_offset"]
    if tuple(saved.shape) != shape or str(saved.dtype) != spec["dtype"]:
        raise ValueError("Saved tensor shape/dtype differs from layout metadata")
    if (len(shape) == 3 and (shape[1:] != (32, 128) or len(stride) != 3
                            or stride[1:] != (128, 1) or not 4096 <= stride[0] <= 65536)):
        raise ValueError("Unsupported or overlapping GDN input stride")
    if len(shape) == 1 and (shape != (128,) or stride != (1,)):
        raise ValueError("Unsupported norm-weight stride")
    if len(shape) not in (1, 3) or not isinstance(offset, int) or not 0 <= offset <= 65536:
        raise ValueError("Unsupported capture shape/storage offset")
    size = offset + 1 + sum((n - 1) * step for n, step in zip(shape, stride))
    backing = torch.empty(size, dtype=saved.dtype, device=device)
    view = backing.as_strided(shape, stride, offset)
    view.copy_(saved)
    return view
