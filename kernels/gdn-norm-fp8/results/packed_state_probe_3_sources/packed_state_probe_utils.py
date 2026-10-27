"""Private fixture helpers. Torch is imported only by invoked tensor operations."""
import copy
from dataclasses import is_dataclass

INDEX_FIELDS = {"spec_state_indices_tensor", "non_spec_state_indices_tensor",
                "prefill_state_indices", "state_indices"}


def clone_tree(value, memo=None):
    """Clone every tensor, including dynamic checkpoint scratch; reject opaque objects."""
    import torch
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        result = value.detach().clone(memory_format=torch.preserve_format)
    elif value is None or type(value) in (int, float, bool, str):
        return value
    elif isinstance(value, dict):
        result = {}; memo[id(value)] = result
        result.update({key: clone_tree(item, memo) for key, item in value.items()})
    elif isinstance(value, (list, tuple)):
        result = [clone_tree(item, memo) for item in value]
        if isinstance(value, tuple): result = tuple(result)
    elif is_dataclass(value):
        result = copy.copy(value); memo[id(value)] = result
        for key, item in vars(value).items():
            object.__setattr__(result, key, clone_tree(item, memo))
    else:
        raise TypeError(f"Unsupported metadata/scratch type: {type(value).__name__}")
    memo[id(value)] = result
    return result


def index_fields(value):
    """Yield all real and cached-checkpoint index fields, including dynamic attrs."""
    if is_dataclass(value):
        for key, item in vars(value).items():
            if key in INDEX_FIELDS:
                if item is not None: yield value, key, item
            elif "state" in key and "indice" in key:
                raise ValueError(f"Unreviewed state index field: {key}")
            else:
                yield from index_fields(item)
    elif isinstance(value, dict):
        for item in value.values(): yield from index_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value: yield from index_fields(item)


def private_fixture(cache, metadata):
    """Compact referenced rows; preserve NULL_BLOCK_ID=0 and pad slot -1."""
    import torch
    if len(cache) != 2 or cache[0].shape[0] != cache[1].shape[0]:
        raise ValueError("Expected two caches with the same row count")
    copied = clone_tree(metadata)
    fields = list(index_fields(copied))
    rows = {0}
    for _, _, indices in fields:
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("State indices must be int32/int64")
        values = indices.detach().cpu().reshape(-1).tolist()
        if any(v < -1 or v >= cache[0].shape[0] for v in values):
            raise ValueError("Out-of-bounds state index")
        rows.update(v for v in values if v >= 0)
    rows = sorted(rows)
    mapping = {old: new for new, old in enumerate(rows)}
    remapped = {}
    for parent, key, indices in fields:
        if id(indices) not in remapped:
            result = indices.clone()
            for old, new in mapping.items(): result[indices == old] = new
            remapped[id(indices)] = result
        object.__setattr__(parent, key, remapped[id(indices)])
    private = []
    for state in cache:
        selection = torch.tensor(rows, dtype=torch.int64, device=state.device)
        target = torch.empty_strided((len(rows), *state.shape[1:]), state.stride(),
                                     dtype=state.dtype, device=state.device)
        target.copy_(state.index_select(0, selection))
        if target.untyped_storage().data_ptr() == state.untyped_storage().data_ptr():
            raise AssertionError("Private cache aliases the live cache")
        private.append(target)
    return tuple(private), copied, rows


def private_owner(owner, cache):
    """Share read-only weights/submodules; shadow only private state on the copy."""
    proxy = copy.copy(owner)
    # A post-init instance hook must not remain bound to the live owner.
    proxy.__dict__.pop("_forward_core_fused_norm_packed", None)
    object.__setattr__(proxy, "kv_cache", cache)
    for name in ("_forward_core", "_forward_core_fused_norm_packed", "_forward_core_fused_norm",
                 "_forward_core_decode_spec_fused_norm", "_forward_core_decode_spec_post_conv_fused_norm",
                 "_forward_core_decode_non_spec", "_rms_norm_gated_cuda"):
        if not hasattr(proxy, name): continue
        if getattr(getattr(proxy, name), "__self__", None) is not proxy:
            raise AssertionError(f"{name} did not bind the private owner")
    return proxy


def byte_mismatches(a, b):
    import torch
    if a.shape != b.shape or a.dtype != b.dtype: raise ValueError("Tensor contract differs")
    return int((a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum().item())
