"""Insert a test-only GDN state dump hook into qwen_gdn_linear_attn.py (build/prod from the image, build/opt from the
repo's patched build) and write the unified diff of the hook to state-dump-hook.patch.

The hook is inert unless VLLM_CG_DUMP_DIR is set AND the file $VLLM_CG_DUMP_DIR/ENABLE exists. When on, every
prefill forward of the selected GDN layers (VLLM_CG_DUMP_LAYERS, default "0,2") saves the final recurrent state and
conv state of each prefill row to $VLLM_CG_DUMP_DIR/<counter>_<layer>.pt. Used by state_check.py to compare the
layer-0 GDN state after a hot (prefix-cache hit) prefill with the state after a cold prefill of the same prompt.
"""
import difflib, os

H = os.path.dirname(os.path.abspath(__file__))
HEAD = '''import os
# cache-granularity test hook: dump GDN prefill states (inert unless VLLM_CG_DUMP_DIR/ENABLE exists)
_CG_DUMP = os.environ.get("VLLM_CG_DUMP_DIR")
_CG_DUMP_LAYERS = set(os.environ.get("VLLM_CG_DUMP_LAYERS", "0,2").split(","))
_cg_counter = [0]


def _cg_dump(prefix, state, conv, md, init=None):
    import torch as _t

    _cg_counter[0] += 1
    qsl = md.prefill_query_start_loc
    _t.save(
        {"prefix": prefix, "state": state.float().cpu(), "conv": conv.float().cpu(),
         "init": None if init is None else init.float().cpu(),
         "qlens": (qsl[1:] - qsl[:-1]).tolist(), "has_init": md.prefill_has_initial_state.tolist()},
        f"{_CG_DUMP}/{_cg_counter[0]:07d}_{prefix.split('.layers.')[-1].split('.')[0]}.pt",
    )
'''
ANCHOR = "            ssm_state[prefill_state_indices] = last_recurrent_state.to(ssm_state.dtype)\n"
HOOK = ANCHOR + '''            if (
                _CG_DUMP
                and self.prefix.split(".layers.")[-1].split(".")[0] in _CG_DUMP_LAYERS
                and os.path.exists(_CG_DUMP + "/ENABLE")
            ):
                _cg_dump(
                    self.prefix, last_recurrent_state, conv_state[prefill_state_indices], attn_metadata, _cg_init
                )
'''
ANCHOR0 = "            initial_state[~prefill_has_initial_state, ...] = 0\n"
HOOK0 = ANCHOR0 + '''            # snapshot before the kernel (it may update the state in place)
            _cg_init = initial_state.clone() if _CG_DUMP else None
'''
for kind in ("prod", "opt"):
    src = open(f"{H}/build/{kind}/orig.py").read()
    assert src.count("import os\n") >= 1 and src.count(ANCHOR) == 1 and src.count(ANCHOR0) == 1
    out = src.replace("import os\n", HEAD, 1).replace(ANCHOR, HOOK).replace(ANCHOR0, HOOK0)
    open(f"{H}/build/{kind}/.tmp.py", "w").write(out)
    os.replace(f"{H}/build/{kind}/.tmp.py", f"{H}/build/{kind}/qwen_gdn_linear_attn.py")
    if kind == "prod":
        rel = "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
        diff = difflib.unified_diff(src.splitlines(True), out.splitlines(True), f"a/{rel}", f"b/{rel}")
        open(f"{H}/state-dump-hook.patch", "w").writelines(diff)
print("ok")
