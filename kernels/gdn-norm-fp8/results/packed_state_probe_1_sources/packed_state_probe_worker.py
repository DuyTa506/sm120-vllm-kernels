"""Post-init instrumentation for one owned eager worker; no import-time GPU work."""
import copy
import importlib
import json
import os
from pathlib import Path
import re
import sys
from types import MethodType, ModuleType, SimpleNamespace

from packed_state_probe_utils import private_fixture, private_owner, byte_mismatches

MAX_CASES = 5


def metadata_summary(metadata):
    result = {name: int(getattr(metadata, name)) for name in (
        "num_actual_tokens", "num_prefills", "num_prefill_tokens", "num_decodes",
        "num_decode_tokens", "num_spec_decodes", "num_spec_decode_tokens")}
    result["accepted_tokens"] = (None if metadata.num_accepted_tokens is None
                                  else metadata.num_accepted_tokens.detach().cpu().tolist())
    checkpoint = metadata.checkpoint
    result["checkpoint_offsets"] = [] if checkpoint is None else list(checkpoint.offsets or [])
    return result


def case_kind(metadata):
    if metadata.num_prefills:
        if metadata.num_decodes or metadata.num_spec_decodes:
            return "mixed_spec" if metadata.num_spec_decodes else "mixed_non_spec"
        if metadata.num_actual_tokens >= 1024: return "long_prefill"
        return None
    if metadata.num_decodes or metadata.num_spec_decodes: return "pure_decode"
    return None


def load_isolated_overlay():
    name = "_gdn_private_state_probe_overlay"
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(Path(__file__).parent / "overlay/vllm/model_executor/layers/mamba/gdn")]
        sys.modules[name] = package
    return importlib.import_module(name + ".gdn_norm_fp8_fusion")


class GDNStateProbeWorkerExtension:
    def gdn_state_probe_install(self, directory, run_id):
        if os.environ.get("GDN_STATE_PROBE") != run_id:
            raise RuntimeError("Only the explicitly owned standalone driver may install this probe")
        request = json.loads((Path(directory) / "request.json").read_text())
        if request["run_id"] != run_id or request["layer"] != 18:
            raise ValueError("Owned request identity differs")
        c = self.vllm_config
        if (not c.model_config.enforce_eager or int(c.compilation_config.mode) != 0
                or c.parallel_config.tensor_parallel_size != 1 or c.parallel_config.pipeline_parallel_size != 1
                or not c.scheduler_config.enable_chunked_prefill or c.scheduler_config.max_num_seqs != 2
                or c.scheduler_config.max_num_batched_tokens != 2048
                or c.speculative_config is None or c.speculative_config.method != "mtp"
                or c.num_speculative_tokens != 2 or c.parallel_config.enable_dbo):
            raise ValueError("Requires owned eager TP/PP1, MTP2, two sequences, 2048-token chunks, no DBO")
        import torch
        from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
        from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
        from vllm.model_executor.layers.mamba.ops.fused_conv_post_conv import NULL_BLOCK_ID as CONV_NULL
        if NULL_BLOCK_ID != 0 or CONV_NULL != 0:
            raise RuntimeError("Probe remapping requires the installed GDN null-row-zero convention")
        from fallback_quant_out import make_raw_quant_out
        if torch.cuda.is_current_stream_capturing() or torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("Requires non-capturing SM120 execution")
        selected = [(name, module) for name, module in self.model_runner.get_model().named_modules()
                    if re.search(r"(?:^|\.)layers\.18\.linear_attn$", name)]
        if len(selected) != 1: raise ValueError(f"Expected one layer18 owner, got {len(selected)}")
        name, owner = selected[0]
        if (not owner.enable_fused_gdn_spec_decode or owner.gdn_prefill_backend != "flashinfer"
                or owner.gqa_interleaved_layout or owner.norm.activation != "silu"
                or len(owner.kv_cache) != 2):
            raise ValueError("Original packed CUDA/FlashInfer GDN path is required")
        quant = owner.out_proj.scheme.fp8_linear.quant_fp8
        if (quant.static or quant.group_shape != GroupShape.PER_TOKEN or quant.is_group_quant
                or quant.num_token_padding is not None or quant.column_major_scales or quant.tma_aligned_scales):
            raise ValueError("Unexpected original native QuantFP8 contract")
        # Explicit raw native oracle: eager model dispatch otherwise selects CUDA quantization.
        reference = torch.compile(quant.forward_native, dynamic=True, fullgraph=True)
        quant_out = torch.compile(make_raw_quant_out(quant), dynamic=True, fullgraph=True)
        overlay = load_isolated_overlay()
        state = dict(directory=directory, run_id=run_id, owner=owner, original=owner._forward_core_fused_norm_packed,
                     cases=[], seen=set(), observed={}, calls=0, name=name, hook_present=False)
        self._gdn_state_probe = state
        sm_count = int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)

        def probe(this, mixed_qkvz, ba, core_attn_out):
            from vllm.forward_context import get_forward_context, override_forward_context
            from vllm.config import CUDAGraphMode
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("State probe must never double-execute under CUDA capture")
            context = get_forward_context()
            if context.cudagraph_runtime_mode != CUDAGraphMode.NONE:
                raise RuntimeError("State probe requires eager forward context")
            metadata = context.attn_metadata.get(this.prefix) if isinstance(context.attn_metadata, dict) else None
            if metadata is None: return state["original"](mixed_qkvz, ba, core_attn_out)
            if (mixed_qkvz.dtype != torch.bfloat16 or mixed_qkvz.ndim != 2
                    or mixed_qkvz.shape[1] != 12288 or mixed_qkvz.stride() != (12288, 1)
                    or ba.dtype != torch.bfloat16 or ba.shape != (mixed_qkvz.shape[0], 64)
                    or not 0 <= metadata.num_actual_tokens <= mixed_qkvz.shape[0]):
                raise RuntimeError("Actual packed input/metadata contract differs from the candidate")
            state["calls"] += 1
            kind = case_kind(metadata)
            info = metadata_summary(metadata)
            if kind: state["observed"][kind] = state["observed"].get(kind, 0) + 1
            key = kind if kind not in state["seen"] else None
            if key is None and any(info["checkpoint_offsets"]) and "checkpoint" not in state["seen"]:
                key = "checkpoint"
            if key and len(state["cases"]) < MAX_CASES:
                # Full selected-layer cache snapshots detect any live cache mutation,
                # including a write outside the compact referenced row set.
                live_before = tuple(t.detach().cpu().clone() for t in this.kv_cache)
                record = {"case": key, "kind": kind, "metadata": info, "variants": [],
                          "original_tokens": mixed_qkvz.shape[0], "qkvz_stride": list(mixed_qkvz.stride()),
                          "ba_stride": list(ba.stride()), "live_cache_strides": [list(t.stride()) for t in this.kv_cache]}
                previous = None
                try:
                    for added_padding in (0, 17):
                        tokens = mixed_qkvz.shape[0] + added_padding
                        inputs = torch.full((tokens, 12288), float("nan"), device=mixed_qkvz.device, dtype=mixed_qkvz.dtype)
                        gates = torch.full((tokens, 64), float("nan"), device=ba.device, dtype=ba.dtype)
                        inputs[:mixed_qkvz.shape[0]].copy_(mixed_qkvz)
                        gates[:ba.shape[0]].copy_(ba)
                        pairs = []
                        for candidate in (False, True):
                            cache, meta, rows = private_fixture(this.kv_cache, metadata)
                            proxy = private_owner(this, cache)
                            object.__setattr__(proxy, "_gdn_norm_fp8_enabled", True)
                            object.__setattr__(proxy, "_gdn_norm_fp8_sm_count", sm_count)
                            object.__setattr__(proxy, "out_proj", SimpleNamespace(scheme=SimpleNamespace(
                                fp8_linear=SimpleNamespace(_ready=True, _quant_out=quant_out, _warmup_sizes=()))))
                            cloned_context = copy.copy(context)
                            cloned_context.attn_metadata = dict(context.attn_metadata)
                            cloned_context.attn_metadata[this.prefix] = meta
                            core = torch.zeros((tokens, 32, 128), device=inputs.device, dtype=torch.bfloat16)
                            # Independent inputs guard against hidden input mutation by either route.
                            x, a = inputs.clone(), gates.clone()
                            with override_forward_context(cloned_context):
                                if candidate:
                                    q = torch.empty((tokens, 4096), device=x.device, dtype=torch.float8_e4m3fn)
                                    scale = torch.empty((tokens, 1), device=x.device, dtype=torch.float32)
                                    overlay.packed_norm_fp8(proxy, x, a, core, q, scale, meta)
                                else:
                                    proxy._forward_core_fused_norm_packed(x, a, core)
                                    q, scale = reference(core.flatten(-2), None, None)
                            if get_forward_context() is not context:
                                raise AssertionError("Original forward context was not restored")
                            pairs.append((q, scale, cache))
                        ref, got = pairs
                        mismatch = dict(fp8_bytes=byte_mismatches(ref[0], got[0]),
                                        scale_bytes=byte_mismatches(ref[1], got[1]),
                                        cache_bytes=[byte_mismatches(a, b) for a, b in zip(ref[2], got[2])])
                        actual = metadata.num_actual_tokens
                        tail_q = got[0][actual:]
                        tail_scale = got[1][actual:]
                        tail_ok = bool((tail_q.view(torch.uint8) == 0).all()) and bool(
                            (tail_scale == overlay.SCALE_FLOOR).all())
                        padding_invariant = True
                        if previous is not None:
                            padding_invariant = (byte_mismatches(previous[0], got[0][:previous[0].shape[0]]) == 0
                                and byte_mismatches(previous[1], got[1][:previous[1].shape[0]]) == 0
                                and all(byte_mismatches(a, b) == 0 for a, b in zip(previous[2], got[2])))
                        previous = got
                        record["variants"].append(dict(added_padding=added_padding, tokens=tokens,
                            compact_original_rows=rows, cache_shapes=[list(t.shape) for t in got[2]],
                            route=overlay.packed_route(tokens, actual, metadata.num_prefills),
                            mismatch=mismatch, tail_zero_floor=tail_ok, padding_invariant=padding_invariant))
                        if (mismatch["fp8_bytes"] or mismatch["scale_bytes"] or any(mismatch["cache_bytes"])
                                or not tail_ok or not padding_invariant):
                            raise AssertionError("Packed state/Q/S exact comparison failed")
                finally:
                    live_mismatch = [byte_mismatches(before, now.detach().cpu())
                                     for before, now in zip(live_before, this.kv_cache)]
                    record["live_cache_byte_mismatches"] = live_mismatch
                    with (Path(directory) / "cases.jsonl").open("a") as stream:
                        stream.write(json.dumps(record, sort_keys=True) + "\n")
                    if any(live_mismatch): raise AssertionError("Private probe mutated the live cache")
                state["cases"].append(record)
                state["seen"].add(key)
                if any(info["checkpoint_offsets"]): state["seen"].add("checkpoint")
            # Exactly one normal stateful call advances the real model trajectory.
            return state["original"](mixed_qkvz, ba, core_attn_out)

        owner._forward_core_fused_norm_packed = MethodType(probe, owner)
        state["hook_present"] = True
        return {"layer": name, "pid": os.getpid(), "torch_version": str(torch.__version__),
                "scope": "owned eager layer18 private state parity, raw native QuantFP8 explicitly compiled",
                "max_cases": MAX_CASES, "live_cache_shapes": [list(t.shape) for t in owner.kv_cache]}

    def gdn_state_probe_finish(self):
        state = self._gdn_state_probe
        self.gdn_state_probe_remove()
        seen = state["seen"]
        complete = "long_prefill" in seen and "pure_decode" in seen and bool({"mixed_spec", "mixed_non_spec"} & seen)
        report = dict(status="complete" if complete else "incomplete", cases=state["cases"],
                      observed=state["observed"], calls=state["calls"], coverage=sorted(seen),
                      checkpoint_observed="checkpoint" in seen,
                      scope="Eager/private compact state only; not compiled graph replay or full-model equivalence")
        (Path(state["directory"]) / "worker_report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    def gdn_state_probe_remove(self):
        state = getattr(self, "_gdn_state_probe", None)
        if state is not None and state["hook_present"]:
            state["owner"]._forward_core_fused_norm_packed = state["original"]
            state["hook_present"] = False
        return {"probe_removed": True}
