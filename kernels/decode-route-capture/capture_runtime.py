"""In-process instrumentation of vLLM 0.32 V2; imports CUDA dependencies only on install.

No package files are edited. Source-derived hooks require a GPU validation run;
the analyzer's CPU tests do not establish runtime capture correctness.
"""
import functools
import gzip
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path


def _geometry(model, scope):
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    result = []
    for name, module in model.named_modules():
        if not isinstance(module, MoERunner):
            continue
        config, owner = module.moe_config, module.routed_experts
        weight_names = ("w13_weight", "w2_weight")
        if scope == "target":
            names = weight_names + ("w13_weight_scale", "w2_weight_scale")
            fmt = "nvfp4"
            if "nvfp4" not in type(owner.quant_method).__name__.lower():
                raise RuntimeError("target route probe requires the NVFP4 quant method")
        else:
            names, fmt = weight_names, "bf16"
            if any(str(getattr(owner, n).dtype) != "torch.bfloat16" for n in names):
                raise RuntimeError("draft route probe expects the checkpoint's BF16 experts")
        tensors = {n: {"shape": list(getattr(owner, n).shape),
                       "dtype": str(getattr(owner, n).dtype),
                       "bytes": getattr(owner, n).numel() * getattr(owner, n).element_size()}
                   for n in names}
        total = sum(t["bytes"] for t in tensors.values())
        if total % config.num_experts:
            raise RuntimeError("expert payload does not divide by the expert count")
        result.append({"layer_id": module.layer_id, "module_name": name,
                       "hidden": config.hidden_dim,
                       "intermediate": config.intermediate_size_per_partition,
                       "experts": config.num_experts, "topk": config.experts_per_token,
                       "format": fmt, "quant_method": type(owner.quant_method).__name__,
                       "stored_payload_bytes_per_expert": total // config.num_experts,
                       "tensors": tensors})
    result.sort(key=lambda g: g["layer_id"])
    if not result or len({g["layer_id"] for g in result}) != len(result):
        raise RuntimeError(f"missing or duplicate {scope} routed layers")
    for g in result:
        if (g["hidden"], g["intermediate"], g["experts"], g["topk"]) != (2048, 512, 256, 8):
            raise RuntimeError(f"unexpected Qwen expert geometry: {g}")
    return result


def _check_config(config):
    parallel = config.parallel_config
    for attr in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                 "prefill_context_parallel_size", "decode_context_parallel_size"):
        if getattr(parallel, attr) != 1:
            raise RuntimeError(f"probe only supports {attr}=1")
    if parallel.enable_eplb or parallel.use_ubatching:
        raise RuntimeError("probe excludes EPLB and microbatch overlap")
    spec = config.speculative_config
    if spec is None or spec.method != "mtp" or spec.num_speculative_tokens != 2:
        raise RuntimeError("probe requires fixed MTP-2")
    if spec.enable_adaptive_verification or spec.uses_dynamic_speculative_decoding():
        raise RuntimeError("probe excludes adaptive/dynamic verification")
    if not config.use_v2_model_runner:
        raise RuntimeError("probe requires the V2 GPU model runner")


class DraftSink:
    """Fixed graph-owned buffers. Only copies/scalar stores execute inside graphs."""
    def __init__(self, speculator, model, geometry):
        import torch
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
        from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
        self.geometry = geometry
        self.layer_ids = [g["layer_id"] for g in geometry]
        size, device = speculator.max_num_tokens, speculator.device
        self.ids = torch.full((size, len(geometry), 8), -2, dtype=torch.int32, device=device)
        self.router_is_padding = torch.zeros((size, len(geometry)), dtype=torch.bool, device=device)
        self.mask_present = torch.zeros(len(geometry), dtype=torch.int32, device=device)
        self.layer_rows = torch.zeros(len(geometry), dtype=torch.int32, device=device)
        self.input_ids = torch.zeros(size, dtype=torch.int32, device=device)
        self.positions = torch.zeros(size, dtype=torch.int64, device=device)
        self.input_rows = torch.zeros((), dtype=torch.int32, device=device)
        self.draft_step = torch.zeros((), dtype=torch.int64, device=device)
        self.tensors = [self.ids, self.router_is_padding, self.mask_present, self.layer_rows,
                        self.input_ids, self.positions, self.input_rows, self.draft_step]
        self.pointers = [t.data_ptr() for t in self.tensors]
        for module in model.modules():
            if isinstance(module, MoERunner):
                if not isinstance(module.router, BaseRouter) or module.router.capture_fn is not None:
                    raise RuntimeError("draft router is unsupported or already has a capture callback")
                slot = self.layer_ids.index(module.layer_id)
                module.router.set_capture_fn(functools.partial(self.capture_routes, slot))

    def capture_routes(self, slot, ids):
        from vllm.forward_context import get_forward_context
        n = ids.shape[0]
        self.ids[:n, slot].copy_(ids)
        self.layer_rows[slot].fill_(n)
        mask = get_forward_context().is_padding
        if mask is None:
            self.mask_present[slot].fill_(0)
            self.router_is_padding[:n, slot].fill_(False)
        else:
            self.mask_present[slot].fill_(1)
            self.router_is_padding[:n, slot].copy_(mask[:n])

    def capture_inputs(self, speculator, n):
        # _prefill and _generate_draft mutate these inputs AFTER the model runs.
        # Recording these operations in the graph preserves the actual routed inputs.
        self.input_ids[:n].copy_(speculator.input_buffers.input_ids[:n])
        self.positions[:n].copy_(speculator.input_buffers.positions[:n])
        self.input_rows.fill_(n)
        self.draft_step.copy_(speculator.current_draft_step)

    def snapshot(self, phase, target, speculator, max_rows):
        # This boundary is OUTSIDE capture/replay; synchronization is intentional.
        if self.pointers != [t.data_ptr() for t in self.tensors]:
            raise RuntimeError("draft graph buffer pointer changed")
        n = int(self.input_rows.item())
        if not 0 < n <= max_rows:
            raise RuntimeError(f"draft row count {n} exceeds diagnostic bound {max_rows}")
        if self.layer_rows.tolist() != [n] * len(self.layer_ids):
            raise RuntimeError("draft layer capture is stale or incomplete")
        expected_step = 0 if phase == "draft_prefill" else 1
        if int(self.draft_step.item()) != expected_step:
            raise RuntimeError("wrong/stale draft phase captured")
        c = len(target["request_ids"])
        first = phase == "draft_prefill"
        starts = list(target["query_start_loc"]) if first else list(range(c + 1))
        rejected = target["proposal_num_rejected"].clone() if first else [0] * c
        real = starts[-1]
        if n < real:
            raise RuntimeError("draft tensor is smaller than its logical input batch")
        return {"step": target["step"], "phase": phase, "ids_dtype": "int32",
                "layer_ids": self.layer_ids, "request_ids": target["request_ids"],
                "num_rows": n, "num_real_rows": real, "query_start_loc": starts,
                "num_rejected": rejected, "num_sampled": target["proposal_num_sampled"].clone(),
                "target_num_rejected": target["proposal_num_rejected"].clone(),
                "is_padding": [i >= real for i in range(n)],
                "is_padding_source": "logical graph tail, NOT the router's mask",
                "ids": self.ids[:n].clone(), "input_ids": self.input_ids[:n].clone(),
                "positions": self.positions[:n].clone(),
                "router_mask_present": self.mask_present.clone(),
                "router_is_padding": self.router_is_padding[:n].clone(),
                "captured_draft_step": self.draft_step.clone(),
                "last_token_indices": speculator.last_token_indices[:c].clone()}


class V2RouteCapture:
    def __init__(self, max_steps=64, max_rows=512, capture_draft=True):
        self.max_steps, self.max_rows, self.capture_draft = max_steps, max_rows, capture_draft
        self.records, self.targets = [], []
        self.manifest = {"schema_version": 1, "instrumentation_only": True,
                         "gpu_validation": "not_run", "geometry": {"target": [], "draft": []},
                         "draft_capture": "requested_unverified" if capture_draft else "disabled_target_only",
                         "max_steps": max_steps, "max_rows": max_rows}
        self.originals = []

    def _patch(self, cls, method, wrapped):
        original = getattr(cls, method)
        self.originals.append((cls, method, original))
        setattr(cls, method, functools.wraps(original)(wrapped))

    def install(self):
        import vllm
        from vllm.distributed.aux_output_connector.worker import AuxOutputWorkerConnector as Aux
        from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator as MTP
        from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import TargetDependentARSpeculator as AR
        self.manifest["versions"] = {name: importlib.metadata.version(name)
                                     for name in ("vllm", "torch", "flashinfer-python")}
        root = Path(vllm.__file__).parent
        files = ["v1/worker/gpu/model_runner.py", "v1/worker/gpu/input_batch.py",
                 "v1/engine/core_client.py", "v1/worker/gpu/spec_decode/mtp/speculator.py",
                 "v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py",
                 "model_executor/layers/fused_moe/router/base_router.py",
                 "model_executor/layers/fused_moe/router/fused_topk_router.py",
                 "model_executor/layers/fused_moe/routed_experts_capturer.py",
                 "distributed/aux_output_connector/worker.py"]
        self.manifest["source_sha256"] = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
        original_init, original_prepare, original_process = Aux.__init__, Aux.prepare_output, Aux.process_output
        owner = self

        def init(connector, *args, **kwargs):
            config, model = kwargs["vllm_config"], kwargs["model"]
            _check_config(config)
            original_init(connector, *args, **kwargs)
            connector._routing_research_buffer_ptr = connector._capturer.device_buffer.data_ptr()
            owner.manifest["geometry"]["target"] = _geometry(model, "target")
            if len(owner.manifest["geometry"]["target"]) != 40:
                raise RuntimeError("expected 40 target routed layers")
            owner.manifest["config_repr"] = str(config)
            owner.manifest["runner"] = "vllm.v1.worker.gpu.model_runner.GPUModelRunner"

        def prepare(connector, batch):
            batch._routing_research_target = None
            pending = original_prepare(connector, batch)
            if pending is None or batch.has_prefill or len(owner.targets) >= owner.max_steps:
                return pending
            n, c = batch.num_tokens_after_padding, batch.num_reqs
            if n > owner.max_rows:
                raise RuntimeError(f"target rows {n} exceed diagnostic bound {owner.max_rows}")
            # Read the original int32 buffer, NOT pending.routed_experts_gpu (uint8).
            raw = connector._capturer.device_buffer
            if str(raw.dtype) != "torch.int32":
                raise RuntimeError("native route buffer is not int32")
            if raw.data_ptr() != connector._routing_research_buffer_ptr:
                raise RuntimeError("target graph buffer pointer changed")
            proposed = batch.num_draft_tokens_per_req
            record = {"step": len(owner.targets), "phase": "target_verify", "ids_dtype": "int32",
                      "layer_ids": list(range(raw.shape[1])), "request_ids": list(batch.req_ids),
                      "num_rows": n, "num_real_rows": batch.num_tokens,
                      "query_start_loc": batch.query_start_loc_np[:c + 1].tolist(),
                      "query_start_loc_gpu": batch.query_start_loc[:c + 1].clone(),
                      "num_draft_tokens_per_req": proposed[:c].tolist() if proposed is not None else [0] * c,
                      "num_computed_tokens": batch.num_computed_tokens_np[:c].tolist(),
                      "decode_graph_eligible": bool(batch.decode_graph_eligible),
                      "ids": raw[:n].clone(), "is_padding": batch.is_padding[:n].clone(),
                      "input_ids": batch.input_ids[:n].clone(), "positions": batch.positions[:n].clone()}
            owner.targets.append(record)
            owner.records.append(record)
            pending._routing_research_target = record
            batch._routing_research_target = record
            return pending

        def process(connector, pending):
            record = getattr(pending, "_routing_research_target", None)
            if record is not None:
                record["num_sampled"] = pending.num_sampled.tolist()
                record["num_rejected"] = pending.num_rejected.tolist()
            return original_process(connector, pending)

        self._patch(Aux, "__init__", init)
        self._patch(Aux, "prepare_output", prepare)
        self._patch(Aux, "process_output", process)
        if not self.capture_draft:
            return
        load, run, propose = MTP.load_draft_model, AR._run_model, AR.propose
        signature = inspect.signature(propose)

        def load_draft(speculator, *args, **kwargs):
            model = load(speculator, *args, **kwargs)
            _check_config(speculator.vllm_config)
            geometry = _geometry(model, "draft")
            if len(geometry) != 1:
                raise RuntimeError("this probe is scoped to the single Qwen MTP layer")
            owner.manifest["geometry"]["draft"] = geometry
            speculator._routing_research_sink = DraftSink(speculator, model, geometry)
            return model

        def run_model(speculator, num_tokens, *args, **kwargs):
            if speculator.pcp_manager is not None:
                raise RuntimeError("PCP input remapping is outside this probe")
            sink = getattr(speculator, "_routing_research_sink", None)
            if sink is not None:
                sink.capture_inputs(speculator, num_tokens)
            return run(speculator, num_tokens, *args, **kwargs)

        def proposal(speculator, *args, **kwargs):
            bound = signature.bind(speculator, *args, **kwargs)
            bound.apply_defaults()
            batch = bound.arguments["input_batch"]
            target = getattr(batch, "_routing_research_target", None)
            if bound.arguments["dummy_run"] or bound.arguments["is_profile"]:
                target = None
            if target is not None:
                target["proposal_num_rejected"] = bound.arguments["num_rejected"].clone()
                target["proposal_num_sampled"] = bound.arguments["num_sampled"].clone()
            speculator._routing_research_active = target
            try:
                return propose(speculator, *args, **kwargs)
            finally:
                speculator._routing_research_active = None

        def phase_hook(original, phase):
            def hook(speculator, num_reqs):
                result = original(speculator, num_reqs)
                target = getattr(speculator, "_routing_research_active", None)
                if target is not None:
                    record = speculator._routing_research_sink.snapshot(
                        phase, target, speculator, owner.max_rows)
                    owner.records.append(record)
                return result
            return hook

        self._patch(MTP, "load_draft_model", load_draft)
        self._patch(AR, "_run_model", run_model)
        self._patch(AR, "propose", proposal)
        self._patch(MTP, "on_prefill_end", phase_hook(MTP.on_prefill_end, "draft_prefill"))
        self._patch(MTP, "on_multi_step_decode_end", phase_hook(MTP.on_multi_step_decode_end, "draft_decode"))

    def export(self, out, *, success, error=None, validate_structure=None):
        import torch
        from analyze import analyze_capture
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            if isinstance(value, dict):
                return {k: cpu(v) for k, v in value.items()}
            if isinstance(value, list):
                return [cpu(v) for v in value]
            return value
        def json_value(value):
            if isinstance(value, torch.Tensor):
                return value.tolist()
            if isinstance(value, dict):
                return {k: json_value(v) for k, v in value.items()}
            if isinstance(value, list):
                return [json_value(v) for v in value]
            return value
        records = cpu(self.records)
        # The authoritative raw artifact preserves torch.int32 and the -1 sentinel.
        torch.save(records, out / "raw_routes.pt")
        rows = json_value(records)
        with gzip.open(out / "records.jsonl.gz", "wt") as f:
            for record in rows:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.manifest["gpu_validation"] = "capture_finished_pending_analysis" if success else "failed"
        self.manifest["error"] = error
        self.manifest["structural_validation"] = "not_run"
        if self.capture_draft:
            expected = {(r["step"], phase) for r in self.targets
                        for phase in ("draft_prefill", "draft_decode")}
            actual = {(r["step"], r["phase"]) for r in rows if r["phase"].startswith("draft_")}
            self.manifest["draft_capture"] = "captured_both_phases" if expected and expected == actual else "incomplete"
        analysis_error = None
        should_validate = success if validate_structure is None else validate_structure
        try:
            if should_validate:
                if not self.targets:
                    raise RuntimeError("no pure decode steps captured")
                for record in rows:
                    if record["phase"] == "target_verify":
                        if record["query_start_loc"] != record["query_start_loc_gpu"]:
                            raise RuntimeError("CPU/GPU target query boundaries differ")
                        for name in ("num_sampled", "num_rejected"):
                            if "proposal_" + name in record and record[name] != record["proposal_" + name]:
                                raise RuntimeError("proposal/target acceptance metadata do not match")
                if self.capture_draft and self.manifest["draft_capture"] != "captured_both_phases":
                    raise RuntimeError("draft data incomplete; use explicit --target-only for target capture")
                analysis = analyze_capture(self.manifest, rows)
                analysis["overall_run_success"] = success
                analysis["control_token_ids_match"] = self.manifest.get("control_token_ids_match")
                name = "analysis.json" if success else "analysis_diagnostic.json"
                (out / name).write_text(json.dumps(analysis, indent=2) + "\n")
                self.manifest["structural_validation"] = "passed"
                if success:
                    self.manifest["gpu_validation"] = "captured_and_structurally_validated"
        except Exception as ex:
            self.manifest["structural_validation"] = "failed"
            self.manifest["structural_error"] = str(ex)
            if success:
                self.manifest["gpu_validation"] = "analysis_failed"
                self.manifest["error"] = str(ex)
            analysis_error = ex
        finally:
            (out / "manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n")
        if analysis_error is not None:
            raise analysis_error
