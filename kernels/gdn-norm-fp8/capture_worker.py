"""Opt-in vLLM v0.32 worker extension for one eager GDN prefill capture.

No imports of torch/vLLM and no hooks are installed at module import time.
Only the standalone capture driver's post-initialization RPC arms the hooks.
"""
import json
import os
from pathlib import Path
import re

from capture_io import tensor_spec, write_capture


class GDNCaptureWorkerExtension:
    def gdn_capture_install(self, directory, capture_id, layers, expected_tokens):
        if os.environ.get("GDN_ACTIVATION_CAPTURE") != "1":
            raise RuntimeError("Capture requires the standalone --run-gpu driver")
        if getattr(self, "_gdn_activation_capture", None) is not None:
            raise RuntimeError("Capture hooks are already installed")
        request = json.loads((Path(directory) / "capture_request.json").read_text())
        if (request["capture_id"] != capture_id or request["layers"] != layers
                or request["tokens"] != expected_tokens):
            raise ValueError("Capture RPC does not match the owned output directory")
        config = self.vllm_config
        if (not config.model_config.enforce_eager or int(config.compilation_config.mode) != 0
                or config.speculative_config is not None
                or config.parallel_config.tensor_parallel_size != 1
                or config.parallel_config.pipeline_parallel_size != 1
                or config.scheduler_config.enable_chunked_prefill
                or config.scheduler_config.max_num_seqs != 1
                or config.cache_config.enable_prefix_caching):
            raise ValueError("Capture requires eager, compilation-disabled, unchunked TP=PP=1, one sequence, no prefix cache or speculation")

        import torch

        model = self.model_runner.get_model()
        selected = {}
        for name, module in model.named_modules():
            match = re.search(r"(?:^|\.)layers\.(\d+)\.linear_attn\.norm$", name)
            if match and int(match.group(1)) in layers:
                layer = int(match.group(1))
                if layer in selected:
                    raise ValueError(f"Ambiguous norm for layer {layer}")
                attributes = {"eps": module.eps, "group_size": module.group_size,
                              "norm_before_gate": module.norm_before_gate,
                              "activation": module.activation, "bias": module.bias is not None,
                              "forward_method": module._forward_method.__name__}
                if (module.__class__.__name__ != "RMSNormGated" or module.eps != 1e-6
                        or module.group_size is not None or not module.norm_before_gate
                        or module.activation not in ("silu", "swish") or module.bias is not None
                        or attributes["forward_method"] != "forward_cuda"
                        or module.weight.dtype != torch.bfloat16
                        or tuple(module.weight.shape) != (128,)):
                    raise ValueError(f"Unexpected GDN norm contract: {name}: {attributes}")
                selected[layer] = (name, module, attributes)
        if set(selected) != set(layers):
            raise ValueError(f"Requested GDN layers {layers}; found {sorted(selected)}")
        state = {"capture_id": capture_id, "directory": str(directory), "layers": layers,
                 "tokens": expected_tokens, "handles": [], "captures": [], "calls": {},
                 "torch_version": str(torch.__version__), "pid": os.getpid(),
                 "resolved_options": {"enforce_eager": config.model_config.enforce_eager,
                                      "compilation_mode": int(config.compilation_config.mode),
                                      "custom_ops": list(config.compilation_config.custom_ops),
                                      "enable_chunked_prefill": config.scheduler_config.enable_chunked_prefill,
                                      "enable_prefix_caching": config.cache_config.enable_prefix_caching,
                                      "max_num_seqs": config.scheduler_config.max_num_seqs,
                                      "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens}}
        self._gdn_activation_capture = state

        def make_hooks(layer, name, attributes):
            pending = {}

            def before(module, args, kwargs):
                x = args[0] if args else kwargs["x"]
                z = args[1] if len(args) > 1 else kwargs.get("z")
                state["calls"][str(layer)] = state["calls"].get(str(layer), 0) + 1
                if state["calls"][str(layer)] != 1:
                    raise RuntimeError(f"Layer {layer} ran more than once; capture is not one prefill")
                if (tuple(x.shape) != (expected_tokens, 32, 128) or z is None
                        or z.shape != x.shape or x.dtype != torch.bfloat16 or z.dtype != torch.bfloat16
                        or x.stride()[1:] != (128, 1) or z.stride()[1:] != (128, 1)):
                    raise ValueError(f"Layer {layer}: unexpected norm inputs; expected {expected_tokens} tokens")
                values = {"x": x, "z": z, "weight": module.weight}
                pending["specs"] = {key: tensor_spec(value) for key, value in values.items()}
                # Blocking CPU copies intentionally perturb execution. No latency is measured.
                pending["tensors"] = {key: value.detach().to("cpu").contiguous().clone()
                                      for key, value in values.items()}

            def after(module, args, kwargs, output):
                if output.shape != (expected_tokens, 32, 128) or output.dtype != torch.bfloat16:
                    raise ValueError(f"Layer {layer}: unexpected BF16 norm output")
                pending["specs"]["norm_output"] = tensor_spec(output)
                pending["tensors"]["norm_output"] = output.detach().to("cpu").contiguous()
                state["captures"].append(write_capture(
                    Path(directory), layer, name, pending["tensors"], specs=pending["specs"],
                    norm_attributes=attributes))
                pending.clear()

            return before, after

        try:
            for layer, (name, module, attributes) in selected.items():
                before, after = make_hooks(layer, name, attributes)
                state["handles"].append(module.register_forward_pre_hook(before, with_kwargs=True))
                state["handles"].append(module.register_forward_hook(after, with_kwargs=True))
        except Exception:
            self.gdn_capture_remove()
            raise
        return {"capture_id": capture_id, "layers": sorted(selected), "pid": os.getpid(),
                "hooks_installed_after_model_initialization": True}

    def gdn_capture_finish(self):
        state = self._gdn_activation_capture
        self.gdn_capture_remove()
        report = {key: value for key, value in state.items() if key != "handles"}
        report["captures"].sort(key=lambda entry: entry["layer"])
        report["status"] = "complete" if (
            [entry["layer"] for entry in report["captures"]] == sorted(state["layers"])
            and all(state["calls"].get(str(layer)) == 1 for layer in state["layers"])) else "incomplete"
        (Path(state["directory"]) / "worker_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    def gdn_capture_remove(self):
        state = getattr(self, "_gdn_activation_capture", None)
        if state is not None:
            for handle in state["handles"]:
                handle.remove()
            state["handles"].clear()
        return {"hooks_removed": True}
