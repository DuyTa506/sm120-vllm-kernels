"""Capture real MoE-block inputs (post-attention-norm hidden states) from the full model.

Runs vLLM offline (eager, no compile, in-process engine) on one ~8k-token real-text prompt and saves the
hidden_states fed to the routed experts of selected layers. The router (gate) is recomputed offline from
those inputs, so the routing distribution in the benches is the real one.
usage: capture_moe_inputs.py [n_tokens]   -> data/moe_inputs.pt
"""
import os, sys, torch
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
from vllm import LLM, SamplingParams

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
LAYERS = [0, 9, 19, 29, 39]
HERE = os.path.dirname(os.path.abspath(__file__))
llm = LLM(model="unsloth/Qwen3.6-35B-A3B-NVFP4-Fast", enforce_eager=True, optimization_level=0,
          max_model_len=N + 64, max_num_batched_tokens=N + 64, gpu_memory_utilization=0.45,
          enable_prefix_caching=False, language_model_only=True, max_num_seqs=4)
tok = llm.get_tokenizer()
ids = tok(open(os.path.join(HERE, "data/prompt.txt")).read())["input_ids"][:N]

def install(model):
    store = {}
    lm = model.language_model if hasattr(model, "language_model") else model
    layers = lm.model.layers
    for i in LAYERS:
        blk = layers[i].mlp
        def hook(mod, args, kwargs, i=i):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            if hs is not None and hs.shape[0] >= 1024 and i not in store:
                store[i] = hs.detach().to("cpu").clone()
        blk.experts.register_forward_pre_hook(hook, with_kwargs=True)
    gates = {i: layers[i].mlp.gate.weight.detach().cpu().clone() for i in LAYERS}
    model._moe_capture = store
    return gates

gates = llm.apply_model(install)[0]
out = llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0))
store = llm.apply_model(lambda m: {k: v for k, v in m._moe_capture.items()})[0]
print({k: tuple(v.shape) for k, v in store.items()})
torch.save({"hidden": store, "gate": gates, "layers": LAYERS, "n_tokens": len(ids)},
           os.path.join(HERE, "data/moe_inputs.pt"))
print("saved")
