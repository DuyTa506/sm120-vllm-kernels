#!/usr/bin/env python3
"""Separate opt-in caller-buffer native FP8 experiment; no serving edits.

--cpu-preflight extracts the exact installed native method with AST and runs
CPU FX capture only. --run-gpu checks exactness, unchanged real GEMM and replay,
then optionally times the original separate-copy chain and compiled-out chain.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import statistics
import sys

from fallback_quant_out import make_raw_quant_out


def emit(**item):
    print(json.dumps(item, sort_keys=True), flush=True)


def cpu_preflight(source):
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx
    # No vLLM import/platform probing. Execute only this inspected method.
    text = source.read_text()
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "QuantFP8")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_native")
    class GroupShape:
        PER_TOKEN = "per_token"
    class Quant:
        is_group_quant = False
        static = False
        group_shape = GroupShape.PER_TOKEN
        num_token_padding = None
    def broadcast(scale, shape):
        return scale
    ns = {"torch": torch, "GroupShape": GroupShape, "_FP8_MAX": 448.0, "_FP8_MIN": -448.0,
          "_FP8_MIN_SCALING_FACTOR": 1.0 / (448.0 * 512), "_FP8_DTYPE": torch.float8_e4m3fn,
          "group_broadcast": broadcast}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
    Quant.forward_native = ns["forward_native"]
    qz = Quant()
    raw = make_raw_quant_out(qz)
    x = torch.randn((3, 4096), dtype=torch.bfloat16)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = torch.empty((3, 1))
    graph = make_fx(raw)(x, q, s)
    graph(x, q, s)
    refq, refs = qz.forward_native(x)
    assert torch.equal(q.view(torch.uint8), refq.view(torch.uint8))
    assert torch.equal(s.view(torch.int32), refs.view(torch.int32))
    copies = [n for n in graph.graph.nodes if n.target == torch.ops.aten.copy_.default]
    assert len(copies) == 2
    assert not torch.cuda.is_initialized()
    emit(status="CPU_FX_PASS", source=str(source), source_sha256=hashlib.sha256(text.encode()).hexdigest(),
         cuda_initialized=torch.cuda.is_initialized(), copy_nodes=len(copies), graph=str(graph.graph),
         scope="Traceability only; CUDA copy-kernel elimination, exactness and timing pending")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-gpu", action="store_true")
    p.add_argument("--cpu-preflight", action="store_true")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--tokens", nargs="+", type=int, default=[1, 2, 3, 31, 64, 192, 1025])
    p.add_argument("--snapshot", type=Path)
    p.add_argument("--source", type=Path, default=Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/input_quant_fp8.py"))
    args = p.parse_args()
    if args.cpu_preflight and args.run_gpu:
        p.error("Choose CPU preflight or GPU execution")
    if any(t < 1 for t in args.tokens):
        p.error("Token counts must be positive")
    if args.cpu_preflight:
        cpu_preflight(args.source)
        return
    if not args.run_gpu:
        emit(status="PENDING_GPU", cuda_imported="torch" in sys.modules)
        return
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    from vllm import _custom_ops as ops
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 required")
    torch.set_grad_enabled(False)
    torch.manual_seed(513)
    with set_current_vllm_config(VllmConfig()):
        quant = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN, compile_native=False)
    ref = torch.compile(quant.forward_native, dynamic=True, fullgraph=True)
    raw = make_raw_quant_out(quant)
    candidate = torch.compile(raw, dynamic=True, fullgraph=True)
    b = bs = None
    if args.snapshot:
        from safetensors import safe_open
        index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        def get(suffix):
            name = "model.language_model.layers.18.linear_attn.out_proj." + suffix
            with safe_open(str(args.snapshot / index[name]), framework="pt", device="cpu") as f:
                return f.get_tensor(name).cuda()
        b, bs = get("weight").t(), get("weight_scale").float().reshape(-1)
        assert b.shape == (4096, 2048) and bs.shape == (2048,)
    def mm(q, s):
        return ops.cutlass_scaled_mm(q, b, scale_a=s, scale_b=bs, out_dtype=torch.bfloat16)
    def check(x, q, s, **context):
        rq, rs = ref(x, None, None)
        qm = int((q.view(torch.uint8) != rq.view(torch.uint8)).sum())
        sm = int((s.view(torch.int32) != rs.view(torch.int32)).sum())
        gm = int((mm(q, s).view(torch.int16) != mm(rq, rs).view(torch.int16)).sum()) if b is not None else None
        emit(event="exact", **context, fp8_mismatches=qm, scale_bit_mismatches=sm, gemm_bf16_mismatches=gm)
        if qm or sm or gm:
            raise AssertionError("Compiled-out exactness failure")
    for tokens in args.tokens:
        x = torch.randn((tokens, 4096), device="cuda", dtype=torch.bfloat16)
        q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        s = torch.empty((tokens, 1), device="cuda", dtype=torch.float32)
        for kind in ("random", "zero", "tiny", "padded", "boundary"):
            x.normal_()
            if kind == "zero": x.zero_()
            if kind == "tiny": x.mul_(2**-12)
            if kind == "padded": x[tokens // 2:].zero_()
            if kind == "boundary":
                vals = torch.tensor([0, 2**-10, 2**-9, 0.9921875, 1, 1.0625, 1.1875, 2.125, 7.75, 15.5], device="cuda")
                x.copy_(vals[torch.arange(4096, device="cuda") % len(vals)].expand(tokens, -1))
            candidate(x, q, s)
            check(x, q, s, tokens=tokens, kind=kind)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            candidate(x, q, s)
        for replay in range(3):
            x.normal_()
            x[tokens // 2:].zero_()
            graph.replay()
            check(x, q, s, tokens=tokens, replay=replay)
        if args.bench:
            inputs = [torch.randn_like(x) for _ in range(3)]
            outputs = [(torch.empty_like(q), torch.empty_like(s)) for _ in inputs]
            def separate(x, q, s):
                tq, ts = ref(x, None, None)
                q.copy_(tq); s.copy_(ts)
                return tq, ts
            def time_graph(fn):
                for inp, out in zip(inputs, outputs): fn(inp, *out)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                keepalive = []
                with torch.cuda.graph(g):
                    for inp, out in zip(inputs, outputs):
                        keepalive.append(fn(inp, *out))
                times = []
                for _ in range(30):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record(); g.replay(); end.record(); end.synchronize()
                    times.append(start.elapsed_time(end) * 1000 / len(inputs))
                return statistics.median(times)
            for order in (("separate_copy", separate), ("compiled_out", candidate),
                          ("compiled_out", candidate), ("separate_copy", separate)):
                emit(event="timing", tokens=tokens, chain=order[0], median_us=time_graph(order[1]))
    emit(status="PASS", scope="Isolated native quantizer/output buffers; not stateful GDN or serving validation")


if __name__ == "__main__":
    main()
