"""Opt-in GPU checks for an UNVALIDATED GDN norm/gate/FP8 prototype.

Nothing imports torch/vLLM or executes GPU work without --run-gpu.
The references are installed vLLM implementations, not a copied formula.
"""

import argparse
import json
from pathlib import Path
import statistics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-gpu", action="store_true", help="explicitly run GPU work")
    p.add_argument("--mode", choices=("verify", "bench", "both"), default="verify")
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 3, 31, 128, 1023, 1024, 1025, 4096, 8192, 16384])
    p.add_argument("--bench-tokens", type=int, nargs="+", default=[4096, 8192, 16384])
    p.add_argument("--warps", type=int, nargs="+", choices=(4, 8, 16), default=[8])
    p.add_argument("--snapshot", type=Path, help="local model snapshot, for real norm/GEMM weights")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 18, 38])
    p.add_argument("--with-gemm", action="store_true", help="also check/time real output GEMM; requires --snapshot")
    p.add_argument("--replays", type=int, default=30)
    p.add_argument("--buffers", type=int, default=3, help="distinct inputs per timing graph")
    args = p.parse_args()
    if args.with_gemm and args.snapshot is None:
        p.error("--with-gemm requires --snapshot")
    if min(args.tokens + args.bench_tokens + [args.replays, args.buffers]) < 1:
        p.error("token counts, replays and buffers must be positive")
    return args


def emit(**data):
    print(json.dumps(data, sort_keys=True), flush=True)


def main():
    args = parse_args()
    if not args.run_gpu:
        emit(status="PENDING_GPU", message="No GPU work executed. Pass --run-gpu only when GPU testing is permitted.")
        return

    import torch
    from safetensors import safe_open
    from vllm import _custom_ops as ops
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn
    from fused import fused_norm_gate_fp8

    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This experiment is scoped to SM120")
    torch.set_grad_enabled(False)
    torch.manual_seed(1729)
    with set_current_vllm_config(VllmConfig()):
        quantizer = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN, compile_native=False)
    compiled_quant = torch.compile(quantizer.forward_native, fullgraph=True, dynamic=True)

    def norm(x, z, w):
        return rmsnorm_fn(x, w, None, z=z, eps=1e-6, group_size=None, norm_before_gate=True, activation="silu")

    def baseline(x, z, w, *, eager=False):
        y = norm(x, z, w).reshape(-1, 4096)
        if eager:
            return ops.scaled_fp8_quant(y, use_per_token_if_dynamic=True)
        return compiled_quant(y)

    weight_map = None
    if args.snapshot:
        weight_map = json.loads((args.snapshot / "model.safetensors.index.json").read_text())["weight_map"]

    def checkpoint_tensor(name):
        with safe_open(str(args.snapshot / weight_map[name]), framework="pt", device="cpu") as f:
            return f.get_tensor(name).cuda()

    def weights(layer):
        if layer is None:
            w = torch.linspace(0.125, 2.5, 128, dtype=torch.float32, device="cuda").to(torch.bfloat16)
            return w, None
        prefix = f"model.language_model.layers.{layer}.linear_attn"
        w = checkpoint_tensor(prefix + ".norm.weight").to(torch.bfloat16).contiguous()
        if w.shape != (128,):
            raise AssertionError(f"Unexpected norm shape: {w.shape}")
        gemm_weights = None
        if args.with_gemm:
            b = checkpoint_tensor(prefix + ".out_proj.weight").t()
            bs = checkpoint_tensor(prefix + ".out_proj.weight_scale").float().reshape(-1)
            if b.shape != (4096, 2048) or bs.shape != (2048,):
                raise AssertionError((b.shape, bs.shape))
            gemm_weights = b, bs
        return w, gemm_weights

    def mm(qs, bw):
        q, s = qs
        return ops.cutlass_scaled_mm(q, bw[0], scale_a=s, scale_b=bw[1], out_dtype=torch.bfloat16)

    def inputs(t, kind, strided):
        x = torch.randn((t, 32, 128), device="cuda", dtype=torch.bfloat16)
        backing = torch.empty((t, 12288), device="cuda", dtype=torch.bfloat16)
        z = backing[:, 8192:].view(t, 32, 128)
        z.normal_()
        if kind == "zeros":
            x.zero_()
        elif kind == "tiny":
            x.mul_(2**-12)
            z.mul_(2**-10)
        elif kind == "gate_extremes":
            z[..., ::2] = -80
            z[..., 1::2] = 80
        elif kind == "head_outliers":
            x.mul_(0.125)
            z.mul_(0.25)
            for h in (0, 15, 31):
                x[:, h, h] = 64
                z[:, h, h] = 16
        elif kind == "rounding_edges":
            # BF16 neighbours, FP8 midpoint magnitudes and tiny scale-floor cases.
            values = torch.tensor([0, 2**-10, 2**-9, 0.9921875, 1, 1.0078125,
                                   1.0625, 1.1875, 1.9375, 2.125, 7.75, 15.5], device="cuda")
            tiled = values[torch.arange(4096, device="cuda") % values.numel()]
            x.copy_(tiled.reshape(1, 32, 128))
            z.copy_(torch.linspace(-8, 8, 4096, device="cuda").reshape(1, 32, 128))
        if not strided:
            z = z.contiguous()
        return x, z

    def compare(actual, expected):
        q, s = actual
        eq, es = expected
        return {
            "fp8_mismatches": int((q.view(torch.uint8) != eq.view(torch.uint8)).sum()),
            "scale_bit_mismatches": int((s.view(torch.int32) != es.view(torch.int32)).sum()),
            "scale_max_abs": float((s - es).abs().max()),
        }

    def exact(stats):
        return stats["fp8_mismatches"] == 0 and stats["scale_bit_mismatches"] == 0

    def verify_case(t, kind, strided, w, bw, layer, warps):
        x, z = inputs(t, kind, strided)
        ref = baseline(x, z, w)
        eager = baseline(x, z, w, eager=True)
        got = fused_norm_gate_fp8(x, z, w, num_warps=warps)
        stats = compare(got, ref)
        eager_stats = compare(got, eager)
        record = dict(event="numerics", tokens=t, case=kind, strided_z=strided,
                      layer=layer, warps=warps, compiled_reference=stats, eager_reference=eager_stats)
        if bw is not None:
            a, b = mm(got, bw), mm(ref, bw)
            record["gemm_bf16_mismatches"] = int((a.view(torch.int16) != b.view(torch.int16)).sum())
        emit(**record)
        if not exact(stats):
            raise AssertionError("Compiled serving reference mismatch; do not benchmark or integrate")
        # A second reference is diagnostic: differences between CUDA eager and
        # compiled quantization are reported, never silently counted as equality.
        if bw is not None and record["gemm_bf16_mismatches"]:
            raise AssertionError("Unchanged output GEMM mismatch")
        if kind == "zeros":
            if not torch.equal(got[1], torch.full_like(got[1], 1 / (448 * 512))):
                raise AssertionError("Wrong zero-row scale floor")
        # Exercise tails, pointer reuse and input changes under graph replay.
        if kind == "random":
            q = torch.empty_like(got[0])
            s = torch.empty_like(got[1])
            fused_norm_gate_fp8(x, z, w, out=(q, s), num_warps=warps)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fused_norm_gate_fp8(x, z, w, out=(q, s), num_warps=warps)
            ptrs = q.data_ptr(), s.data_ptr()
            for iteration in range(3):
                x.add_(0.0078125)
                z.mul_(-0.5)
                graph.replay()
                replay_stats = compare((q, s), baseline(x, z, w))
                emit(event="graph_replay", tokens=t, iteration=iteration, strided_z=strided,
                     layer=layer, warps=warps, **replay_stats)
                if not exact(replay_stats) or ptrs != (q.data_ptr(), s.data_ptr()):
                    raise AssertionError("Graph replay mismatch")

    def graph_time(fns):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                for fn in fns:
                    fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        # Retain all results through replay for both implementations. Otherwise
        # the graph pool could reuse baseline output storage between buffers,
        # while the candidate's caller-owned outputs have distinct storage.
        captured_outputs = []
        with torch.cuda.graph(graph):
            for fn in fns:
                captured_outputs.append(fn())
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(args.replays):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000 / len(fns))
        return {"median_us": statistics.median(times), "min_us": min(times),
                "max_us": max(times), "replays": args.replays, "buffers": len(fns)}

    layers = args.layers if args.snapshot else [None]
    emit(event="environment", gpu=torch.cuda.get_device_name(), torch=torch.__version__,
         snapshot=str(args.snapshot) if args.snapshot else None,
         note="Synthetic activations; real norm/GEMM weights only with --snapshot")
    for layer in layers:
        w, bw = weights(layer)
        for warps in args.warps:
            # Bench-only still runs correctness for every shape/layout being timed.
            ts = args.tokens if args.mode != "bench" else args.bench_tokens
            kinds = ["random", "zeros", "tiny", "gate_extremes", "head_outliers", "rounding_edges"]
            if args.mode == "bench":
                kinds = ["random", "zeros", "rounding_edges"]
            for t in ts:
                for strided in (False, True):
                    for kind in kinds:
                        verify_case(t, kind, strided, w, bw, layer, warps)
            if args.mode == "verify":
                continue
            for t in args.bench_tokens:
                for strided in (False, True):
                    pairs = [inputs(t, "random", strided) for _ in range(args.buffers)]
                    for x, z in pairs:
                        stats = compare(fused_norm_gate_fp8(x, z, w, num_warps=warps), baseline(x, z, w))
                        if not exact(stats):
                            raise AssertionError("Timing-buffer correctness mismatch")
                    for with_mm in ([False, True] if bw is not None else [False]):
                        baseline_fns, fused_fns = [], []
                        for x, z in pairs:
                            q = torch.empty((t, 4096), dtype=torch.float8_e4m3fn, device="cuda")
                            s = torch.empty((t, 1), dtype=torch.float32, device="cuda")
                            def base(x=x, z=z):
                                qs = baseline(x, z, w)
                                return mm(qs, bw) if with_mm else qs
                            def candidate(x=x, z=z, q=q, s=s):
                                qs = fused_norm_gate_fp8(x, z, w, out=(q, s), num_warps=warps)
                                return mm(qs, bw) if with_mm else qs
                            baseline_fns.append(base)
                            fused_fns.append(candidate)
                        for order in ("AB", "BA"):
                            runs = [("baseline", baseline_fns), ("fused", fused_fns)]
                            if order == "BA":
                                runs.reverse()
                            for name, fns in runs:
                                emit(event="timing", layer=layer, tokens=t, warps=warps,
                                     layout="include_gate_copy" if strided else "contiguous_gate",
                                     with_gemm=with_mm, order=order, implementation=name, **graph_time(fns))
    emit(status="PASS", scope="requested synthetic-input numerical/graph cases only; serving validation still required")


if __name__ == "__main__":
    main()
