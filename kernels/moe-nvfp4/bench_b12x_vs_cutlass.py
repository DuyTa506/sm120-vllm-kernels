"""Time FlashInfer CUTLASS fused MoE (vLLM default on SM120, autotuned like vLLM's warmup) vs FlashInfer b12x
fused MoE (with the corrected scales), same real layer, same real routing, same bf16 input.
Both sides include their BF16->NVFP4 input quantization (vLLM's scaled_fp4_quant for CUTLASS; b12x does it
in-kernel). Router/top-k excluded on both sides. CUDA graphs, median of 30 replays (5 calls per replay).
usage: bench_b12x_vs_cutlass.py [layer] [M ...]
"""
import sys, torch
from moe_common import *
from bench_tactics import graph_time


def main():
    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe import B12xMoEWrapper
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 19
    Ms = [int(a) for a in sys.argv[2:]] or [3, 12, 48, 192, 1024, 4096, 8192, 16384]
    raw = load_layer_raw(L)
    prep = prep_cutlass(raw)
    xall = load_hidden(L)
    if xall is None:
        xall = (torch.randn(8192, H, device="cuda") * 0.6).to(torch.bfloat16)
    s13m = convert_sf_to_mma_layout(prep["s13"].reshape(E * 2 * I, H // 16), m=2 * I, k=H, num_groups=E)
    s2m = convert_sf_to_mma_layout(prep["s2"].reshape(E * H, I // 16), m=H, k=I, num_groups=E)
    # static calibrated activation scales passed as b12x expects them (reciprocal / dequant style)
    a13v = (1.0 / prep["a1_gscale"]).contiguous(); a2v = (1.0 / prep["a2_gscale"]).contiguous()
    w1a = (1.0 / raw["wgs13"]).contiguous(); w2a = ((1.0 / raw["wgs2"]) * a2v).contiguous()
    wr = B12xMoEWrapper(num_experts=E, top_k=TOPK, hidden_size=H, intermediate_size=I,
                        use_cuda_graph=True, max_num_tokens=max(Ms), activation="silu")
    print(f"layer {L}: us per MoE call (quant + experts + reduce), CUDA graph")
    print(f"{'M':>6} {'rows':>7} {'cutlass':>9} {'b12x':>9} {'b12x/cut':>8}  {'rel_l2(b12x vs cutlass)':>24}")
    for M in Ms:
        if M <= xall.shape[0]:
            x = xall[:M].contiguous() if M >= 1024 else xall[torch.randperm(xall.shape[0], device="cuda")[:M]].contiguous()
        else:
            x = xall[torch.randint(0, xall.shape[0], (M,), device="cuda")].contiguous()
        wts, ids = route(x, raw["gate"])
        out = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
        with autotune(True):
            run_cutlass(prep, x, ids, wts, out=out)
        torch.cuda.synchronize()
        fc = lambda: run_cutlass(prep, x, ids, wts, out=out)
        fb = lambda: wr.run(x=x, w1_weight=prep["w13"], w1_weight_sf=s13m, w1_alpha=w1a, fc2_input_scale=a2v,
                            w2_weight=prep["w2"], w2_weight_sf=s2m, w2_alpha=w2a,
                            token_selected_experts=ids, token_final_scales=wts, input_global_scale=a13v)
        yc = fc().clone(); yb = fb().clone()
        tc = graph_time(fc); tb = graph_time(fb)
        r = rel_err(yb, yc)["rel_l2"]
        print(f"{M:6d} {M*TOPK:7d} {tc:9.1f} {tb:9.1f} {tb/tc:8.2f}  {r:24.4f}", flush=True)


if __name__ == "__main__":
    main()
