"""Does prefetching the NEXT layer's router weight into L2 from the fused kernel help, and how does the fused router
compare with the unfused chain when ~30-40 us of other work (here: a 48 MB streaming read standing in for the expert
GEMM weight traffic) sits between two routers, like in a real MoE layer?

Per layer: [router variant] -> filler (reads 48 MB, ~30 us). Time per layer = replay time / 40 minus the filler-only time.
Variants: skinny+topk (decode-roofline Triton GEMM + vLLM topkGating), fused, fused + prefetch of the next layer's weight.
"""
import sys, statistics
import torch
import vllm._custom_ops as ops
sys.path.insert(0, "/dr")
from skinny_bf16_gemm import skinny_bf16_gemm
from common import real, normed_input, L
import fused_router as fr

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
dev = "cuda"
torch.manual_seed(0)
NL = 40
Wc = [torch.cat([real(L + f"{i}.mlp.gate.weight"), real(L + f"{i}.mlp.shared_expert_gate.weight")]).contiguous() for i in range(NL)]
Wg = [w[:256] for w in Wc]
lnw = [real(L + f"{i}.post_attention_layernorm.weight") for i in range(NL)]
flush = torch.empty(int(2 ** 29), dtype=torch.uint8, device=dev)
filler = torch.empty(48 * 2 ** 20, dtype=torch.uint8, device=dev)


def run_filler():
    filler.view(torch.int32).max()


def graph_time(fn, reps=REPS):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for r in range(reps + 3):
        flush.zero_()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); b.synchronize()
        if r >= 3:
            ts.append(a.elapsed_time(b) * 1e3 / NL)
    return statistics.median(ts)


def filler_only():
    for _ in range(NL):
        run_filler()


t_fill = graph_time(filler_only)
print(f"filler alone: {t_fill:.2f} us per layer")
print(f"{'M':>3s} | {'skinny+topk':>11s} {'fused':>7s} {'fused+pf':>9s}   (router cost per layer = chain time minus filler)   | saved vs skinny+topk: fused  fused+pf  (us/layer)")
for M in (1, 3, 4, 8, 16):
    xs = [normed_input(M, lnw[i]) for i in range(NL)]
    tw = [torch.empty(M, 8, dtype=torch.float32, device=dev) for _ in range(NL)]
    ti = [torch.empty(M, 8, dtype=torch.int32, device=dev) for _ in range(NL)]
    te = [torch.empty(M, 8, dtype=torch.int32, device=dev) for _ in range(NL)]
    ws = fr.Workspace(dev)

    def f_skinny():
        for i in range(NL):
            y = skinny_bf16_gemm(xs[i], Wg[i]) if M <= 3 else torch.nn.functional.linear(xs[i], Wg[i])
            ops.topk_softmax(tw[i], ti[i], te[i], y, True)
            run_filler()

    def mk(pf):
        def f():
            for i in range(NL):
                fr.fused_router(xs[i], Wc[i], 8, num_experts=256, ws=ws, out=(tw[i], ti[i]),
                                prefetch_next=(Wc[(i + 1) % NL] if pf else None))
                run_filler()
        return f

    def mkc(pf):
        def f():
            for i in range(NL):
                fr.fused_router(xs[i], Wc[i], 8, num_experts=256, out=(tw[i], ti[i]), mode="cluster",
                                prefetch_next=(Wc[(i + 1) % NL] if pf else None))
                run_filler()
        return f

    a = graph_time(f_skinny) - t_fill
    b = graph_time(mk(False)) - t_fill
    c = graph_time(mk(True)) - t_fill
    try:
        d = graph_time(mkc(False)) - t_fill
        e = graph_time(mkc(True)) - t_fill
    except Exception as ex:
        print("      cluster variant failed:", repr(ex)[:300]); d = e = float("nan")
    print(f"      cluster: {d:.2f}  cluster+pf: {e:.2f}  (saved vs skinny+topk {a-d:.2f} / {a-e:.2f})")
    print(f"{M:3d} | {a:11.2f} {b:7.2f} {c:9.2f}                                                       | {a-b:6.2f} {a-c:8.2f}", flush=True)
