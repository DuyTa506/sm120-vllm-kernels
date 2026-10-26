"""Where do the ~6 us of the fused kernel go?  Same harness as bench_graph.py (40 layers, distinct real weights,
L2 flushed between replays) with
  * launch-gap floors: a chain of empty kernels (1 CTA / 136 CTAs),
  * the fused kernel stopped early (ablate=1: after partial stores; 2: + ticket; 3: + partial reads; 0: full),
  * per-thread fences (relacq=0) vs one release/acquire pair (relacq=1),
  * a few (nt,nw,split) configs.
usage: python3 bench_ablate.py [reps]
"""
import sys, statistics
import torch
import torch.nn.functional as F
import vllm._custom_ops as ops
from common import real, normed_input, L
import fused_router as fr

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
dev = "cuda"
torch.manual_seed(0)
NL = 40
Wc = [torch.cat([real(L + f"{i}.mlp.gate.weight"), real(L + f"{i}.mlp.shared_expert_gate.weight")]).contiguous() for i in range(NL)]
Wg = [w[:256] for w in Wc]
lnw = [real(L + f"{i}.post_attention_layernorm.weight") for i in range(NL)]
flush = torch.empty(int(2 ** 29), dtype=torch.uint8, device=dev)


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


def noop_chain(ncta, nthr):
    def f():
        for _ in range(NL):
            fr.noop(ncta, nthr)
    return f


print("launch-gap floors (us per kernel in a captured chain of 40):")
print(f"  empty kernel 1 CTA x 32 thr     {graph_time(noop_chain(1, 32)):.2f}")
print(f"  empty kernel 136 CTA x 128 thr  {graph_time(noop_chain(136, 128)):.2f}")
print(f"  empty kernel 136 CTA x 1024 thr {graph_time(noop_chain(136, 1024)):.2f}")

CFGS = {1: [(1, 4, 8), (1, 8, 4), (1, 4, 4), (1, 8, 8), (1, 4, 16)], 3: [(1, 4, 8), (1, 8, 4), (1, 4, 4), (1, 8, 8), (1, 4, 16)],
        8: [(1, 8, 4), (1, 8, 8), (1, 4, 8)], 16: [(2, 16, 4), (2, 8, 8), (2, 16, 8)], 24: [(3, 32, 4), (3, 16, 8), (3, 24 if False else 16, 4)]}
for M in (1, 3, 8, 16, 24):
    xs = [normed_input(M, lnw[i]) for i in range(NL)]
    tw = [torch.empty(M, 8, dtype=torch.float32, device=dev) for _ in range(NL)]
    ti = [torch.empty(M, 8, dtype=torch.int32, device=dev) for _ in range(NL)]
    ws = fr.Workspace(dev)
    print(f"\nM={M}")
    for cfg in CFGS[M]:
        row = []
        for relacq in (0, 1):
            for ab in (1, 2, 3, 0):
                def f(cfg=cfg, relacq=relacq, ab=ab):
                    for i in range(NL):
                        fr.fused_router(xs[i], Wc[i], 8, num_experts=256, ws=ws, cfg=cfg, out=(tw[i], ti[i]), ablate=ab, relacq=relacq)
                row.append(graph_time(f, 25))
        torch.cuda.synchronize()
        assert bool((ws.counters == 0).all()), "counters not re-armed"
        print(f"  cfg(nt,nw,split)={cfg}: relacq=0 stop@partials/ticket/reads/full {row[0]:.2f} {row[1]:.2f} {row[2]:.2f} {row[3]:.2f} | relacq=1 {row[4]:.2f} {row[5]:.2f} {row[6]:.2f} {row[7]:.2f}", flush=True)
