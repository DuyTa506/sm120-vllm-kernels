"""Microbenchmark: unfused vs fused router in a CUDA graph of 40 layers with distinct real weights and L2 flushed
between replays (the 40 routers are 40 MB, which would otherwise sit in the 128 MB L2 across replays).

Per-layer time = graph replay time / 40, median of REPS replays. Variants (all produce topk_w fp32 [M,8], topk_i int32 [M,8]):
  cublas+topk   : vLLM production tier-6 path: F.linear (cuBLAS wmma kernel) -> ops.topk_softmax (renormalize)
  skinny+topk   : decode-roofline Triton skinny GEMM (the M<=3 rule of VLLM_SKINNY_BF16_GEMM, else cuBLAS) -> ops.topk_softmax
  skinnyNoRule  : same Triton GEMM at every M (to see where it stops winning)
  gemm only     : the router GEMM alone (lower bound for any fusion that keeps the GEMM as is)
  fused         : fused_router (default heuristic), and best (nt,nw,split) from a sweep
usage: python3 bench_graph.py [reps]
"""
import sys, itertools, statistics
import torch
import torch.nn.functional as F
import vllm._custom_ops as ops
sys.path.insert(0, "/dr")
from skinny_bf16_gemm import skinny_bf16_gemm
from common import real, normed_input, L
import fused_router as fr

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
dev = "cuda"
torch.manual_seed(0)
NL = 40
Wg = [real(L + f"{i}.mlp.gate.weight") for i in range(NL)]
Wsh = [real(L + f"{i}.mlp.shared_expert_gate.weight") for i in range(NL)]
Wc = [torch.cat([a, b]).contiguous() for a, b in zip(Wg, Wsh)]
lnw = [real(L + f"{i}.post_attention_layernorm.weight") for i in range(NL)]
flush = torch.empty(int(2 ** 29), dtype=torch.uint8, device=dev)   # 512 MB > 128 MB L2


def graph_time(fn, reps=None):
    reps = reps or REPS
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for r in range(reps + 3):
        flush.zero_()                       # evict weights/activations from L2 (outside the timed region)
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); b.synchronize()
        if r >= 3:
            ts.append(a.elapsed_time(b) * 1e3 / NL)
    return statistics.median(ts), g


def make(M):
    xs = [normed_input(M, lnw[i]) for i in range(NL)]
    tw = [torch.empty(M, 8, dtype=torch.float32, device=dev) for _ in range(NL)]
    ti = [torch.empty(M, 8, dtype=torch.int32, device=dev) for _ in range(NL)]
    te = [torch.empty(M, 8, dtype=torch.int32, device=dev) for _ in range(NL)]
    return xs, tw, ti, te


results = {}
print(f"{'M':>3s} | {'cublas+topk':>11s} {'skinny+topk':>11s} {'skinnyNoRule':>12s} {'gemm only':>9s} {'topk only':>9s} | {'fused dflt':>10s} {'fused best':>10s} (nt,nw,split) | saved vs skinny+topk  us/layer  us/step(41 layers)")
for M in (1, 2, 3, 4, 6, 8, 12, 16, 24):
    xs, tw, ti, te = make(M)
    lg = [F.linear(xs[i], Wg[i]) for i in range(NL)]

    def f_cublas():
        for i in range(NL):
            ops.topk_softmax(tw[i], ti[i], te[i], F.linear(xs[i], Wg[i]), True)

    def f_skinny_rule():
        for i in range(NL):
            y = skinny_bf16_gemm(xs[i], Wg[i]) if M <= 3 else F.linear(xs[i], Wg[i])
            ops.topk_softmax(tw[i], ti[i], te[i], y, True)

    def f_skinny_all():
        for i in range(NL):
            ops.topk_softmax(tw[i], ti[i], te[i], skinny_bf16_gemm(xs[i], Wg[i]), True)

    def f_gemm_only():
        for i in range(NL):
            skinny_bf16_gemm(xs[i], Wg[i])

    def f_topk_only():
        for i in range(NL):
            ops.topk_softmax(tw[i], ti[i], te[i], lg[i], True)

    ws = fr.Workspace(dev)

    def mk_fused(cfg):
        def f():
            for i in range(NL):
                fr.fused_router(xs[i], Wc[i], 8, num_experts=256, ws=ws, cfg=cfg, out=(tw[i], ti[i]))
        return f

    def mk_cluster(cfg):
        def f():
            for i in range(NL):
                fr.fused_router(xs[i], Wc[i], 8, num_experts=256, out=(tw[i], ti[i]), mode="cluster", cfg=cfg)
        return f

    t_c, _ = graph_time(f_cublas)
    t_s, _ = graph_time(f_skinny_rule)
    t_sa, _ = graph_time(f_skinny_all)
    t_g, _ = graph_time(f_gemm_only)
    t_t, _ = graph_time(f_topk_only)
    t_f, _ = graph_time(mk_fused(None))
    # sweep configs
    best = (1e9, None)
    nt0 = (M + 7) // 8
    for nw in {1: (4, 8), 2: (4, 8), 3: (4, 8), 4: (4, 8), 6: (8, 12), 8: (4, 8, 12), 12: (12, 16), 16: (12, 16), 24: (16, 24)}[M]:
        for split in (2, 4, 8):
            t, _ = graph_time(mk_fused((nt0, nw, split)), 10)
            if t < best[0]:
                best = (t, (nt0, nw, split))
    best = (graph_time(mk_fused(best[1]))[0], best[1])
    cbest = (1e9, None)
    try:
        for nw in (8, 16):
            for cs in (8, 16):
                t, _ = graph_time(mk_cluster((nt0, nw, cs)), 10)
                print(f"      cluster (nt,nw,cs)=({nt0},{nw},{cs}): {t:.2f}", flush=True)
                if t < cbest[0]:
                    cbest = (t, (nt0, nw, cs))
        cbest = (graph_time(mk_cluster(cbest[1]))[0], cbest[1])
    except Exception as ex:
        print("      cluster variant failed:", repr(ex)[:300], flush=True)
        cbest = (float("nan"), None)
    print(f"      cluster variant best (nt,nw,cs)={cbest[1]}: {cbest[0]:.2f} us/layer  (split-K best {best[0]:.2f}, skinny+topk {t_s:.2f}, cublas+topk {t_c:.2f})", flush=True)
    results[M] = dict(cluster=cbest[0], cluster_cfg=cbest[1], cublas=t_c, skinny=t_s, skinny_all=t_sa, gemm=t_g, topk=t_t, fused=t_f, fused_best=best[0], cfg=best[1])
    saved = t_s - best[0]
    print(f"{M:3d} | {t_c:11.2f} {t_sa if M>3 else t_s:11.2f} {t_sa:12.2f} {t_g:9.2f} {t_t:9.2f} | {t_f:10.2f} {best[0]:10.2f} {best[1]} | {saved:6.2f} {saved*41:8.0f}", flush=True)
    # correctness spot check of the best config
    out = fr.fused_router(xs[0], Wc[0], 8, num_experts=256, ws=ws, cfg=best[1])
    ref_w = torch.empty(M, 8, device=dev); ref_i = torch.empty(M, 8, dtype=torch.int32, device=dev); ref_e = torch.empty(M, 8, dtype=torch.int32, device=dev)
    ops.topk_softmax(ref_w, ref_i, ref_e, F.linear(xs[0], Wg[0]), True)
    assert (out[1].sort(-1).values == ref_i.sort(-1).values).float().mean() > 0.9, "mismatch"
import json
print(json.dumps(results))
