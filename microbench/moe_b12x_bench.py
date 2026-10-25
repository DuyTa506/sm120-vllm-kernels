import sys, time, torch
from flashinfer.fused_moe import B12xMoEWrapper
from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

E, TOPK, K, N = 256, 8, 2048, 512
MAXTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
Ms = [int(x) for x in sys.argv[2:]] or [1024, 2048, 4608, 8192]
dev = "cuda"
import os
ROUTE = os.environ.get("ROUTE", "rand")
torch.manual_seed(0)

def rand_fp4(*shape):
    return torch.randint(0, 256, shape, dtype=torch.uint8, device=dev)

def rand_sf(*shape):
    # plausible ue4m3 block scales around 1.0 (0x38)
    b = torch.randint(0x30, 0x40, shape, dtype=torch.uint8, device=dev)
    return b.view(torch.float8_e4m3fn)

w1 = rand_fp4(E, 2 * N, K // 2)
w2 = rand_fp4(E, K, N // 2)
w1_sf = rand_sf(E, 2 * N, K // 16)
w2_sf = rand_sf(E, K, N // 16)
w1_sf_mma = convert_sf_to_mma_layout(w1_sf.reshape(E * 2 * N, K // 16), m=2 * N, k=K, num_groups=E)
w2_sf_mma = convert_sf_to_mma_layout(w2_sf.reshape(E * K, N // 16), m=K, k=N, num_groups=E)
alpha1 = torch.full((E,), 0.01, device=dev)
alpha2 = torch.full((E,), 0.01, device=dev)
fc2_scale = torch.ones(E, device=dev)

wr = B12xMoEWrapper(num_experts=E, top_k=TOPK, hidden_size=K, intermediate_size=N,
                    use_cuda_graph=True, max_num_tokens=MAXTOK, activation="silu")
for M in Ms:
    x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
    # balanced-ish random routing with distinct experts per token
    ids = torch.rand(M, E, device=dev).topk(TOPK, dim=1).indices.to(torch.int32)
    if ROUTE == 'same': ids = torch.arange(TOPK, dtype=torch.int32, device=dev).repeat(M, 1)
    wts = torch.softmax(torch.randn(M, TOPK, device=dev), dim=1)
    def run():
        return wr.run(x=x, w1_weight=w1, w1_weight_sf=w1_sf_mma, w1_alpha=alpha1,
                      fc2_input_scale=fc2_scale, w2_weight=w2, w2_weight_sf=w2_sf_mma,
                      w2_alpha=alpha2, token_selected_experts=ids, token_final_scales=wts)
    out = run(); torch.cuda.synchronize()
    for _ in range(3): run()
    torch.cuda.synchronize()
    n = 20
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): run()
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / n
    fl = 2 * M * TOPK * (2 * N * K + N * K) / 1e12
    print(f"b12x M={M:6d}: {ms*1e3:8.1f} us/layer  ({fl/ (ms/1e3):6.0f} TFLOPS eff)  out finite={torch.isfinite(out).all().item()}", flush=True)

# ---- CUDA graph capture/replay test ----
print("graph test", flush=True)
for M in Ms:
    x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
    ids = torch.rand(M, E, device=dev).topk(TOPK, dim=1).indices.to(torch.int32)
    wts = torch.softmax(torch.randn(M, TOPK, device=dev), dim=1)
    def run():
        return wr.run(x=x, w1_weight=w1, w1_weight_sf=w1_sf_mma, w1_alpha=alpha1,
                      fc2_input_scale=fc2_scale, w2_weight=w2, w2_weight_sf=w2_sf_mma,
                      w2_alpha=alpha2, token_selected_experts=ids, token_final_scales=wts)
    ref = run().clone(); torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o = run()
    torch.cuda.synchronize()
    g.replay(); torch.cuda.synchronize()
    err = (o.float() - ref.float()).abs().max().item()
    s_, e_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s_.record()
    for _ in range(20): g.replay()
    e_.record(); torch.cuda.synchronize()
    print(f"graph M={M:6d}: replay ok, maxdiff={err:.3g}, {s_.elapsed_time(e_)/20*1e3:.1f} us", flush=True)
