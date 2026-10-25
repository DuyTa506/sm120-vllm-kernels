import sys, torch, flashinfer
from flashinfer.fused_moe import cutlass_fused_moe
E, TOPK, K, N = 256, 8, 2048, 512
Ms = [int(x) for x in sys.argv[1:]] or [512, 1024, 2048, 4608, 8192]
dev = "cuda"; torch.manual_seed(0)
w1 = torch.randint(0, 256, (E, 2 * N, K // 2), dtype=torch.uint8, device=dev)
w2 = torch.randint(0, 256, (E, K, N // 2), dtype=torch.uint8, device=dev)
sf = lambda *s: torch.randint(0x30, 0x40, s, dtype=torch.uint8, device=dev)
w1_sf = sf(E, 2 * N, K // 16).view(torch.int32) if False else sf(E, 2 * N, K // 16)
w2_sf = sf(E, K, N // 16)
w1_sf_i = w1_sf.view(torch.float8_e4m3fn).view(torch.int32) if False else w1_sf.view(torch.int32)
w2_sf_i = w2_sf.view(torch.int32)
one = torch.ones(1, device=dev, dtype=torch.float32)
a1g = torch.tensor(1.0, device=dev); a2g = torch.tensor(1.0, device=dev)
al1 = torch.full((E,), 0.01, device=dev); al2 = torch.full((E,), 0.01, device=dev)
for M in Ms:
    x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
    ids = torch.rand(M, E, device=dev).topk(TOPK, dim=1).indices.to(torch.int32)
    wts = torch.softmax(torch.randn(M, TOPK, device=dev), dim=1)
    out = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
    def run():
        xq, xsf = flashinfer.fp4_quantize(x, a1g, 16, False, True)
        return cutlass_fused_moe(input=xq, token_selected_experts=ids, token_final_scales=wts,
            fc1_expert_weights=w1.view(torch.long), fc2_expert_weights=w2.view(torch.long),
            output_dtype=torch.bfloat16, output=out,
            quant_scales=[a1g, w1_sf_i, al1, a2g, w2_sf_i, al2], input_sf=xsf)
    run(); torch.cuda.synchronize()
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        run()
    torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    torch.cuda.synchronize()
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    n = 50; s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): g.replay()
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / n
    print(f"cutlass(graph) M={M:6d}: {ms*1e3:8.1f} us/layer", flush=True)
