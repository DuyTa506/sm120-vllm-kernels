import math, torch
import flashinfer.attention.cute_dsl.sm120_fmha as m
HQ, HKV, D, PAGE, QL, KL = 16, 2, 256, 16, 3584, 8192
dev = "cuda"; torch.manual_seed(0)
npg = KL // PAGE + 3
cache = (torch.randn(npg, HKV, PAGE, 2 * D, device=dev) ).to(torch.float8_e4m3fn)   # vLLM-like packed [K|V]
k_v, v_v = cache[..., :D], cache[..., D:]
k_c, v_c = k_v.contiguous(), v_v.contiguous()
perm = torch.randperm(npg, device=dev)[: KL // PAGE].to(torch.int32)[None, :].contiguous()
q8 = torch.randn(QL, HQ, D, device=dev).to(torch.float8_e4m3fn)
cu = torch.tensor([0, QL], dtype=torch.int32, device=dev); sk = torch.tensor([KL], dtype=torch.int32, device=dev)
o_c = torch.empty(QL, HQ, D, device=dev, dtype=torch.bfloat16); o_v = torch.empty_like(o_c)
m.sm120_fmha_fp8_paged_prefill(q8, k_c, v_c, o_c, perm, sk, cu, is_causal=True, max_seqlen_q=QL)
m._validate_hnd_paged_pool = lambda name, pool: None
try:
    m.sm120_fmha_fp8_paged_prefill(q8, k_v, v_v, o_v, perm, sk, cu, is_causal=True, max_seqlen_q=QL)
    torch.cuda.synchronize()
    print("strided view: max abs diff vs compact =", (o_c.float() - o_v.float()).abs().max().item())
except Exception as ex:
    print("strided view FAILED:", type(ex).__name__, str(ex)[:300])
# timing strided
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
for name, kk, vv, oo in (("compact", k_c, v_c, o_c), ("strided", k_v, v_v, o_v)):
    s.record()
    for _ in range(10): m.sm120_fmha_fp8_paged_prefill(q8, kk, vv, oo, perm, sk, cu, is_causal=True, max_seqlen_q=QL)
    e.record(); torch.cuda.synchronize(); print(name, f"{s.elapsed_time(e)/10*1e3:.1f} us")
