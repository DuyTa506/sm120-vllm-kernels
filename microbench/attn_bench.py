"""Prefill attention at Qwen3.6-35B-A3B shapes: FlashInfer FA2 (bf16 Q, fp8 KV, what vLLM uses now)
vs FlashInfer SM120 FP8 FMHA (fp8 Q/K/V). Reports time, TFLOPS and error vs an fp32 reference."""
import math, sys, torch, flashinfer
from flashinfer.attention.cute_dsl.sm120_fmha import sm120_fmha_fp8_paged_prefill

HQ, HKV, D, PAGE = 16, 2, 256, int(sys.argv[1]) if len(sys.argv) > 1 else 2128
dev = "cuda"; torch.manual_seed(0)
cases = [(4608, 4608), (3584, 8192), (4096, 32768)][:int(sys.argv[2]) if len(sys.argv) > 2 else 3]

def timeit(fn, n=10):
    fn(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n

for QL, KL in cases:
    npages = math.ceil(KL / PAGE)
    # realistic-ish magnitudes: q/k come out of RMSNorm (unit RMS), v ~ N(0,1)
    q = torch.randn(QL, HQ, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(KL, HKV, D, device=dev)
    v = torch.randn(KL, HKV, D, device=dev)
    k8, v8 = k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn)
    # paged pools, HND (pages, Hkv, page, D); pad tail with zeros (finite)
    kp = torch.zeros(npages * PAGE, HKV, D, device=dev, dtype=torch.float8_e4m3fn); kp[:KL] = k8
    vp = torch.zeros_like(kp); vp[:KL] = v8
    k_hnd = kp.view(npages, PAGE, HKV, D).permute(0, 2, 1, 3).contiguous()
    v_hnd = vp.view(npages, PAGE, HKV, D).permute(0, 2, 1, 3).contiguous()
    k_nhd = kp.view(npages, PAGE, HKV, D).contiguous(); v_nhd = vp.view(npages, PAGE, HKV, D).contiguous()
    flops = 4 * HQ * D * (QL * KL - QL * QL / 2)

    # reference in fp32 from the fp8-rounded K/V and bf16 Q (bottom-right causal)
    with torch.no_grad():
        ref = torch.empty(QL, HQ, D, device=dev)
        kf, vf = k8.float(), v8.float()
        qi = torch.arange(QL, device=dev)[:, None] + (KL - QL); kj = torch.arange(KL, device=dev)[None, :]
        mask = kj <= qi
        for h in range(HQ):
            s = (q[:, h].float() @ kf[:, h // (HQ // HKV)].T) / math.sqrt(D)
            s.masked_fill_(~mask, float("-inf"))
            ref[:, h] = torch.softmax(s, -1) @ vf[:, h // (HQ // HKV)]

    def err(o):
        o = o.float(); return ((o - ref).norm() / ref.norm()).item()

    # 1) FlashInfer FA2 paged prefill, bf16 Q, fp8 KV (vLLM's current path)
    ws = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
    qo_indptr = torch.tensor([0, QL], dtype=torch.int32, device=dev)
    kv_indptr = torch.tensor([0, npages], dtype=torch.int32, device=dev)
    kv_indices = torch.arange(npages, dtype=torch.int32, device=dev)
    last = torch.tensor([KL - (npages - 1) * PAGE], dtype=torch.int32, device=dev)
    w.plan(qo_indptr, kv_indptr, kv_indices, last, HQ, HKV, D, PAGE, causal=True,
           q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn)
    o1 = w.run(q, (k_nhd, v_nhd))
    t1 = timeit(lambda: w.run(q, (k_nhd, v_nhd)))
    print(f"q={QL:5d} kv={KL:5d}  FA2 bf16Q/fp8KV : {t1*1e3:8.1f} us  {flops/t1/1e9:6.0f} TFLOPS  relerr={err(o1):.4f}", flush=True)

    # 2) SM120 FP8 FMHA (Q quantized per-tensor to fp8; scale folded into sm_scale)
    qs = (q.float().abs().amax() / 448.0).clamp(min=1e-8)
    q8 = (q.float() / qs).to(torch.float8_e4m3fn)
    o2 = torch.empty(QL, HQ, D, device=dev, dtype=torch.bfloat16)
    bt = kv_indices[None, :].contiguous()
    seqk = torch.tensor([KL], dtype=torch.int32, device=dev)
    sm = qs.item() / math.sqrt(D)
    run2 = lambda: sm120_fmha_fp8_paged_prefill(q8, k_hnd, v_hnd, o2, bt, seqk, qo_indptr, is_causal=True,
                                                 sm_scale=sm, max_seqlen_q=QL)
    try:
        t2 = timeit(run2)
        print(f"q={QL:5d} kv={KL:5d}  SM120 FP8 FMHA  : {t2*1e3:8.1f} us  {flops/t2/1e9:6.0f} TFLOPS  relerr={err(o2):.4f}  speedup {t1/t2:.2f}x", flush=True)
        # isolate: reference recomputed from the fp8-rounded Q, and FA2 fed the same rounded Q
        q_r = (q8.float() * qs).to(torch.bfloat16)
        o3 = w.run(q_r, (k_nhd, v_nhd))
        print(f"    FA2 with fp8-rounded Q relerr={err(o3):.4f}  (Q-quant share);  FMHA vs FA2(rounded Q) relerr={((o2.float()-o3.float()).norm()/o3.float().norm()).item():.4f}  (P/accum share)", flush=True)
    except Exception as ex:
        print(f"q={QL:5d} kv={KL:5d}  SM120 FP8 FMHA  : FAILED {type(ex).__name__}: {str(ex)[:300]}", flush=True)
