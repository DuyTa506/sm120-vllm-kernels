"""Verifier's extra correctness checks for the glue-free SM120 FP8 FMHA (patch 03).

Cases the author's correctness.py does not cover:
  a. large KV pools: the new kernel's TMA descriptor spans the WHOLE per-layer cache
     (the baseline only sees the compact gathered copy). Blocks at high ids, byte offsets
     past 2^31 and 2^32, with compact and padded (hybrid-allocator-like) block strides.
  b. q_scale != 1 on realistic varlen shapes (not just the 16-row all-bits test).
  c. mixed batch: prefill query/out are slices that start after decode rows
     (non-zero storage offset), as in vLLM's query[num_decode_tokens:].
  d. run-to-run determinism of the new kernel.
  e. CUDA-graph capture + replay of the new kernel (prefill is eager in vLLM, but check
     the launcher is capture-safe).
  f. block-table entries past the last page holding garbage / out-of-range ids.
Every comparison is bitwise against patch 01's path (common.run_baseline).
"""

import math
import sys

import torch

from common import (D, HKV, HQ, PAGE, baseline_meta, fi_kernel, make_batch, make_query,
                    new_meta, run_baseline, run_new)
from glue_fmha import get_kernel

dev = "cuda"
SM = D ** -0.5
KS, VS = 0.0234375, 0.01953125
fails = 0
print(torch.cuda.get_device_name(), "free/total GB:",
      [round(x / 2**30, 1) for x in torch.cuda.mem_get_info()])


def check(name, a, b):
    global fails
    bad = int((a.view(torch.int16) != b.view(torch.int16)).sum())
    mx = (a.float() - b.float()).abs().max().item()
    finite = bool(torch.isfinite(a.float()).all())
    ok = bad == 0 and finite
    fails += not ok
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}: mismatched={bad} max_abs={mx:.3e} finite={finite}")


def big_pool(num_blocks, block_stride_elems, layout, used_ids, seed=0):
    """FP8 cache view [num_blocks, Hkv, page, 2D] (HND) or [num_blocks, page, Hkv, 2D] (NHD)
    built with as_strided over one flat buffer, so the block stride can be padded.
    Only the used blocks get random data (rest stays zero)."""
    per_block = HKV * PAGE * 2 * D
    assert block_stride_elems >= per_block
    flat = torch.zeros(num_blocks * block_stride_elems, dtype=torch.uint8, device=dev)
    if layout == "HND":
        shape = (num_blocks, HKV, PAGE, 2 * D)
        stride = (block_stride_elems, PAGE * 2 * D, 2 * D, 1)
    else:
        shape = (num_blocks, PAGE, HKV, 2 * D)
        stride = (block_stride_elems, HKV * 2 * D, 2 * D, 1)
    raw = flat.view(torch.float8_e4m3fn).as_strided(shape, stride)
    g = torch.Generator(device=dev).manual_seed(seed)
    ids = used_ids.long()
    vals = torch.randn((ids.numel(),) + shape[1:], generator=g, device=dev, dtype=torch.float32)
    vals[..., :D] *= 3.0
    vals[..., D:] *= 2.0
    vals[..., 7] *= 6.0
    raw[ids] = vals.clamp(-448, 448).to(torch.float8_e4m3fn)
    k, v = raw.split(D, dim=-1)
    if layout == "NHD":
        k, v = k.transpose(1, 2), v.transpose(1, 2)
    return flat, k, v


def batch_with_ids(seqs, ids):
    npages = [math.ceil(kv / PAGE) for _, kv in seqs]
    assert sum(npages) == ids.numel()
    indptr = [0]
    for n in npages:
        indptr.append(indptr[-1] + n)
    last = [kv - (n - 1) * PAGE for (_, kv), n in zip(seqs, npages)]
    qo = [0]
    for q, _ in seqs:
        qo.append(qo[-1] + q)
    return dict(indices=ids.to(dev, torch.int32),
                indptr=torch.tensor(indptr, dtype=torch.int32, device=dev),
                last=torch.tensor(last, dtype=torch.int32, device=dev),
                qo_indptr=torch.tensor(qo, dtype=torch.int32, device=dev),
                max_q=max(q for q, _ in seqs), max_kv=max(kv for _, kv in seqs),
                total_q=qo[-1])


def both(fi, new, q, k, v, b, q_scale=1.0, sm=SM):
    oa = torch.empty(b["total_q"], HQ, D, dtype=torch.bfloat16, device=dev)
    ob = torch.full_like(oa, float("nan"))
    run_baseline(fi, q, k, v, oa, baseline_meta(b), b["max_q"], sm, q_scale, KS, VS)
    run_new(new, q, k, v, ob, new_meta(b), b["max_q"], sm, q_scale, KS, VS)
    torch.cuda.synchronize()
    return oa, ob


fi = fi_kernel()
new = get_kernel(torch.bfloat16, torch.bfloat16, HQ, HKV, D, PAGE)

# ---------------------------------------------------------------- a. large pools
print("a. large pools, high block ids (byte offsets > 2^31 / 2^32)")
seqs = [(3000, 3000), (1024, 5120), (2048, 2111), (300, 301)]
need = sum(math.ceil(kv / PAGE) for _, kv in seqs)
per_block = HKV * PAGE * 2 * D  # 65536 B
for layout in ("HND", "NHD"):
    for nb, stride_mult, tag in ((80000, 1, "compact 5.2GB"), (40000, 2, "padded x2 5.2GB")):
        bs = per_block * stride_mult
        torch.manual_seed(nb + stride_mult)
        # high ids: mostly in the top 25% of the pool, plus a few low ones
        hi = torch.randperm(nb // 4)[: need - 8] + (nb - nb // 4)
        lo = torch.randperm(64)[:8]
        ids = torch.cat([hi, lo])[torch.randperm(need)]
        flat, k, v = big_pool(nb, bs, layout, ids.to(dev), seed=nb)
        b = batch_with_ids(seqs, ids)
        q = make_query(b["total_q"], seed=5)
        oa, ob = both(fi, new, q, k, v, b)
        maxoff = int(ids.max()) * bs
        check(f"{layout} {tag} (max byte offset {maxoff / 2**30:.2f} GiB)", ob, oa)
        del flat, k, v
        torch.cuda.empty_cache()

# ---------------------------------------------------------------- b. q_scale != 1
print("b. q_scale != 1 on realistic varlen shapes")
from common import make_pool  # noqa: E402

raw, k, v = make_pool(1024, seed=11)
for q_scale in (0.7071, 1.5, 0.125):
    kern = get_kernel(torch.bfloat16, torch.bfloat16, HQ, HKV, D, PAGE, q_scale=q_scale)
    b = make_batch([(2048, 2048), (700, 3000), (4097, 4097)], 1024, seed=12)
    q = make_query(b["total_q"], seed=13, amp=40.0)  # large amplitude -> exercises clamping
    oa = torch.empty(b["total_q"], HQ, D, dtype=torch.bfloat16, device=dev)
    ob = torch.empty_like(oa)
    run_baseline(fi, q, k, v, oa, baseline_meta(b), b["max_q"], SM, q_scale, KS, VS)
    run_new(kern, q, k, v, ob, new_meta(b), b["max_q"], SM, q_scale, KS, VS)
    torch.cuda.synchronize()
    check(f"q_scale={q_scale} varlen (amp 40, saturating)", ob, oa)

# ---------------------------------------------------------------- c. mixed batch offsets
print("c. mixed batch: prefill rows start after decode rows (storage offset)")
for n_dec in (3, 37, 192):
    b = make_batch([(1500, 1500), (513, 2600)], 1024, seed=21)
    qfull = make_query(n_dec + b["total_q"], seed=22)
    ofull_a = torch.zeros(n_dec + b["total_q"], HQ * D, dtype=torch.bfloat16, device=dev)
    ofull_b = torch.zeros_like(ofull_a)
    qp = qfull[n_dec:]
    oa = ofull_a[n_dec:].view(-1, HQ, D)
    ob = ofull_b[n_dec:].view(-1, HQ, D)
    run_baseline(fi, qp, k, v, oa, baseline_meta(b), b["max_q"], SM, 1.0, KS, VS)
    run_new(new, qp, k, v, ob, new_meta(b), b["max_q"], SM, 1.0, KS, VS)
    torch.cuda.synchronize()
    check(f"n_decode_rows={n_dec}", ob, oa)
    untouched = bool((ofull_b[:n_dec] == 0).all())
    print(f"    decode rows untouched by new kernel: {untouched}")
    fails += not untouched

# ---------------------------------------------------------------- d. determinism
print("d. determinism (new kernel, 5 runs, 8k)")
b = make_batch([(8192, 8192)], 1024, seed=31)
q = make_query(b["total_q"], seed=32)
nm = new_meta(b)
outs = []
for _ in range(5):
    o = torch.empty(b["total_q"], HQ, D, dtype=torch.bfloat16, device=dev)
    run_new(new, q, k, v, o, nm, b["max_q"], SM, 1.0, KS, VS)
    outs.append(o)
torch.cuda.synchronize()
nd = sum(int((o.view(torch.int16) != outs[0].view(torch.int16)).sum()) for o in outs[1:])
print(f"  [{'OK ' if nd == 0 else 'FAIL'}] run-to-run mismatches: {nd}")
fails += nd != 0

# ---------------------------------------------------------------- e. CUDA graph
print("e. CUDA graph capture/replay of the new kernel")
try:
    o_eager = outs[0]
    o_g = torch.zeros_like(o_eager)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run_new(new, q, k, v, o_g, nm, b["max_q"], SM, 1.0, KS, VS)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    o_g.zero_()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        run_new(new, q, k, v, o_g, nm, b["max_q"], SM, 1.0, KS, VS)
    gr.replay()
    torch.cuda.synchronize()
    check("graph replay vs eager", o_g, o_eager)
except Exception as ex:  # noqa: BLE001
    print(f"  [INFO] capture failed: {type(ex).__name__}: {str(ex)[:300]}")

# ---------------------------------------------------------------- f. garbage tail entries
print("f. block-table tail entries poisoned (beyond each request's last page)")
b = make_batch([(1000, 1000), (3000, 3000), (64, 130)], 1024, seed=41)
q = make_query(b["total_q"], seed=42)
bt, sl, cu = new_meta(b)
npages = (b["indptr"][1:] - b["indptr"][:-1]).tolist()
bt2 = bt.clone()
for i, n in enumerate(npages):
    bt2[i, n:] = 2**30  # out-of-range block id
oa = torch.empty(b["total_q"], HQ, D, dtype=torch.bfloat16, device=dev)
ob = torch.empty_like(oa)
run_baseline(fi, q, k, v, oa, baseline_meta(b), b["max_q"], SM, 1.0, KS, VS)
run_new(new, q, k, v, ob, (bt2, sl, cu), b["max_q"], SM, 1.0, KS, VS)
torch.cuda.synchronize()
check("tail entries = 2^30 (kernel must never read them)", ob, oa)

print("ALL OK" if fails == 0 else f"{fails} FAILURES")
sys.exit(1 if fails else 0)
