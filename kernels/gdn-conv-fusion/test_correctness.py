"""Correctness: fused conv1d+post-conv vs causal_conv1d_fn + fused_post_conv_prep.

Real layer weights + real-text activations. Cases: 8k single sequence, varlen batch
with mixed has_initial_state and scattered cache lines, very short sequences
(seqlen < state_len, both conv_state update branches), decode rows in front of
prefills (raw output), both conv-state layouts, prefix-caching (APC) intermediate
state stores, null-block skip, and g vs exp(g).
"""

import sys

import torch

import os

from common import DEV, DIM, H, HV, K, conv_meta, layer_params, make_conv_states, make_inputs, reference
from fused_conv_post_conv import fused_conv1d_post_conv_prep

torch.manual_seed(0)
dev = DEV
SUBS = [1, 2, 4]
QUICK = os.environ.get("QUICK", "0") == "1"  # small shapes (Triton interpreter on CPU)
FAIL = []
NOTE = []


def err(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    mx = d.max().item() if d.numel() else 0.0
    rel = (d.max() / b.abs().max().clamp_min(1e-30)).item() if d.numel() else 0.0
    return mx, rel, (d > 0).float().mean().item() if d.numel() else 0.0


def max_ulps(got, ref):
    """largest difference in units of the reference's bf16 ulp"""
    g, r = got.float(), ref.float()
    ulp = torch.where(r == 0, torch.full_like(r, 2.0**-133),
                      torch.exp2(torch.floor(torch.log2(r.abs())) - 7))
    return ((g - r).abs() / ulp).max().item() if r.numel() else 0.0


def check(name, got, ref, tol=0.0):
    mx, rel, frac = err(got, ref)
    ok = mx <= tol
    note = ""
    if not ok and got.dtype == torch.bfloat16 and name in ("q", "k"):
        u = max_ulps(got, ref)
        note = f" ({u:.0f} ulp)"
        ok = u <= 1.0 and frac < 1e-4  # reduction-order rounding flips only
        if ok:
            NOTE.append(name)
    print(f"  {name:10s} max_abs={mx:.3e} max_rel={rel:.3e} frac_diff={frac:.2e}{note} {'OK' if ok else 'FAIL'}")
    if not ok:
        FAIL.append(name)
    return mx


def run_case(title, lens, init, layout="SD", raw_seqs=0, exp_g=True, apc=False,
             null_seq=None, layer=0, use_meta=True):
    print(f"[{title}] lens={lens if len(lens) < 12 else str(lens[:12]) + '...'} layout={layout} "
          f"raw_seqs={raw_seqs} exp_g={exp_g} apc={apc}")
    T = sum(lens)
    P = layer_params(layer)
    x, a, b = make_inputs(T, layer=layer, seed=len(lens) + T)
    nseq = len(lens)
    qsl_cpu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
    qsl = qsl_cpu.to(dev)
    has_init = torch.tensor(init, dtype=torch.bool, device=dev)
    meta = conv_meta(qsl_cpu) if use_meta else None
    num_lines = 4 * nseq + 8
    if apc:
        num_lines = max(num_lines, nseq * (max(lens) // 64 + 3) + 2)
    base, cs = make_conv_states(num_lines, layout)
    # realistic conv_state contents: rows of real mixed_qkv
    src = make_inputs(3 * num_lines, layer=layer, seed=99)[0].reshape(num_lines, 3, DIM)
    cs.copy_(src.transpose(1, 2))
    apc_kw = {}
    if apc:
        # 2-D cache_indices (batch, n_blocks): block bookkeeping as in mamba APC mode
        block = 64
        nblk = max(lens) // block + 3
        perm = torch.randperm(num_lines - 1, device=dev)[: nseq * nblk] + 1
        cache_indices = perm.view(nseq, nblk).to(torch.int32)
        # block-aligned computed prefix (vLLM align mode): the initial state lives in
        # block first-1, so no program reads a cache line another program writes
        computed = torch.tensor([block * (1 + i % 2) for i in range(nseq)],
                                dtype=torch.int32, device=dev)
        computed = torch.where(has_init, computed, torch.zeros_like(computed))
        first = (computed // block).to(torch.int32)
        last = ((computed + torch.tensor(lens, device=dev) - 1) // block).to(torch.int32)
        init_idx = torch.clamp(first - (computed % block == 0).int() * has_init.int(), min=0)
        apc_kw = dict(block_idx_first_scheduled_token=first, block_idx_last_scheduled_token=last,
                      initial_state_idx=init_idx.to(torch.int32), num_computed_tokens=computed,
                      block_size_to_align=block)
    else:
        cache_indices = (torch.randperm(num_lines - 1, device=dev)[:nseq] + 1).to(torch.int32)
    if null_seq is not None:
        if apc:
            cache_indices[null_seq, :] = 0
        else:
            cache_indices[null_seq] = 0
    raw_tokens = sum(lens[:raw_seqs])

    cs_ref = cs.clone()
    ref = reference(x, P.conv_w, cs_ref, qsl, cache_indices, has_init, a, b, P.A_log, P.dt_bias,
                    meta=meta, output_g_exp=exp_g, num_raw_tokens=raw_tokens, apc=apc_kw)
    rows = torch.ones(T - raw_tokens, dtype=torch.bool, device=dev)
    if null_seq is not None:  # skipped sequence: outputs undefined in both paths
        s, e = int(qsl_cpu[null_seq]) - raw_tokens, int(qsl_cpu[null_seq + 1]) - raw_tokens
        rows[max(s, 0):e] = False
    for sub in SUBS:
        base_fused = base.clone()
        cs_fused = base_fused.transpose(-1, -2) if layout == "SD" else base_fused
        got = fused_conv1d_post_conv_prep(
            x, P.conv_w, None, cs_fused, qsl, cache_indices, has_init, a, b, P.A_log, P.dt_bias,
            num_k_heads=H, head_k_dim=K, head_v_dim=K, output_g_exp=exp_g, metadata=meta,
            num_raw_seqs=raw_seqs, num_raw_tokens=raw_tokens, sub=sub, **apc_kw)
        if dev == "cuda":
            torch.cuda.synchronize()
        print(f"  -- sub={sub}")
        for n, gt, rf in zip(["q", "k", "v", "g", "beta"], got[:5], ref[:5]):
            check(n, gt[rows], rf[rows])
        if raw_seqs:
            check("raw", got[5], ref[5])
        check("conv_state", cs_fused, cs_ref)


if QUICK:
    run_case("single", [40], [False])
    run_case("single init", [33], [True])
    run_case("varlen", [21, 9, 3, 2, 1, 17], [False, True, True, True, False, True])
    run_case("varlen DS", [19, 2, 8], [True, True, False], layout="DS")
    run_case("decode+prefill", [1, 1, 1, 30, 5], [True, False, True, False, True], raw_seqs=3)
    run_case("log g", [25, 3], [True, False], exp_g=False)
    run_case("no metadata", [26, 2], [True, False], use_meta=False)
    run_case("apc", [150, 70, 9], [True, False, True], apc=True)
    run_case("null block", [20, 12, 9], [True, True, False], null_seq=1)
else:
    cases = [
        ("8k single", [8192], [False]),
        ("8k single init", [8192], [True]),
        ("varlen", [3001, 1500, 2048, 1000, 600, 37, 5, 1], [False, True, True, False, True, False, True, True]),
        ("short", [1, 2, 3, 4, 2, 1, 9, 8], [True, True, True, True, False, False, True, False]),
    ]
    for c in cases:
        run_case(*c)
    run_case("varlen DS", [3001, 1500, 2048, 37, 2], [False, True, True, True, True], layout="DS")
    run_case("decode+prefill", [1, 1, 1, 1, 2000, 333, 7], [True, True, True, False, False, True, True], raw_seqs=4)
    run_case("log g", [2048, 900, 3], [True, False, True], exp_g=False)
    run_case("no metadata", [700, 300, 2], [True, False, True], use_meta=False)
    run_case("apc", [1000, 513, 64, 200, 7], [True, False, True, True, False], apc=True)
    run_case("null block", [500, 300, 200], [True, True, False], null_seq=1)
    run_case("layer 20", [4096, 1000, 17], [True, False, True], layer=20)
    run_case("many seqs", [int(x) for x in torch.randint(1, 300, (60,))], [bool(i % 3) for i in range(60)])


if FAIL:
    print("FAILED: " + ", ".join(FAIL))
elif NOTE:
    print(f"ALL OK: bit-exact except {len(NOTE)} q/k comparisons with isolated 1-ulp bf16 flips")
else:
    print("ALL OK (bit-exact)")
sys.exit(1 if FAIL else 0)
