"""Bitwise tests: production op vs verbatim copy vs verify+commit vs deferred.
Run inside vllm/vllm-openai:v0.32.0 (needs the production op and the HF cache)."""
import os, sys, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import vllm  # noqa: F401  (loads torch.ops._C)
from vllm import _custom_ops as ops
from common import *

ext = build()
dev = "cuda"
fails = 0


def eq(x, y):
    return torch.equal(x.contiguous().view(torch.int32), y.contiguous().view(torch.int32)) if x.dtype == torch.float32 \
        else torch.equal(x.contiguous().view(torch.int16), y.contiguous().view(torch.int16))


def check(name, ok):
    global fails
    if not ok:
        fails += 1
    print(("PASS " if ok else "FAIL ") + name, flush=True)


def rand_state(S, gen, scale=0.3):
    return (torch.randn(S, HV, V, K, generator=gen, device=dev) * scale)


def run_prod(inp, params, si, acc, state, sigmoid=False):
    A, dt, nw = params
    ops.fused_gdn_decode_post_conv_mtp(inp.mixed_qkv, inp.a, inp.b, A, dt, si, inp.cu, acc, state, inp.gate, nw, inp.out,
                                        SCALE, EPS, "sigmoid" if sigmoid else "silu")


def mine_args(inp, params, si, acc, state, out):
    A, dt, nw = params
    return (inp.mixed_qkv, inp.a, inp.b, A, dt, si, inp.cu, acc, state, inp.gate, nw, out)


def scratch(B, device=dev, k_heads=H):
    return (torch.zeros(B, W, HV, V, device=device), torch.zeros(B, W, k_heads, K, device=device),
            torch.zeros(B, W, HV, device=device))


def ptrs(ts):
    return torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)


def commit(state, rep, si, inp, acc, n_new, bound_m=None, k_heads=H, **kw):
    d, k, dec = rep
    strides = torch.tensor([state.stride(0)], dtype=torch.int64, device=dev)
    ext.commit(ptrs([state]), ptrs([d]), ptrs([k]), ptrs([dec]), strides, si, inp.cu, acc, n_new, k_heads=k_heads,
               bound_m=bound_m, **kw)


def single_step_tests(dt_dtype, w_dtype, sigmoid):
    params = real_params(dt_dtype, w_dtype)
    gen = torch.Generator(device=dev); gen.manual_seed(1234)
    tag = f"dt={str(dt_dtype)[6:]},w={str(w_dtype)[6:]},{'sigmoid' if sigmoid else 'silu'}"
    for B in (1, 2, 5, 16, 64):
        for mode in ("full3", "mixed"):
            qlens = [3] * B if mode == "full3" else [int(x) for x in torch.randint(1, 4, (B,), generator=gen, device=dev).tolist()]
            inp = Inputs(qlens, gen)
            si = window_indices(B)
            S = 1 + B * W
            acc = torch.randint(1, 4, (B,), generator=gen, device=dev, dtype=torch.int32)
            st0 = rand_state(S, gen)
            ql = torch.tensor(qlens, dtype=torch.int32, device=dev)
            n_new = torch.minimum(torch.randint(1, 4, (B,), generator=gen, device=dev, dtype=torch.int32), ql)
            # production
            sp = st0.clone(); run_prod(inp, params, si, acc, sp, sigmoid); out_prod = inp.out.clone(); inp.out.zero_()
            # verbatim copy
            sb = st0.clone(); ext.base_decode(*mine_args(inp, params, si, acc, sb, inp.out), sigmoid=sigmoid)
            out_base = inp.out.clone(); inp.out.zero_()
            check(f"[{tag}] B={B} {mode}: base copy == production (out)", eq(out_prod, out_base))
            check(f"[{tag}] B={B} {mode}: base copy == production (state)", eq(sp, sb))
            # verify: no state writes, same outputs
            sv = st0.clone(); rep = scratch(B)
            ext.verify(*mine_args(inp, params, si, acc, sv, inp.out), rep=rep, sigmoid=sigmoid)
            check(f"[{tag}] B={B} {mode}: verify out == production", eq(out_prod, inp.out))
            check(f"[{tag}] B={B} {mode}: verify leaves state untouched", eq(st0, sv))
            # commit: window slot n-1 equals what production left there
            bm = torch.zeros(B, dtype=torch.int32, device=dev)
            hi = (n_new - 1).clamp(min=0)
            r = torch.randint(0, 1000, (B,), generator=gen, device=dev)
            bm = torch.where(n_new > 1, 1 + (r % hi.clamp(min=1)), torch.zeros_like(n_new)).to(torch.int32)  # m in [1, n-1]
            use_bm = bm if mode == "mixed" else None
            commit(sv, rep, si, inp, acc, n_new, use_bm)
            ok = True; okb = True; ok_c = True
            for r_ in range(B):
                n = int(n_new[r_]); slot = int(si[r_, n - 1])
                ok &= eq(sv[slot], sp[slot]); ok_c &= eq(sv[slot], sb[slot])
                if use_bm is not None and int(bm[r_]) > 0:
                    okb &= eq(sv[int(si[r_, int(bm[r_]) - 1])], sp[int(si[r_, int(bm[r_]) - 1])])
            check(f"[{tag}] B={B} {mode}: commit state[n-1] == verbatim copy (all {B} rows)", ok_c)
            check(f"[{tag}] B={B} {mode}: commit state[n-1] == production (all {B} rows)", ok)
            if use_bm is not None:
                check(f"[{tag}] B={B} {mode}: commit boundary state[m-1] == production", okb)


def chain_test(steps=40, B=8, seed=7):
    """Reference chain (production op, n tracked via acc) vs deferred chain (fold + base, acc=1)."""
    params = real_params()
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    si = window_indices(B); S = 1 + B * W
    st_ref = torch.zeros(S, HV, V, K, device=dev)
    st_def = torch.zeros(S, HV, V, K, device=dev)
    st_ver = torch.zeros(S, HV, V, K, device=dev)       # verify + same-step commit chain
    rep_d = scratch(B, k_heads=HV); rep_v = scratch(B)
    acc_ref = torch.ones(B, dtype=torch.int32, device=dev)
    ones = torch.ones(B, dtype=torch.int32, device=dev)
    fold = torch.zeros(B, dtype=torch.int32, device=dev)
    all_ok_out_d = all_ok_out_v = True; worst = 0
    for s in range(steps):
        qlens = [3 if float(torch.rand(1, generator=gen, device=dev)) < 0.85 else int(torch.randint(1, 3, (1,), generator=gen, device=dev)) for _ in range(B)]
        inp = Inputs(qlens, gen)
        ql = torch.tensor(qlens, dtype=torch.int32, device=dev)
        n_new = torch.minimum(torch.randint(1, 4, (B,), generator=gen, device=dev, dtype=torch.int32), ql)
        # reference (production): source si[acc-1], writes all window slots
        run_prod(inp, params, si, acc_ref, st_ref); out_ref = inp.out.clone(); inp.out.zero_()
        # verify + commit chain: source si[acc-1], commit writes si[n-1]
        ext.verify(*mine_args(inp, params, si, acc_ref, st_ver, inp.out), rep=rep_v)
        all_ok_out_v &= eq(out_ref, inp.out); inp.out.zero_()
        commit(st_ver, rep_v, si, inp, acc_ref, n_new)
        # deferred chain: lagging base at si[0] (acc=1), fold previous n
        ext.deferred(*mine_args(inp, params, si, ones, st_def, inp.out), rep=rep_d, fold_n=fold)
        all_ok_out_d &= eq(out_ref, inp.out); inp.out.zero_()
        fold = n_new.clone()
        acc_ref = n_new.clone()
        # states of the verify+commit chain must match production at slot n-1 every step
        for r_ in range(B):
            slot = int(si[r_, int(n_new[r_]) - 1])
            if not eq(st_ver[slot], st_ref[slot]):
                all_ok_out_v = False
    check(f"chain {steps} steps B={B}: verify+commit outputs and states bitwise == production", all_ok_out_v)
    check(f"chain {steps} steps B={B}: deferred outputs bitwise == production", all_ok_out_d)
    # flush the deferred chain: fold last n into slot n-1 and compare with the production state there
    inp_flush = Inputs([3] * B, gen)
    inp_flush.cu = torch.arange(0, 3 * B + 1, 3, dtype=torch.int32, device=dev)
    commit(st_def, rep_d, si, inp_flush, ones, fold, k_heads=HV)
    ok = True
    for r_ in range(B):
        slot = int(si[r_, int(fold[r_]) - 1])
        ok &= eq(st_def[slot], st_ref[slot])
    check(f"chain {steps} steps B={B}: deferred base+flush state == production state", ok)



def marker_boundary_test(B=16, MB=2176, seed=3):
    """Epoch marker gating and the in-kernel boundary derivation (idx_mapping, post-step num_computed, block size)."""
    params = real_params()
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    si = window_indices(B); S = 1 + B * W
    inp = Inputs([3] * B, gen)
    acc = torch.randint(1, 4, (B,), generator=gen, device=dev, dtype=torch.int32)
    st0 = rand_state(S, gen)
    n_new = torch.randint(1, 4, (B,), generator=gen, device=dev, dtype=torch.int32)
    sp = st0.clone(); run_prod(inp, params, si, acc, sp)
    # pick c so that a boundary falls inside [c+1, c+n] for about half of the rows
    c = torch.randint(0, 5000, (B,), generator=gen, device=dev, dtype=torch.int32)
    hit = torch.arange(B, device=dev) % 2 == 0
    c = torch.where(hit, (c // MB + 1) * MB - torch.randint(0, 3, (B,), generator=gen, device=dev, dtype=torch.int32), c).to(torch.int32)
    newc_by_slot = torch.zeros(B + 5, dtype=torch.int32, device=dev)
    idx_mapping = torch.randperm(B + 5, generator=gen, device=dev)[:B].to(torch.int32)
    newc_by_slot[idx_mapping.long()] = c + n_new
    # expected m per row
    nc = c + n_new; b = (nc // MB) * MB
    m_exp = torch.where(b >= c + 1, b - c, torch.zeros_like(c)); m_exp = torch.where((m_exp > 0) & (m_exp < n_new), m_exp, torch.zeros_like(c))
    sv = st0.clone(); rep = scratch(B)
    epoch = torch.tensor([7], dtype=torch.int32, device=dev)
    marker = torch.full((B,), -1, dtype=torch.int32, device=dev)
    ext.verify(*mine_args(inp, params, si, acc, sv, inp.out), rep=rep, marker=marker, epoch=epoch)
    check("marker set to epoch by verify", bool((marker == 7).all()))
    stale = epoch.clone() + 1   # commit with a different epoch must do nothing
    sv2 = sv.clone()
    commit(sv2, rep, si, inp, acc, n_new, marker_ptrs=ptrs([marker]), epoch=stale)
    check("commit skips rows whose marker != epoch", eq(sv2, sv))
    commit(sv, rep, si, inp, acc, n_new, marker_ptrs=ptrs([marker]), epoch=epoch, idx_mapping=idx_mapping,
           newc=newc_by_slot, mamba_block=MB)
    ok = okb = True
    for r_ in range(B):
        n = int(n_new[r_]); slot = int(si[r_, n - 1]); ok &= eq(sv[slot], sp[slot])
        m = int(m_exp[r_])
        if m > 0:
            bs = int(si[r_, m - 1]); okb &= eq(sv[bs], sp[bs])
    check(f"marker+idx_mapping commit: final states == production ({int((m_exp>0).sum())} boundary rows)", ok)
    check("in-kernel boundary state == production", okb)


if __name__ == "__main__":
    single_step_tests(torch.bfloat16, torch.bfloat16, False)
    single_step_tests(torch.float32, torch.float32, False)
    single_step_tests(torch.bfloat16, torch.bfloat16, True)
    marker_boundary_test()
    chain_test(40, 8, 7)
    chain_test(40, 64, 8)
    print("TOTAL FAILS:", fails)
    sys.exit(1 if fails else 0)
