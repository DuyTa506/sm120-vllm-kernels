"""Python side of the fused MoE router kernel (fused_router.cu).

    topk_w, topk_i, sh, logits = fused_router(x, w, topk, ...)

x [M,K] bf16, w [E+S,K] bf16 (the S shared-expert-gate rows are appended after the E router rows, exactly like
vLLM's `_combined_gate_weight` in the fused-shared-expert path). Everything else is a parameter:
scoring ("softmax" | "sigmoid" | "sqrtsoftplus" | "none"), renormalize, selection bias (DeepSeek
e_score_correction_bias: select on score+bias, weight = unbiased score), routed scaling factor, bf16
rounding of the logits (what vLLM's GateLinear does when its out_dtype is None), shared-gate rows.
Not covered: grouped top-k (n_group/topk_group), expert maps, EPLB, padding masks.
"""
import os
from torch.utils.cpp_extension import load
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_mod = None

SCORING = {"softmax": 0, "sigmoid": 1, "sqrtsoftplus": 2, "none": 3}
F_RENORM, F_ROUND_BF16, F_BIAS, F_WRITE_LOGITS, F_SH_FP32 = 1, 2, 4, 8, 16


def _ext():
    global _mod
    if _mod is None:
        bd = os.environ.get("FUSED_ROUTER_BUILD", "/tmp/fused_router_build")
        os.makedirs(bd, exist_ok=True)
        _mod = load("fused_router_ext", sources=[os.path.join(_HERE, "fused_router.cu")],
                    extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_120,code=sm_120"],
                    build_directory=bd, verbose=bool(int(os.environ.get("FUSED_ROUTER_VERBOSE", "0"))))
    return _mod


class Workspace:
    """Scratch for partial logits + ticket counters. One per call site if several streams can overlap;
    a single one is enough for layers that run back to back on one stream."""

    def __init__(self, device, max_tokens=128, max_rows=544, max_split=16):
        tb_max = (max_tokens + 31) // 32
        self.scratch = torch.empty(tb_max * max_split * 32 * max_rows, dtype=torch.float32, device=device)
        self.counters = torch.zeros(tb_max, dtype=torch.int32, device=device)


def default_config(M, num_rows):
    """(n-tiles, warps per CTA, K split). Picked from the CUDA-graph sweep on the RTX PRO 6000 (188 SMs) for
    E=256 (+1 gate row), K=2048: few CTAs and a small split win because the cross-CTA hand-off is latency-bound."""
    groups = (num_rows + 15) // 16
    nt = min(4, (M + 7) // 8)
    nw, split = (4, 4) if M <= 4 else (8, 4) if M <= 6 else (12, 2) if M <= 12 else (16, 2) if M <= 24 else (16, 2)
    if groups > 24:   # bigger E (e.g. 512): more row groups already give enough CTAs
        split = max(1, split // 2)
    return nt, nw, split


def default_cluster_config(M):
    """(n-tiles, warps per CTA, cluster size) of the cluster variant."""
    return min(4, (M + 7) // 8), 16, 16


def fused_router(x, w, topk, *, num_experts=None, scoring="softmax", renormalize=True, bias=None, scale=1.0,
                 round_logits=True, ws=None, write_logits=False, sh_fp32=False, cfg=None, out=None, pad=None, ablate=0, relacq=1, accmode=0, prefetch_next=None, mode="splitk"):
    M, K = x.shape
    R = w.shape[0]
    E = R if num_experts is None else num_experts
    S = R - E
    dev = x.device
    if mode == "cluster":
        nt, nw, cs = cfg if cfg is not None else default_cluster_config(M)
        nt = max(nt, (min(M, 32) + 7) // 8)
        if out is None:
            topk_w = torch.empty(M, topk, dtype=torch.float32, device=dev)
            topk_i = torch.empty(M, topk, dtype=torch.int32, device=dev)
        else:
            topk_w, topk_i = out
        sh = torch.empty(M, S, dtype=torch.float32 if sh_fp32 else torch.bfloat16, device=dev) if S else None
        logits = torch.empty(M, E, dtype=torch.bfloat16, device=dev) if write_logits else None
        flags = (F_RENORM if renormalize else 0) | (F_ROUND_BF16 if round_logits else 0) | \
                (F_BIAS if bias is not None else 0) | (F_WRITE_LOGITS if write_logits else 0) | (F_SH_FP32 if sh_fp32 else 0)
        _ext().fused_router_cluster(x, w, E, topk, SCORING[scoring], flags, float(scale), bias, topk_w, topk_i, sh, logits,
                                    pad, prefetch_next, nt, nw, cs, ablate)
        return topk_w, topk_i, sh, logits
    nt, nw, split = cfg if cfg is not None else default_config(M, R)
    nt = max(nt, (min(M, 32) + 7) // 8)
    if ws is None:
        ws = Workspace(dev, max(M, 32))
    if out is None:
        topk_w = torch.empty(M, topk, dtype=torch.float32, device=dev)
        topk_i = torch.empty(M, topk, dtype=torch.int32, device=dev)
    else:
        topk_w, topk_i = out
    sh = torch.empty(M, S, dtype=torch.float32 if sh_fp32 else torch.bfloat16, device=dev) if S else None
    logits = torch.empty(M, E, dtype=torch.bfloat16, device=dev) if write_logits else None
    flags = (F_RENORM if renormalize else 0) | (F_ROUND_BF16 if round_logits else 0) | \
            (F_BIAS if bias is not None else 0) | (F_WRITE_LOGITS if write_logits else 0) | (F_SH_FP32 if sh_fp32 else 0)
    _ext().fused_router(x, w, E, topk, SCORING[scoring], flags, float(scale), bias, ws.scratch, ws.counters,
                        topk_w, topk_i, sh, logits, pad, prefetch_next, nt, nw, split, ablate, relacq, accmode)
    return topk_w, topk_i, sh, logits


def noop(ncta=1, nthreads=32):
    _ext().noop(ncta, nthreads)
