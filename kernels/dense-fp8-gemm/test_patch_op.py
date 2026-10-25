"""Smoke test of the vLLM patch module (run with the patch files bind-mounted over the image and
VLLM_SM120_SKINNY_FP8=1): JIT build, eager op vs cutlass_scaled_mm on all dense shapes, the op under
torch.compile with a dynamic token dim, and inside CUDA graph capture."""
import os
import sys

import torch

assert os.environ.get("VLLM_SM120_SKINNY_FP8") == "1"
sys.path.insert(0, "/work/kernels/dense-fp8-gemm")
from common import SHAPES, load_weight, make_act, quant_per_token  # noqa: E402

import vllm.model_executor.kernels.linear.scaled_mm.sm120_skinny_fp8 as m  # noqa: E402
from vllm import _custom_ops as ops  # noqa: E402

op = torch.ops.vllm.sm120_fp8_scaled_mm
ok = True
for sn in SHAPES:
    w, s = load_weight(sn, 0 if sn.startswith("gdn") else (3 if sn.startswith("attn") else None))
    s = s.view(-1, 1)  # vLLM keeps channel scales as [N, 1]
    for M in (1, 3, 4, 12, 16, 24, 48, 64, 65, 192, 1000):
        a, a_s = quant_per_token(make_act(M, w.shape[1], seed=M))
        ref = ops.cutlass_scaled_mm(a, w.t(), a_s, s, torch.bfloat16)
        out = op(a, w.t(), a_s, s)
        d = (out.float() - ref.float()).abs().max().item()
        eq = (out.view(torch.int16) == ref.view(torch.int16)).float().mean().item()
        used = m._pick(M, w.shape[0], w.shape[1]) is not None
        bad = d > 0.02 * ref.float().abs().max().item() or (not used and eq != 1.0)
        ok &= not bad
        print(f"{sn:9s} M={M:5d} skinny={used!s:5s} bitEq={100*eq:6.2f}% maxdiff={d:.3g} {'BAD' if bad else ''}")

# torch.compile with a dynamic token dimension
w, s = load_weight("gdn_out", 0)
s = s.view(-1, 1)


def f(x):
    xq, xs = quant_per_token(x)
    return op(xq, w.t(), xs, s)


cf = torch.compile(f, dynamic=True, fullgraph=False)
for M in (4, 16, 48, 300):
    x = make_act(M, w.shape[1], seed=7)
    d = (cf(x).float() - f(x).float()).abs().max().item()
    ok &= d == 0
    print(f"torch.compile M={M}: maxdiff vs eager {d}")

# CUDA graph capture
x = make_act(4, w.shape[1], seed=9)
xq, xs = quant_per_token(x)
o = op(xq, w.t(), xs, s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    og = op(xq, w.t(), xs, s)
g.replay()
torch.cuda.synchronize()
d = (og.float() - o.float()).abs().max().item()
ok &= d == 0
print(f"cuda graph replay maxdiff {d}")
print("PASS" if ok else "FAIL")
