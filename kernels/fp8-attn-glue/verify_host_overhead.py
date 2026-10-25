"""Verifier: host (CPU) cost per attention call of the real patched vLLM function.

Imports the patch-03 vllm/v1/attention/backends/flashinfer.py (bind-mount it over the image
path, see VERIFY.md) and calls _sm120_fp8_paged_prefill with stand-in impl/layer/wrapper
objects, _noglue=True (patch 03) vs _noglue=False (patch 01 code path). Measures CPU wall
time per call while the GPU is kept busy (so launches never block), and GPU time per call.
Also checks the two outputs are bitwise identical.
"""

import os
import statistics
import time
from types import SimpleNamespace

os.environ["VLLM_SM120_FP8_PREFILL"] = "1"
os.environ["VLLM_SM120_FP8_PREFILL_NOGLUE"] = "1"

import torch  # noqa: E402

from common import D, HKV, HQ, PAGE, make_batch, make_pool, make_query  # noqa: E402
import vllm.v1.attention.backends.flashinfer as fb  # noqa: E402

assert fb._SM120_FP8_NOGLUE, "patch 03 flashinfer.py not mounted"
NB = 4096
raw, k, v = make_pool(NB)
busy = torch.empty(8192, 8192, device="cuda", dtype=torch.bfloat16)

for cname, seqs in (("512", [(512, 512)]), ("8k", [(8192, 8192)]),
                    ("varlen", [(2048, 2048), (1024, 5120), (3000, 3000), (2120, 2120)])):
    b = make_batch(seqs, NB, seed=7)
    q = make_query(b["total_q"])
    wrapper = SimpleNamespace(_kv_layout="HND", _paged_kv_indices_buf=b["indices"],
                              _paged_kv_indptr_buf=b["indptr"],
                              _paged_kv_last_page_len_buf=b["last"], _qo_indptr_buf=b["qo_indptr"],
                              _max_q_len=b["max_q"], _max_kv_len=b["max_kv"])
    impl = SimpleNamespace(scale=D ** -0.5)
    layer = SimpleNamespace(_q_scale_float=1.0, _k_scale_float=0.0234375, _v_scale_float=0.01953125)
    outs = {}
    res = {}
    for name, ng in (("patch01", False), ("patch03", True)):
        out = torch.empty(b["total_q"], HQ * D, dtype=torch.bfloat16, device="cuda")
        md = SimpleNamespace()  # fresh per-step metadata object (meta cached on it after call 1)
        fb._sm120_fp8_paged_prefill(impl, layer, md, wrapper, q, (k, v), out, _noglue=ng)
        torch.cuda.synchronize()
        outs[name] = out.clone()
        cpu, gpu = [], []
        for rep in range(7):
            # keep the GPU busy ~30 ms so the CPU never waits on the launch queue
            for _ in range(6):
                busy @ busy
            t0 = time.perf_counter()
            for _ in range(11):  # one forward's worth of attention calls
                fb._sm120_fp8_paged_prefill(impl, layer, md, wrapper, q, (k, v), out, _noglue=ng)
            t1 = time.perf_counter()
            torch.cuda.synchronize()
            cpu.append((t1 - t0) / 11 * 1e6)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(11):
                fb._sm120_fp8_paged_prefill(impl, layer, md, wrapper, q, (k, v), out, _noglue=ng)
            e.record()
            e.synchronize()
            gpu.append(s.elapsed_time(e) / 11 * 1e3)
        res[name] = (statistics.median(cpu), statistics.median(gpu))
    bad = int((outs["patch01"].view(torch.int16) != outs["patch03"].view(torch.int16)).sum())
    print(f"{cname:7s} host us/call: patch01 {res['patch01'][0]:7.1f}  patch03 {res['patch03'][0]:7.1f}   "
          f"GPU us/call (back-to-back): patch01 {res['patch01'][1]:8.1f}  patch03 {res['patch03'][1]:8.1f}   "
          f"mismatched={bad}")
