"""Is the prefill FP8 GEMM power/clock limited?  Runs vLLM's cutlass_scaled_mm (gdn_qkvz, M=8192)
back-to-back for a few seconds while sampling SM clock / power via NVML, and reports achieved TFLOPS
next to the clock-scaled tensor peak (2045 FP8 FLOP/clk/SM x 188 SMs x f_sm)."""
import threading
import time

import pynvml
import torch

from vllm import _custom_ops as ops

M, N, K = 8192, 12288, 2048
a = (torch.randn(M, K, device="cuda") * 0.5).to(torch.float8_e4m3fn)
w = (torch.randn(N, K, device="cuda") * 0.5).to(torch.float8_e4m3fn)
sa = torch.rand(M, 1, device="cuda") + 0.5
sb = torch.rand(N, device="cuda") + 0.5

pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
samples, stop = [], False


def sampler():
    while not stop:
        samples.append((pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                        pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0))
        time.sleep(0.02)


for _ in range(20):
    ops.cutlass_scaled_mm(a, w.t(), sa, sb, torch.bfloat16)
torch.cuda.synchronize()
t = threading.Thread(target=sampler)
t.start()
n = 0
e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
e0.record()
t0 = time.time()
while time.time() - t0 < 4.0:
    for _ in range(50):
        ops.cutlass_scaled_mm(a, w.t(), sa, sb, torch.bfloat16)
    n += 50
    torch.cuda.synchronize()
e1.record()
e1.synchronize()
stop = True
t.join()
ms = e0.elapsed_time(e1)
tf = 2 * M * N * K * n / (ms * 1e-3) / 1e12
s = samples[len(samples) // 4:]
clk = sum(x[0] for x in s) / len(s)
pw = sum(x[1] for x in s) / len(s)
peak = 2045 * 188 * clk * 1e6 / 1e12
print(f"vLLM cutlass_scaled_mm {M}x{N}x{K}: {ms/n*1e3:.1f} us/call, {tf:.0f} TFLOPS; "
      f"mean SM clock {clk:.0f} MHz, power {pw:.0f} W; clock-scaled peak {peak:.0f} TFLOPS -> {100*tf/peak:.1f}%")
