"""Raw host<->device copy bandwidth (pinned host memory), the ceiling for KV restore. Runs inside the vLLM image.
Also times the block-granular pattern the offload worker uses: N scattered 2.18 MB pages (one attention layer page
of a 2128-token block, FP8, 2 KV heads x 256) per direction."""
import time, torch

dev = torch.device("cuda")
for size_mb in (64, 1024):
    n = size_mb << 20
    h = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    d = torch.empty(n, dtype=torch.uint8, device=dev)
    for direction in ("H2D", "D2H"):
        src, dst = (h, d) if direction == "H2D" else (d, h)
        for _ in range(3):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        t = time.perf_counter(); reps = 10
        for _ in range(reps):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / reps
        print(f"{direction} {size_mb:5d} MB contiguous: {n / dt / 1e9:6.1f} GB/s ({dt*1e3:.2f} ms)")

page = 2128 * 2 * 256 * 2  # bytes of one attention layer page per 2128-token block (K+V, FP8)
npages = 480               # 48 blocks x 10 layers ~ a 100k-token prefix of attention KV
hp = torch.empty(npages, page, dtype=torch.uint8, pin_memory=True)
dp = torch.empty(npages * 2, page, dtype=torch.uint8, device=dev)
idx = torch.randperm(npages * 2)[:npages].tolist()
for _ in range(2):
    for i, j in enumerate(idx):
        dp[j].copy_(hp[i], non_blocking=True)
torch.cuda.synchronize()
t = time.perf_counter()
for i, j in enumerate(idx):
    dp[j].copy_(hp[i], non_blocking=True)
torch.cuda.synchronize()
dt = time.perf_counter() - t
print(f"H2D {npages} scattered pages of {page/1e6:.2f} MB ({npages*page/1e9:.2f} GB): {npages*page/dt/1e9:.1f} GB/s, "
      f"{dt*1e3:.1f} ms")
