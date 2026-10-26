"""Aggregate pcie_correct.py results. usage: pc_summary.py <pc_*.json> [...]"""
import json, sys

for f in sys.argv[1:]:
    rs = json.load(open(f))
    n = len(rs)
    print(f"== {f}: {n} cases, cached cpu==gpu in {sum(r['cached']['cpu'] == r['cached']['gpu'] for r in rs)}/{n}, "
          f"GPU-cache reset ok {sum(r['reset_ok'] for r in rs)}/{n}")
    pairs = list(rs[0]["tv_top20"])
    for p in pairs:
        tv = [r["tv_top20"][p] for r in rs if r["tv_top20"][p] is not None]
        print(f"  {p:11s} mean TV {sum(tv)/len(tv):.3f} max {max(tv):.3f}  top1 agree "
              f"{sum(r['top1_agree'][p] for r in rs)}/{n}  32-token text equal {sum(r['text_equal'][p] for r in rs)}/{n}")
    print("  len(A)  cached  TTFT cold / gpu-hit / cpu-restore (ms)  load GB  load ms  GB/s")
    for r in rs:
        m = r["cpu_path_metrics"]
        print(f"  {r['lenA']:6d} {r['cached']['cpu']:7d}   {r['ttft_ms']['cold']:6.0f} / {r['ttft_ms']['gpu']:5.0f} / "
              f"{r['ttft_ms']['cpu']:5.0f}   {m['kv_offload_load_bytes_total']/1e9:6.2f} {1000*m['kv_offload_load_time_total']:7.1f}"
              f"  {r['load_GBps'] or 0:5.1f}")
