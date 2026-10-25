"""Collect results/<label>/*.json into one comparison table (markdown)."""
import glob, json, os, statistics

H = os.path.dirname(os.path.abspath(__file__))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(int(len(xs) * q), len(xs) - 1)]


rows = []
for f in sorted(glob.glob(f"{H}/results/*/*_c*.json")):
    label, name = f.split("/")[-2], os.path.basename(f)[:-5]
    d = json.load(open(f))
    for c, v in d.items():
        if c == "args":
            continue
        s = v["summary"]; later = [r for r in v["turns"] if r["turn"] > 0]
        # by context bucket: TTFT p50 for turns whose prompt is above 64k tokens
        big = [1000 * r["ttft"] for r in later if r["prompt"] > 65536]
        rows.append((name.split("_")[0], int(c), label, s["later_prefilled_mean"], s["later_new_mean"],
                     s["later_excess_mean"], s["later_ttft_p50_ms"], s["later_ttft_p90_ms"], s["later_ttft_mean_ms"],
                     round(pct(big, .5), 1) if big else None, s["metric_hit_rate"], s["wall_s"]))
rows.sort()
print("| set | agents | config | prefilled/turn | new/turn | excess | TTFT p50 ms | p90 ms | mean ms | p50 >64k ctx | hit rate | wall s |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print("| " + " | ".join(str(x) for x in r) + " |")

print()
for f in sorted(glob.glob(f"{H}/results/*/hotcold.json")):
    print(json.dumps(json.load(open(f))["summary"]))
for f in sorted(glob.glob(f"{H}/results/*/ckpt_dist.txt")):
    print(open(f).read().strip())
for f in sorted(glob.glob(f"{H}/results/*/kvinfo.txt")):
    for l in open(f):
        if "KV cache size" in l or "Setting attention block" in l or "group sizes" in l:
            print(f.split("/")[-2], l.split("]")[-1].strip())
