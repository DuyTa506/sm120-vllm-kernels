"""Print the README tables from results/<label>/ (replays, end-to-end hot/cold, engine-level state check)."""
import glob, json, os

H = os.path.dirname(os.path.abspath(__file__))
R = f"{H}/results"
NAMES = {"P": "production", "P16": "production + unit 16", "O": "optimized", "O64": "optimized + unit 64"}


def load(label, name):
    f = f"{R}/{label}/{name}.json"
    return json.load(open(f)) if os.path.exists(f) else None


def replay_table(kind, part):
    print("| agents | config | prefilled / later turn | new | excess | TTFT p50 ms | p90 ms | mean ms | replay wall s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in ("1", "8"):
        for cfg in ("P", "P16", "O", "O64"):
            d = load(f"{cfg}_{part}", f"{kind}_c{c}")
            if not d:
                continue
            s = d[c]["summary"]
            print(f"| {c} | {NAMES[cfg]} | {s['later_prefilled_mean']:.0f} | {s['later_new_mean']:.0f} | "
                  f"{s['later_excess_mean']:.0f} | {s['later_ttft_p50_ms']:.0f} | {s['later_ttft_p90_ms']:.0f} | "
                  f"{s['later_ttft_mean_ms']:.0f} | {s['wall_s']:.0f} |")
    print()


print("## long (sessions 0,1,4,7)"); replay_table("long", "A")
print("## short (all 8)"); replay_table("short", "B")

print("## end-to-end hot/cold (38 turns; TV of first-token top-5 vs cold run 1)")
print("| config | cold2 TV mean | hot TV mean | drop16 TV mean | dup16 TV mean | top-1 same hot / cold2 | 32 greedy tokens same hot / cold2 |")
print("|---|---|---|---|---|---|---|")
for label in ("P_B", "P16_B", "O_A", "O64_A"):
    d = load(label, "hotcold")
    if not d:
        continue
    s = d["summary"]
    print(f"| {NAMES[label.split('_')[0]]} | {s['cold2']['tv_mean']:.3f} | {s['hot']['tv_mean']:.3f} | "
          f"{s['drop16']['tv_mean']:.3f} | {s['dup16']['tv_mean']:.3f} | {s['hot']['top1_same']} / "
          f"{s['cold2']['top1_same']} | {s['hot']['greedy32_same']} / {s['cold2']['greedy32_same']} |")
print()

print("## engine-level GDN state (layer 0; relative L2 unless noted)")
print("| config | mean hit / prev prompt | restored == producer state at T (bit-exact) | restored vs cold prefill of T tokens: mean / max | off-by-16 control: min / mean | hot final vs cold final: mean / max |")
print("|---|---|---|---|---|---|")
for label, cfg in (("Sv2", "P"), ("S16v2", "P16"), ("SO", "O"), ("SO64", "O64")):
    d = load(label, "state_check")
    if not d:
        continue
    s = d["summary"]
    w = s.get("L0_restored_vs_warm_maxabs")
    exact = f"{sum(1 for r in d['rows'] if r.get('L0_restored_vs_warm_maxabs') == 0.0)}/{w['n']} (others: no chunk ended at T)" if w else "n/a"
    rp, ctl, fin = s["L0_restored_vs_pre_rel"], s["L0_pre16_vs_pre_rel"], s["L0_hot_final_vs_cold_rel"]
    print(f"| {NAMES[cfg]} | {s['hit_mean']:.0f} / {s['prev_len_mean']:.0f} | {exact} | {rp['mean']:.1e} / {rp['max']:.1e} | "
          f"{ctl['min']:.3f} / {ctl['mean']:.3f} | {fin['mean']:.1e} / {fin['max']:.1e} |")
print()
for f in sorted(glob.glob(f"{R}/*/ckpt_dist.txt")):
    print(open(f).read().strip())
