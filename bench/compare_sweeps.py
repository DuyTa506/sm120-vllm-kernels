"""Markdown table of throughput sweeps, relative to the first config.
usage: compare_sweeps.py results/throughput/production.tsv results/throughput/tuned.tsv ..."""
import csv, os, sys

runs = [(os.path.basename(p).rsplit(".", 1)[0], {r["scenario"]: r for r in csv.DictReader(open(p), delimiter="\t")})
        for p in sys.argv[1:]]
base_name, base = runs[0]
cols = [("total_tok_s", "total tok/s"), ("out_tok_s", "output tok/s"),
        ("med_ttft_ms", "median TTFT ms"), ("med_tpot_ms", "median TPOT ms")]
for key, title in cols:
    print(f"\n**{title}**\n")
    print("| scenario | in / out / conc | " + " | ".join(n for n, _ in runs) + " |")
    print("|---|---|" + "---:|" * len(runs))
    for sc, b in base.items():
        cells = []
        for name, run in runs:
            v = float(run[sc][key]) if sc in run else float("nan")
            ref = float(b[key])
            cells.append(f"{v:.1f}" if name == base_name else f"{v:.1f} ({(v / ref - 1) * 100:+.0f}%)")
        print(f"| {sc} | {b['in']} / {b['out']} / {b['conc']} | " + " | ".join(cells) + " |")
