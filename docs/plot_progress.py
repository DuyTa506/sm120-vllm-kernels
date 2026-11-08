"""Plot the measured optimized -> next -> scalar progression from saved results.

Usage: python docs/plot_progress.py
Requires matplotlib. Writes light/dark SVG, PNG, PDF and a source-bound JSON.
"""
from pathlib import Path
import hashlib
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/img"
LENGTHS = [2048, 4096, 5000, 8192, 16384, 32768]
COLORS = ["#0072B2", "#CC79A7", "#009E73"]
LABELS = ["optimized", "next (+ patches 03–05)", "scalar (next + b12x)"]


def load_data():
    sources = {}

    def read(path):
        raw = (ROOT / path).read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return raw.decode()

    old = read(Path("results/prefill/ttft_optimized.txt"))
    optimized = {int(n): float(v) for n, v in re.findall(
        r"prefill (\d+): .*?Median TTFT \(ms\): ([\d.]+)", old)}
    series = {"optimized": [optimized[n] for n in LENGTHS]}
    for name, tag in [("next", "next_matched_3"), ("scalar", "scalar_matched_3")]:
        values = []
        for n in LENGTHS:
            path = Path("results/qwen_sequential") / tag / "serving/serving_matrix_all" / f"prefill-{n}.measured.json"
            result = json.loads(read(path))
            assert result["completed"] == 8 and result["failed"] == 0
            assert result["input_lens"] == [n] * 8
            values.append(result["median_ttft_ms"])
        series[name] = values
    return dict(
        metric="Median cold single-request TTFT (ms), lower is better",
        lengths=LENGTHS, series=series, source_sha256=sources,
        scalar_change_vs_next_percent=[100 * (b / a - 1) for a, b in zip(series["next"], series["scalar"])],
        scope="Progress reference across measured configurations. optimized is historical; next/scalar are the matched_3 pair. Do not infer an isolated patch effect or combine this with kernel microbenchmarks.",
        limits="Eight measured requests per length for next/scalar; one run per configuration; fixed FlashInfer table, three differing Inductor selections. Scalar remains opt-in with documented quality and agentic limits.",
    )


def draw(data, dark):
    bg, fg, muted, grid = (("#171a20", "#edf0f5", "#b9c1ce", "#39414e") if dark else
                           ("#ffffff", "#20252d", "#566171", "#e2e7ed"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.1), gridspec_kw={"width_ratios": [1.65, 1]})
    fig.patch.set_facecolor(bg)
    fig.subplots_adjust(left=.065, right=.985, top=.72, bottom=.23, wspace=.23)
    fig.text(.065, .945, "Measured progress: optimized → next → scalar", color=fg, fontsize=19, weight="bold")
    fig.text(.065, .897, "Qwen3.6-35B-A3B NVFP4  ·  RTX PRO 6000 Blackwell  ·  vLLM 0.32", color=muted, fontsize=11)
    for ax, indices, ymax in zip(axes, [range(4), range(4, 6)], [255, 1080]):
        indices = list(indices)
        x = np.arange(len(indices))
        for j, (name, color, label) in enumerate(zip(data["series"], COLORS, LABELS)):
            values = [data["series"][name][i] for i in indices]
            bars = ax.bar(x + (j-1)*.24, values, .22, color=color, label=label,
                          hatch="//" if j == 0 else None, edgecolor=bg, linewidth=.6, zorder=3)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, value+ymax*.014, f"{value:.1f}",
                        ha="center", va="bottom", fontsize=8.5, color=fg)
        for k, i in enumerate(indices):
            top = max(data["series"][name][i] for name in data["series"])
            ax.text(k, top+ymax*.11, f"{data['scalar_change_vs_next_percent'][i]:.1f}% vs next",
                    ha="center", color=COLORS[2], fontsize=10, weight="bold")
        ax.set_facecolor(bg)
        ax.set_ylim(0, ymax)
        ax.set_xticks(x, [f"{LENGTHS[i]:,}" for i in indices])
        ax.set_xlabel("Prompt tokens", color=muted, labelpad=10)
        ax.set_ylabel("TTFT p50 (ms) ↓", color=muted)
        ax.tick_params(colors=muted, length=0, pad=7)
        ax.grid(axis="y", color=grid, linewidth=.7, zorder=0)
        for spine in ax.spines.values():
            spine.set_color(grid)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.057, .854), ncol=3,
               frameon=False, labelcolor=fg, fontsize=11, columnspacing=2.2)
    fig.text(.065, .115, "optimized: historical reference. next / scalar: matched_3 runs, 8 cold requests per prompt length.", color=muted, fontsize=9)
    fig.text(.065, .077, "Configuration-level comparison; fixed FI, 3 Inductor selections differ. Scalar is opt-in; round 2 candidates are not included.", color=muted, fontsize=9)
    mode = "dark" if dark else "light"
    stem = OUT / f"progress-prefill-{mode}"
    fig.savefig(stem.with_suffix(".svg"), facecolor=bg)
    fig.savefig(stem.with_suffix(".png"), facecolor=bg, dpi=300)
    if not dark:
        fig.savefig(stem.with_suffix(".pdf"), facecolor=bg)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    data = load_data()
    (OUT / "progress-prefill-data.json").write_text(json.dumps(data, indent=2)+"\n")
    draw(data, False)
    draw(data, True)
    print(json.dumps({"scalar_vs_next_percent": data["scalar_change_vs_next_percent"],
                      "outputs": "docs/img/progress-prefill-{light,dark}.{svg,png}; light.pdf; data.json"}))
