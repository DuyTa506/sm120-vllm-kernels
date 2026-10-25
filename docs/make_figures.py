"""Render the README figures (charts from results/, mechanism diagrams) as static SVG, light and dark.
usage: python3 docs/make_figures.py      -> docs/img/<name>-{light,dark}.svg
Charts whose input files are missing are skipped."""
import csv, glob, json, math, os, re
from xml.sax.saxutils import escape

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(D, "docs", "img")
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", box="#f0efec", s=["#2a78d6", "#eb6834", "#1baf7a", "#7c5ccf"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", box="#262624", s=["#3987e5", "#d95926", "#199e70", "#9479e0"]),
}
# colour follows the entity in every chart
SLOT = {"optimized": 0, "tuned": 1, "production": 2}
LABEL = {"production": "production", "tuned": "tuned (config)", "optimized": "optimized (config + patches)"}


class Svg:
    def __init__(self, w, h, t, title=None, subtitle=None):
        self.w, self.h, self.t, self.el = w, h, t, []
        if title:
            self.text(20, 28, title, size=15, weight=600)
        if subtitle:
            self.text(20, 48, subtitle, size=12, color=t["ink2"])

    def text(self, x, y, s, size=12, color=None, anchor="start", weight=400):
        self.el.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
                       f'fill="{color or self.t["ink"]}" text-anchor="{anchor}">{escape(str(s))}</text>')

    def line(self, x1, y1, x2, y2, color, width=1, dash=None, arrow=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#arrow-{color[1:]})"' if arrow else ""
        self.el.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
                       f'stroke-width="{width}"{d}{m} stroke-linecap="round"/>')
        if arrow:
            self.markers.add(color)

    markers = set()

    def rect(self, x, y, w, h, fill, stroke=None, r=6, dash=None):
        s = f' stroke="{stroke}" stroke-width="1.5"' if stroke else ""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.el.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}"{s}{d}/>')

    def hbar(self, x0, y, x1, h, color):
        """horizontal bar, square at the baseline, 4px rounded data end"""
        r = min(4, (x1 - x0) / 2, h / 2)
        self.el.append(f'<path d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} '
                       f'V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z" fill="{color}"/>')

    def poly(self, pts, color):
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.el.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="2" '
                       f'stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in pts:
            self.el.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="{self.t["surface"]}" stroke-width="2"/>')

    def legend(self, x, y, items):
        for name, color in items:
            self.rect(x, y - 9, 10, 10, color, r=2)
            self.text(x + 15, y, name, color=self.t["ink2"])
            x += 24 + 7 * len(name)

    def save(self, name, label):
        defs = "".join(f'<marker id="arrow-{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                       f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
                       for c in sorted(self.markers))
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
               f'height="{self.h}" role="img" aria-label="{escape(label)}" font-family="{FONT}">'
               f'<defs>{defs}</defs><rect width="100%" height="100%" fill="{self.t["surface"]}"/>'
               + "".join(self.el) + "</svg>\n")
        mode = "light" if self.t is THEMES["light"] else "dark"
        open(os.path.join(OUT, f"{name}-{mode}.svg"), "w").write(svg)
        Svg.markers = set()


def nice_max(v):
    e = 10 ** math.floor(math.log10(v))
    return next(m * e for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10) if m * e >= v)


# ---------------------------------------------------------------- data
def span(name):
    p = os.path.join(D, "results", "nsys", f"span_8k_{name}.txt")
    m = re.search(r"overall: span ([\d.]+) ms", open(p).read())
    return float(m.group(1)) / 4  # 4 requests per capture


def ttft(name):
    rows = {}
    for line in open(os.path.join(D, "results", "prefill", f"ttft_{name}.txt")):
        m = re.match(r"prefill (\d+): .*Median TTFT \(ms\): ([\d.]+)", line)
        if m:
            rows[int(m.group(1))] = float(m.group(2))
    return rows


def sweep(cfg):
    return {r["scenario"]: r for r in csv.DictReader(open(os.path.join(D, "results", "throughput", f"{cfg}.tsv")), delimiter="\t")}


# ---------------------------------------------------------------- charts
def chart_span(t):
    steps = [("production: vLLM defaults", span("production")),
             ("tuned: CUDA graphs to 16k tokens", span("tuned")),
             ("+ FP8 attention prefill (patch 01)", span("fp8attn")),
             ("+ GDN output in place (patch 02)", span("fp8attn_gdn_inplace")),
             ("+ in-forward checkpoint (patch 02)", span("optimized"))]
    s = Svg(720, 76 + 44 * len(steps) + 30, t, "8k-token prefill, GPU time per request",
            "nsys GPU span, single request, lower is better")
    x0, x1, vmax = 270, 650, nice_max(max(v for _, v in steps))
    X = lambda v: x0 + (x1 - x0) * v / vmax
    top = 72
    for k in range(0, 5):
        v = vmax * k / 4
        s.line(X(v), top - 6, X(v), top + 44 * len(steps) - 10, t["grid"])
        s.text(X(v), top + 44 * len(steps) + 6, f"{v:.0f} ms", size=11, color=t["muted"], anchor="middle")
    for i, (name, v) in enumerate(steps):
        y = top + 44 * i
        s.text(x0 - 12, y + 16, name, anchor="end", color=t["ink2"])
        s.hbar(x0, y + 2, X(v), 22, t["s"][0])
        s.text(X(v) + 8, y + 18, f"{v:.0f} ms", weight=600)
    s.line(x0, top - 6, x0, top + 44 * len(steps) - 10, t["axis"])
    s.save("prefill-span", "8k prefill GPU time per request: 369 ms with vLLM defaults, 163 ms tuned, 133 ms optimized")


def chart_throughput(t):
    prod, cfgs = sweep("production"), ["tuned", "optimized"]
    data = {c: sweep(c) for c in cfgs}
    names = list(prod)
    rowh = 58
    s = Svg(720, 96 + rowh * len(names) + 24, t, "Serving throughput vs production",
            "total tokens/s relative to production (1.0x), vllm bench serve, random prompts")
    s.legend(20, 74, [(LABEL[c], t["s"][SLOT[c]]) for c in cfgs])
    x0, x1, vmax = 230, 650, 1.75
    X = lambda v: x0 + (x1 - x0) * v / vmax
    top = 92
    for v in (0, 0.5, 1.0, 1.5):
        s.line(X(v), top - 4, X(v), top + rowh * len(names) - 8, t["axis"] if v == 1.0 else t["grid"])
        s.text(X(v), top + rowh * len(names) + 8, f"{v:.1f}x", size=11, color=t["muted"], anchor="middle")
    for i, sc in enumerate(names):
        y = top + rowh * i
        r = prod[sc]
        s.text(x0 - 12, y + 18, sc, anchor="end")
        s.text(x0 - 12, y + 34, f"{r['in']} in / {r['out']} out, {r['conc']} conc", anchor="end", size=11, color=t["muted"])
        for j, c in enumerate(cfgs):
            v = float(data[c][sc]["total_tok_s"]) / float(r["total_tok_s"])
            yy = y + 4 + j * 24
            s.hbar(x0, yy, X(v), 20, t["s"][SLOT[c]])
            s.text(X(v) + 8, yy + 15, f"{(v - 1) * 100:+.0f}%", size=12, color=t["ink2"])
    s.save("throughput", "Serving throughput relative to production for seven workload shapes, tuned and optimized configs")


def line_chart(t, name, title, subtitle, series, xs, xlabel, ylabel, label, logx=True, ymax=None):
    s = Svg(720, 380, t, title, subtitle)
    s.legend(20, 74, [(LABEL[c], t["s"][SLOT[c]]) for c in series])
    x0, x1, y0, y1 = 80, 640, 330, 116
    vmax = ymax or 4 * nice_max(max(v for d in series.values() for v in d.values()) / 4)  # round quarter ticks
    f = (lambda v: math.log(v)) if logx else (lambda v: v)
    X = lambda v: x0 + (x1 - x0) * (f(v) - f(xs[0])) / (f(xs[-1]) - f(xs[0]))
    Y = lambda v: y0 - (y0 - y1) * v / vmax
    for k in range(5):
        v = vmax * k / 4
        s.line(x0, Y(v), x1, Y(v), t["axis"] if k == 0 else t["grid"])
        s.text(x0 - 8, Y(v) + 4, f"{v:,.0f}", size=11, color=t["muted"], anchor="end")
    for v in xs:
        s.text(X(v), y0 + 18, f"{round(v / 1024)}k" if v >= 1024 else str(v), size=11, color=t["muted"], anchor="middle")
    s.text((x0 + x1) / 2, y0 + 38, xlabel, size=11, color=t["ink2"], anchor="middle")
    s.text(x0 - 8, y1 - 14, ylabel, size=11, color=t["ink2"], anchor="end")
    ends = []
    for c, d in series.items():
        pts = [(X(x), Y(d[x])) for x in xs if x in d]
        s.poly(pts, t["s"][SLOT[c]])
        ends.append((SLOT[c], c, pts[-1], d[xs[-1]]))
    # end labels: when two collide, keep the higher-priority series (optimized, then tuned, then production);
    # the legend and the results tables carry the dropped value
    placed = []
    for _, c, (x, y), v in sorted(ends):
        if all(abs(y - py) >= 15 for py in placed):
            s.text(x + 10, y + 4, f"{v:,.0f}", size=12, color=t["ink2"])
            placed.append(y)
    s.save(name, label)


def chart_ttft(t):
    series = {"tuned": ttft("tuned"), "optimized": ttft("optimized")}
    xs = sorted(series["tuned"])
    line_chart(t, "ttft", "Single-request TTFT by prompt length", "median of 8 requests, lower is better",
               series, xs, "prompt tokens", "ms", "TTFT by prompt length: optimized is 12-17% lower than tuned from 5k tokens up")


def chart_agentic(t, kind):
    series_ttft, series_tok = {}, {}
    for c in ("production", "tuned", "optimized"):
        p = os.path.join(D, "results", "agentic", f"replay_{kind}_{c}.json")
        if not os.path.exists(p):
            continue
        rep = json.load(open(p))
        series_ttft[c] = {int(k): v["summary"]["ttft_later_p50_ms"] for k, v in rep.items()}
        series_tok[c] = {int(k): v["summary"]["out_tok_s"] for k, v in rep.items()}
    if not series_ttft:
        return
    xs = sorted({x for d in series_ttft.values() for x in d})
    what = "coding sessions, 10-30k tokens" if kind == "short" else "whole-package reviews, up to 131k tokens"
    line_chart(t, f"agentic-{kind}-ttft", f"Agentic workload: per-turn TTFT ({what})",
               "median TTFT of turns after the first, token-exact replay of recorded tool-calling sessions",
               series_ttft, xs, "concurrent agents", "ms", f"Agentic {kind} sessions: per-turn TTFT by number of agents")
    line_chart(t, f"agentic-{kind}-throughput", f"Agentic workload: output tokens/s ({what})",
               "all agents together, token-exact replay of recorded tool-calling sessions",
               series_tok, xs, "concurrent agents", "tok/s", f"Agentic {kind} sessions: output throughput by number of agents")


def cache_rows():
    """before/after pairs from kernels/cache-granularity and kernels/session-cache (same server, same turns)"""
    K = os.path.join(D, "kernels")
    def cg(run, f):
        s = json.load(open(os.path.join(K, "cache-granularity", "results", run, f)))
        s = next(v["summary"] for k, v in s.items() if k != "args")
        return s["later_prefilled_mean"], s["later_ttft_p50_ms"]
    def sc(f):
        s = json.load(open(os.path.join(K, "session-cache", "results", "s1b", f)))["summary"]
        return s["prefill_mean"], s["ttft_p50_ms"]
    return [("--prefix-match-unit 16", "long sessions, 1 agent", cg("P_A", "long_c1.json"), cg("P16_A", "long_c1.json")),
            ("--prefix-match-unit 16", "long sessions, 8 agents", cg("P_A", "long_c8.json"), cg("P16_A", "long_c8.json")),
            ("session state cache", "long sessions, 1 agent", sc("long_c1_nosess.json"), sc("long_c1_sess.json")),
            ("session state cache", "short sessions, 1 agent", sc("short_con_nat_nosess.json"), sc("short_con_nat_sess.json"))]


def chart_cache(t):
    rows = cache_rows()
    rowh = 62
    s = Svg(860, 112 + rowh * len(rows) + 24, t, "Agentic follow-up turns: caching the GDN state more finely",
            "same server and turns per pair; interim numbers, independent verification pending")
    before, after = t["muted"], t["s"][3]
    s.legend(20, 74, [("stock: GDN state cached every 2128 tokens", before), ("finer cache", after)])
    panels = [(250, 520, "tokens prefilled per turn", 0), (600, 820, "TTFT p50, ms", 1)]
    top = 112
    for x0, x1, head, k in panels:
        vmax = nice_max(max(max(r[2][k], r[3][k]) for r in rows))
        X = lambda v, x0=x0, x1=x1, vmax=vmax: x0 + (x1 - x0) * v / vmax
        s.text(x0, top - 8, head, size=11, color=t["ink2"])
        s.line(x0, top, x0, top + rowh * len(rows) - 8, t["axis"])
        s.text(X(vmax), top + rowh * len(rows) + 8, f"{vmax:,.0f}", size=11, color=t["muted"], anchor="end")
        for i, (_, _, b, a) in enumerate(rows):
            y = top + rowh * i + 6
            for j, (v, c) in enumerate(((b[k], before), (a[k], after))):
                s.hbar(x0, y + j * 24, X(v), 20, c)
                s.text(X(v) + 6, y + j * 24 + 15, f"{v:,.0f}", size=11, color=t["ink2"])
    for i, (name, sub, _, _) in enumerate(rows):
        y = top + rowh * i
        s.text(230, y + 24, name, anchor="end")
        s.text(230, y + 40, sub, anchor="end", size=11, color=t["muted"])
    s.save("agentic-cache", "Finer GDN state caching cuts tokens prefilled per agent turn by 55-71% and per-turn TTFT by 20-56%")


# ---------------------------------------------------------------- mechanism diagrams
def diagram_agentic_cache(t):
    s = Svg(820, 330, t, "Why each agent turn re-prefills old tokens",
            "turn N+1 prompt = turn N prompt + model output + tool result; the GDN state can only resume where it was saved")
    x0, x1 = 210, 800
    seg = [("cached", 0.58), ("old tail", 0.22), ("new", 0.20)]
    rows = [("stock", "state saved every 2128 tokens", (0.58, 0.22)),
            ("--prefix-match-unit 16", "state saved every 16 tokens", (0.795, 0.005)),
            ("session state cache", "state kept at the end of turn N", (0.80, 0.0))]
    for i, (name, sub, (c, waste)) in enumerate(rows):
        y = 84 + i * 76
        s.text(20, y + 16, name, weight=600)
        s.text(20, y + 32, sub, size=11, color=t["ink2"])
        w = x1 - x0
        s.rect(x0, y, w * c, 30, t["box"], stroke=t["muted"], r=4)
        s.text(x0 + w * c / 2, y + 20, "prefix from cache", anchor="middle", size=11, color=t["ink2"])
        if waste > 0.01:
            s.rect(x0 + w * c, y, w * waste, 30, t["s"][1], r=4)
            s.text(x0 + w * (c + waste / 2), y + 20, "recomputed old tokens", anchor="middle", size=11, color=t["surface"])
        s.rect(x0 + w * (c + waste), y, w * (1 - c - waste), 30, t["s"][3], r=4)
        s.text(x0 + w * (c + waste + (1 - c - waste) / 2), y + 20, "new tokens", anchor="middle", size=11, color=t["surface"])
        if i == 0:
            s.line(x0 + w * c, y - 6, x0 + w * c, y + 38, t["ink2"], dash="3 3")
            s.text(x0 + w * c, y + 52, "last 2128-token boundary", anchor="middle", size=11, color=t["muted"])
    s.text(20, 312, "measured on long sessions: ~1,740 new tokens per turn, stock prefills ~4,760; unit 16 prefills ~1,765",
           size=11, color=t["ink2"])
    s.save("mech-agentic-cache", "With block-aligned GDN state caching every turn recomputes the old tail; a finer unit or a per-session state removes it")



def node(s, x, y, w, h, title, sub=None, hot=False, ghost=False):
    t = s.t
    s.rect(x, y, w, h, t["surface"] if ghost else t["box"], stroke=t["s"][0] if hot else (t["muted"] if ghost else t["ink2"]),
           dash="5 4" if ghost else None)
    s.text(x + w / 2, y + (h / 2 if not sub else h / 2 - 4) + 4, title, anchor="middle", size=12,
           color=t["muted"] if ghost else t["ink"], weight=600)
    if sub:
        s.text(x + w / 2, y + h / 2 + 14, sub, anchor="middle", size=11, color=t["muted"] if ghost else t["ink2"])


def arrow(s, x1, y1, x2, y2, label=None, color=None, dash=None, above=True):
    c = color or s.t["ink2"]
    s.line(x1, y1, x2, y2, c, width=1.5, dash=dash, arrow=True)
    if label:
        s.text((x1 + x2) / 2, (y1 + y2) / 2 + (-7 if above else 15), label, anchor="middle", size=11, color=s.t["ink2"])


def diagram_graphs(t):
    s = Svg(720, 250, t, "Tuned config: one CUDA graph replay instead of eager launches",
            "8k-token prefill, kernels on the GPU timeline (schematic)")
    for row, (name, busy, gaps, note) in enumerate([
            ("production", 0.40, True, "eager: CPU launches each kernel, GPU idles between them (40% busy), 369 ms"),
            ("tuned", 0.94, False, "graph replay up to 16k tokens: kernels back to back (94% busy), 163 ms")]):
        y = 80 + row * 80
        s.text(20, y + 16, name, weight=600)
        x, end = 120, 120 + (590 if row == 0 else 590 * 163 / 369)
        k = 0
        while x < end - 4:
            w = 9 + (k * 7) % 13
            if x + w > end:
                break
            s.rect(x, y, w, 24, t["s"][0] if row else t["s"][1], r=2)
            x += w + (9 + (k * 5) % 9 if gaps else 2)
            k += 1
        s.text(120, y + 44, note, size=11, color=t["ink2"])
    s.save("mech-cuda-graphs", "Production launches kernels eagerly with idle gaps; tuned replays one CUDA graph with kernels back to back")


def diagram_split(t):
    s = Svg(820, 330, t, "Patch 02: one forward pass per long prompt",
            "Mamba prefix caching needs the GDN state at the last block boundary (8192 tokens -> boundary at 4256)")
    # stock
    s.text(20, 84, "stock vLLM", weight=600)
    node(s, 20, 96, 140, 50, "8192-token prompt")
    arrow(s, 160, 121, 240, 121, "scheduler")
    node(s, 240, 96, 160, 50, "forward pass 1", "tokens 0-4255, 40 layers")
    arrow(s, 400, 121, 460, 121, "state")
    node(s, 460, 96, 140, 50, "save checkpoint", "GDN + conv state")
    arrow(s, 600, 121, 640, 121)
    node(s, 640, 96, 160, 50, "forward pass 2", "tokens 4256-8191")
    s.text(20, 170, "every weight is read twice and per-pass overheads are paid twice; GEMMs run at half the token count",
           size=11, color=t["ink2"])
    # patched
    s.text(20, 214, "with patch 02", weight=600)
    node(s, 20, 226, 140, 50, "8192-token prompt")
    arrow(s, 160, 251, 240, 251, "scheduler")
    node(s, 240, 226, 280, 50, "one forward pass", "FlashInfer GDN writes state every 64 tokens", hot=True)
    arrow(s, 520, 251, 600, 251, "pick 4256")
    node(s, 600, 226, 200, 50, "export checkpoint", "into the prefix-cache block", hot=True)
    s.text(20, 300, "same cached state (kernel output bit-identical with and without checkpoints); 8k span 145 -> 133 ms",
           size=11, color=t["ink2"])
    s.save("mech-checkpoint", "Stock vLLM splits a long prompt into two forward passes to save the GDN state; patch 02 runs one pass and exports the state from inside it")


def diagram_fp8attn(t):
    s = Svg(820, 420, t, "Patch 01: full-attention prefill on FP8 tensor cores",
            "10 attention layers, FP8 KV cache; prefills with at least 256 query tokens")
    s.text(20, 84, "stock (FA2)", weight=600)
    node(s, 20, 96, 130, 44, "Q (bf16)")
    node(s, 20, 160, 130, 44, "KV cache (fp8)")
    arrow(s, 150, 118, 330, 136)
    arrow(s, 150, 182, 330, 156, "dequant to bf16", above=False)
    node(s, 330, 120, 230, 52, "FA2 kernel", "bf16 MMA, ~550 TFLOPS peak")
    s.text(580, 150, "8k prefill span 163 ms", size=11, color=t["ink2"])
    s.text(20, 250, "patch 01", weight=600)
    node(s, 20, 262, 130, 44, "Q (bf16)")
    arrow(s, 150, 284, 250, 284, "quantize")
    node(s, 250, 262, 130, 44, "Q (fp8)", hot=True)
    node(s, 20, 326, 130, 44, "KV cache (fp8)")
    arrow(s, 150, 348, 250, 348, "gather pages")
    node(s, 250, 326, 130, 44, "K, V compact", ghost=True)
    arrow(s, 380, 284, 480, 306)
    arrow(s, 380, 348, 480, 326)
    node(s, 480, 290, 240, 52, "SM120 FP8 FMHA", "fp8 MMA, ~1100 TFLOPS peak", hot=True)
    s.text(480, 364, "8k prefill span 151 ms", size=11, color=t["ink2"])
    s.text(480, 382, "dashed step: removed by kernels/fp8-attn-glue (-0.9 ms)", size=11, color=t["muted"])
    s.save("mech-fp8-attention", "Stock FA2 dequantizes the FP8 KV cache to bf16; patch 01 quantizes Q to FP8 and runs the SM120 FP8 FMHA")


def diagram_gdn_fusion(t):
    s = Svg(760, 300, t, "kernels/gdn-conv-fusion: one pass over HBM instead of two",
            "GDN prefill front end, per layer, 8192 tokens x 8192 channels bf16 (128 MB)")
    s.text(20, 84, "before", weight=600)
    node(s, 20, 96, 120, 46, "mixed_qkv", "HBM")
    arrow(s, 140, 119, 190, 119, "read")
    node(s, 190, 96, 130, 46, "causal conv1d", "197 us")
    arrow(s, 320, 119, 370, 119, "write")
    node(s, 370, 96, 110, 46, "conv out", "HBM, 128 MB", ghost=True)
    arrow(s, 480, 119, 530, 119, "read")
    node(s, 530, 96, 130, 46, "post-conv prep", "split, l2norm, g, beta")
    arrow(s, 660, 119, 700, 119)
    s.text(704, 123, "q k v", size=11, color=t["ink2"])
    s.text(20, 166, "377 us per layer-call; the conv output makes a full round trip through HBM", size=11, color=t["ink2"])
    s.text(20, 210, "fused", weight=600)
    node(s, 20, 222, 120, 46, "mixed_qkv", "HBM")
    arrow(s, 140, 245, 220, 245, "read once")
    node(s, 220, 222, 440, 46, "fused conv1d + post-conv", "conv state and prefix-cache bookkeeping unchanged", hot=True)
    arrow(s, 660, 245, 700, 245)
    s.text(704, 249, "q k v", size=11, color=t["ink2"])
    s.text(20, 290, "190 us (1.43 TB/s, at the copy floor); x30 layers saves ~5.6 ms per 8k prefill", size=11, color=t["ink2"])
    s.save("mech-gdn-fusion", "The fused kernel reads mixed_qkv once and writes q, k, v, removing the HBM round trip of the conv output")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for t in THEMES.values():
        for fn in (chart_span, chart_throughput, chart_ttft, lambda t: chart_agentic(t, "short"),
                   lambda t: chart_agentic(t, "long"), chart_cache, diagram_graphs, diagram_split, diagram_fp8attn,
                   diagram_gdn_fusion, diagram_agentic_cache):
            try:
                fn(t)
            except FileNotFoundError as ex:
                print("skip:", ex.filename)
    print("\n".join(sorted(os.listdir(OUT))))
