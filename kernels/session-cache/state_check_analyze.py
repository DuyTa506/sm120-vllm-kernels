"""Analyze the debug dumps of state_check.py. Runs inside the server container (needs torch):

  docker exec <name> python3 /w/state_check_analyze.py /tmp/ssdump /w/results/s8/cases.json

For every session snapshot (sequence position E, hash of tokens[:E]) it finds the cold-prefill reference
snapshots of the same case and reports the relative L2 distance ||a - b|| / ||b|| per GDN layer and state
tensor (conv, ssm) against: the reference at the same E with the same tokens (must be ~numerical noise),
references at E-1 and E+1 (what an off-by-one would look like), and two references of the same prompt
(prefill-vs-prefill noise floor). Prints JSON.
"""
import glob, json, os, statistics, sys

import torch

dump_dir, cases_path = sys.argv[1], sys.argv[2]
meta = [json.loads(l) for l in open(os.path.join(dump_dir, "meta.jsonl"))]
files = []
for f in glob.glob(os.path.join(dump_dir, "snap_*.pt")):
    _, rest = os.path.basename(f).split("_", 1)
    *blocks, tns = rest.removesuffix(".pt").split("_")
    files.append(([int(b) for b in blocks], int(tns) / 1e9, f))
files.sort(key=lambda x: x[1])
by_sid = {}
for m in meta:
    by_sid.setdefault(m["sid"], []).append(m)


def dump_for(m):
    for blocks, t, f in files:
        if blocks == m["blocks"] and t >= m["t"]:
            return torch.load(f)
    return None


NUM_SPEC = int(os.environ.get("NUM_SPEC", "2"))  # MTP-2: conv blocks carry 2 extra draft slots
CONV_SHAPES = set()


def rel(x, y):
    return float((x - y).norm() / y.norm().clamp_min(1e-12))


def dist(a, b):
    """conv: the valid window is the first (L - NUM_SPEC) positions of the slide axis (L = block
    length on that axis); the trailing NUM_SPEC slots are scratch for draft tokens and are not part
    of the state (the align copy leaves them stale). Also reported: whole block, and per position."""
    out = {}
    for layer in a:
        lid = layer.split(".")[-2] if "." in layer else layer
        for x, y in zip(a[layer], b[layer]):
            x, y = x.float(), y.float()
            if y.dim() >= 3:  # ssm [h, dk, dv]
                out[f"{lid}/ssm"] = rel(x, y)
                continue
            CONV_SHAPES.add(tuple(y.shape))
            ax = 0 if y.shape[0] < y.shape[1] else 1  # slide axis is the short one
            L = y.shape[ax]
            xs, ys = x.movedim(ax, 0), y.movedim(ax, 0)
            out[f"{lid}/conv"] = rel(xs[: L - NUM_SPEC], ys[: L - NUM_SPEC])
            out[f"{lid}/conv_full"] = rel(xs, ys)
            for p in range(L):
                out[f"{lid}/conv_pos{p}"] = rel(xs[p], ys[p])
    return out


cases = json.load(open(cases_path))["cases"]
rows = []
prev_n = {}
for c in cases:
    ms = [m for m in by_sid.get(c["sid"], []) if m["n"] == c["n"]]
    if not ms:
        rows.append({"session": c["session"], "turn": c["turn"], "error": "no session snapshot"})
        prev_n[c["session"]] = c["n"]
        continue
    m1 = ms[-1]
    a = dump_for(m1)
    refs = []
    for r in c["refs"]:
        for m2 in by_sid.get(r["sid"], []):
            refs.append((m2, dump_for(m2)))
    row = {"session": c["session"], "turn": c["turn"], "E": m1["E"], "n": m1["n"],
           "in_flight": m1["in_flight"], "use_accepted": m1["use_accepted"],
           "adopted_prev": c["turn"] > 0 and c["cached"] >= prev_n.get(c["session"], 1 << 30) - 1}
    same = [(m2, d) for m2, d in refs if m2["E"] == m1["E"] and m2["sha"] == m1["sha"] and d is not None]
    row["ref_E"] = sorted({m2["E"] for m2, _ in refs})
    if same and a is not None:
        row["same"] = dist(a, same[0][1])
        if len(same) > 1:
            row["ref_vs_ref"] = dist(same[1][1], same[0][1])
    for off, key in ((-1, "minus1"), (1, "plus1")):
        cand = [(m2, d) for m2, d in refs if m2["E"] == m1["E"] + off and d is not None]
        if cand and a is not None:
            row[key] = dist(a, cand[0][1])
    rows.append(row)
    prev_n[c["session"]] = c["n"]


def agg(key):
    vals = {}
    for r in rows:
        for k, v in (r.get(key) or {}).items():
            vals.setdefault(k, []).append(v)
    return {k: {"n": len(v), "median": statistics.median(v), "max": max(v), "min": min(v)}
            for k, v in sorted(vals.items())}


def separated(r, kind):
    """same-position distance below both off-by-one distances, for every dumped layer"""
    if not all(k in r for k in ("same", "minus1")):
        return None
    ok = True
    for k, v in r["same"].items():
        if k.endswith("/" + kind):
            for key in ("minus1", "plus1"):
                if key in r and k in r[key]:
                    ok &= v < r[key][k]
    return ok


summary = {"snapshots": len(rows), "with_same_position_reference": sum("same" in r for r in rows),
           "conv_block_shapes": sorted(CONV_SHAPES), "num_spec_assumed": NUM_SPEC,
           "paths": {"in_flight": sum(1 for r in rows if r.get("in_flight", 0) > 0),
                     "no_in_flight": sum(1 for r in rows if r.get("in_flight") == 0),
                     "after_adopted_hit": sum(1 for r in rows if r.get("adopted_prev"))},
           "separated_ssm": sum(1 for r in rows if separated(r, "ssm")),
           "separated_conv": sum(1 for r in rows if separated(r, "conv")),
           "same": agg("same"), "ref_vs_ref": agg("ref_vs_ref"), "minus1": agg("minus1"),
           "plus1": agg("plus1")}
print(json.dumps({"summary": summary, "rows": rows}, indent=1))
