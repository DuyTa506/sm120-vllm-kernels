"""Describe saved expert histograms; no GPU, timing prediction or route changes."""
import argparse
import hashlib
import json
from pathlib import Path


def analyze(path):
    source = json.loads(path.read_text())
    counts = source['row_counts']
    assert len(counts) == 256 and all(isinstance(c, int) and c >= 0 for c in counts)
    assert sum(counts) == source['tokens'] * 8
    rows = sum(counts)
    candidates = []
    for tile_m in (16, 32, 64, 128):
        tiles = sum((c + tile_m - 1) // tile_m for c in counts)
        padded = tiles * tile_m
        candidates.append(dict(tile_m=tile_m, expert_m_tiles=tiles,
                               padded_rows=padded, useful_row_fraction=rows / padded,
                               unused_rows=padded - rows))
    expected_tile = 16 if rows < 15 * 256 else 32 if rows < 48 * 256 else 64 if rows < 96 * 256 else 128
    return dict(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                layer=source['layer'], tokens=source['tokens'], routed_rows=rows,
                nonempty_experts=sum(c > 0 for c in counts), max_rows_per_expert=max(counts),
                observed_wrapper_tile_m=source['tile_m'], functional_selector_tile_m=expected_tile,
                candidates=candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    base = root / 'kernels/moe-task-order/results/task_order_geometry_1/measurement/geometry'
    report = dict(scope='Offline geometry of saved actual L19 routing; not a latency or DRAM-traffic model.',
                  limitation='Smaller M tiles reduce padding but increase tile count, scheduling and possible weight reloads. Native kernels and both full calls must be measured.',
                  cases=[analyze(base / f'applicability_l19_m{m}_real.json') for m in (2048, 3072, 4096)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
