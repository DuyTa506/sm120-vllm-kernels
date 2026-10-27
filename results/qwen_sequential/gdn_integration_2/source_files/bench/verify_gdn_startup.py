"""Fail a candidate run if GDN fusion silently falls back during loading."""
import argparse
import json
import re
from pathlib import Path


def verify(text):
    lines = [line for line in text.splitlines() if "GDN norm-FP8 ready for " in line]
    layers = [int(re.search(r"layers\.(\d+)\.linear_attn", line).group(1)) for line in lines]
    expected = [i for i in range(40) if (i + 1) % 4]
    disabled = [line for line in text.splitlines() if "GDN norm-FP8 disabled" in line]
    result = {"ready_layers": sorted(layers), "expected_layers": expected, "disabled": disabled,
              "scope": "Post-load adapter readiness; runtime/graph correctness is measured separately"}
    if sorted(layers) != expected or disabled:
        raise ValueError(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.log.read_text()), indent=2))
