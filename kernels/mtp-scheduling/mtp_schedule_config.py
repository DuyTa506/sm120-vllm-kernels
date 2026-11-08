"""Exact installed vLLM dynamic-SD knob, maximum depth remains two."""
import json
import sys


def config(variant):
    if variant not in ("fixed2", "batch21"):
        raise ValueError("variant must be fixed2 or batch21; depth0 is not NOSPEC")
    result = dict(method="mtp", num_speculative_tokens=2)
    if variant == "batch21":
        result["num_speculative_tokens_per_batch_size"] = [[1, 8, 2], [9, 64, 1]]
    return result


if __name__ == "__main__":
    print(json.dumps(config(sys.argv[1])))
