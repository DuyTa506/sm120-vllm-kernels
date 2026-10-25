"""Compare e2e_greedy_noglue0.json vs e2e_greedy_noglue1.json (host python, no torch)."""
import json
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "results"
a = json.load(open(f"{d}/e2e_greedy_noglue0.json"))
b = json.load(open(f"{d}/e2e_greedy_noglue1.json"))
print("prompt tokens:", a["prompt_tokens"], b["prompt_tokens"])
print("generated text identical:", a["text"] == b["text"])
print("generated tokens identical:", a["tokens"] == b["tokens"])
dl = [abs(x - y) for x, y in zip(a["token_logprobs"], b["token_logprobs"]) if x is not None and y is not None]
print(f"gen token logprob max |diff|: {max(dl) if dl else 0:.3e} over {len(dl)} tokens")
pa, pb = a.get("prompt_logprobs", []), b.get("prompt_logprobs", [])
dp = [abs(x - y) for x, y in zip(pa, pb) if x is not None and y is not None]
print(f"prompt logprob max |diff|: {max(dp) if dp else 0:.3e} over {len(dp)} tokens;"
      f" exact-equal fraction {sum(1 for x in dp if x == 0) / max(len(dp), 1):.4f}")
import statistics
def cmp(x, y, label):
    dd = [abs(p - q) for p, q in zip(x, y) if p is not None and q is not None]
    if dd:
        print(f"{label}: max |diff| {max(dd):.3e}, mean {statistics.mean(dd):.3e}, "
              f"exact-equal fraction {sum(1 for t in dd if t == 0) / len(dd):.4f}")
for tag, r in (("noglue0", a), ("noglue1", b)):
    if "prompt_logprobs_repeat" in r:
        cmp(r["prompt_logprobs"], r["prompt_logprobs_repeat"],
            f"control: same server ({tag}), same prompt twice")
cmp(pa, pb, "noglue0 vs noglue1 (different server launches)")
print("text[0:200]:", repr(a["text"][:200]))
