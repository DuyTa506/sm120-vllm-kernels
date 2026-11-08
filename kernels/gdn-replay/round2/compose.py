"""CPU-only strict replay composition onto current next sources."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import argparse
from composition_guards import replace_once

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
K = HERE.parent
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=HERE / "generated")
out = parser.parse_args().output
if out.exists():
    raise SystemExit("generated composition exists; preserve it and use a new reviewed revision")
audit = json.loads((K / "results/round2_composition_audit/audit.json").read_text())
patch = K / "patch/04-gdn-state-replay.patch"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(patch) == audit["patch_sha256"]
sources = {}
for name, expected in audit["baseline_sha256"].items():
    source = ROOT / "build" / name if name.endswith("qwen_gdn_linear_attn.py") else K / "patch/a" / name
    assert sha(source) == expected, f"baseline changed: {source}"
    sources[name] = source
out.mkdir()
shutil.copytree(ROOT / "build/vllm", out / "vllm")
for name, source in sources.items():
    target = out / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
result = subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(patch)], cwd=out, text=True, capture_output=True)
(out / "patch.log").write_text(result.stdout + result.stderr)
assert result.returncode == 0, result.stdout + result.stderr
# Debug-only coverage extension: observe every row and the intermediate boundary
# slot in addition to the committed final slot. Default numerical paths unchanged.
p = out / "vllm/model_executor/layers/mamba/gdn/gdn_state_replay.py"
s = replace_once(p.read_text(), 'R = min(n, 8)  # keep memory bounded: compare the first rows only', 'R = min(n, int(os.environ.get("VLLM_GDN_REPLAY_SHADOW_ROWS", "8")))')
s = replace_once(s, 'def _shadow_compare(self, ns: torch.Tensor, n: int) -> None:', 'def _shadow_compare(self, ns: torch.Tensor, n: int, idx_mapping, newc, block) -> None:')
s = replace_once(s, '        first = None\n        for G,', '        first = None\n        boundary_rows = boundary_bad = 0\n        for G,')
needle = '            armed = int((layer._replay_marker[:R] == self.epoch).sum())'
addition = '''            if newc is not None and block > 0:
                nc = newc[idx_mapping[:R].long()].long()
                previous = nc - nu
                boundary = (nc // block) * block
                m = boundary - previous
                boundary_valid = valid & (m > 0) & (m < nu)
                boundary_idx = (m - 1).clamp(0, si.shape[1] - 1)
                boundary_slot = si.gather(1, boundary_idx[:, None])[:, 0]
                boundary_valid &= boundary_slot > 0
                observed = layer.kv_cache[1][boundary_slot.clamp(min=0)]
                expected = G[torch.arange(R, device=G.device), boundary_idx]
                mismatch = (observed.view(torch.int32) != expected.view(torch.int32)).flatten(1).any(1) & boundary_valid
                boundary_rows += int(boundary_valid.sum())
                boundary_bad += int(mismatch.sum())
'''
assert s.count(needle) == 1
s=replace_once(s, needle,addition+needle)
s=replace_once(s, '        self.sh_steps += 1', '''        logger.warning("GDN replay ROUND2 rows=%d bad=%d boundary_rows=%d boundary_bad=%d armed_bad=%d out_bad=%d",
                       tot_valid, tot_bad, boundary_rows, boundary_bad, self.sh_armed_bad, self.bad_out)
        if tot_bad or boundary_bad or self.sh_armed_bad or self.bad_out:
            raise RuntimeError("GDN replay round2 shadow mismatch")
        self.sh_steps += 1''')
s=replace_once(s, 'self._shadow_compare(ns, n)', 'self._shadow_compare(ns, n, idx_mapping, num_computed_post, mamba_block)')
s = replace_once(s, '        assert not torch.cuda.is_current_stream_capturing()\n        state = layer.kv_cache[1]', '        assert not torch.cuda.is_current_stream_capturing()\n        if _SHADOW_STARTUP_SUSPENDED:\n            stock()\n            return\n        state = layer.kv_cache[1]')
s = replace_once(s, '_REGISTRY: GdnStateReplay | None = None', '_SHADOW_STARTUP_SUSPENDED = False  # diagnostic only; scoped by startup plugin\n_REGISTRY: GdnStateReplay | None = None')
p.write_text(s)
serve = (ROOT / "scripts/serve.sh").read_text()
serve = replace_once(serve, 'D=$(cd "$(dirname "$0")/.." && pwd)', 'D=$(cd "$(dirname "$0")/../../../.." && pwd)')
serve = replace_once(serve, 'B=$D/build/vllm', 'B=$D/kernels/gdn-replay/round2/generated/vllm')
(out / "serve.sh").write_text(serve)
manifest = dict(baseline_sha256=audit["baseline_sha256"], patch_sha256=sha(patch),
                generated_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file()},
                debug_extension="all-row shadow plus intermediate-boundary slot equality; authoritative stock path")
(out / "composition.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
