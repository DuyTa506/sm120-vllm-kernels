"""Minimal native queue remap; installed FlashInfer remains untouched."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).parent
source=HERE/'gated_baseline.py'
s=source.read_text()
def once(old,new):
 global s
 assert s.count(old)==1,(old,s.count(old))
 s=s.replace(old,new,1)
once('class MoEGatedDynamicKernel:', 'class MoEGatedDynamicKernelLongFirst:')
once('__all__ = ["MoEGatedDynamicKernel"]', '__all__ = ["MoEGatedDynamicKernelLongFirst"]')
once('''        split_tile_count: Int32,
        expert_idx: Int32,''','''        split_tile_count: Int32,
        total_m_tiles: Int32,
        expert_idx: Int32,''')
once('''        extra_per_split = split_groups - Int32(1)
        split_tiles_before = m_tile_idx
        if split_tiles_before > split_tile_count:
            split_tiles_before = split_tile_count
        start = m_tile_idx + split_tiles_before * extra_per_split''','''        # Same descriptors, long unsplit tiles first, then short split tasks.
        start = m_tile_idx - split_tile_count
        if m_tile_idx < split_tile_count:
            start = (total_m_tiles - split_tile_count) + m_tile_idx * split_groups''')
once('''                            split_tile_count,
                            expert_flush,''','''                            split_tile_count,
                            total_m_tiles,
                            expert_flush,''')
target=HERE/'gated_long_first.py';target.write_text(s)
# Verify untouched methods structurally, including arithmetic, scales and scatter.
def methods(text):
 cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name.startswith('MoEGatedDynamicKernel'))
 return {n.name:ast.dump(n) for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
a,b=methods(source.read_text()),methods(s)
changed=[name for name in a if a[name]!=b[name]]
assert len(changed)==2 and 'publish_variable_deferred_tasks' in changed,changed
manifest=dict(baseline_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),changed_methods=changed,
 default='baseline',scope='device-only permutation of unchanged task descriptors; no routing/scale/MMA edits')
(HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
