"""CPU-only paired output gate; single OFF/ON process timings remain preliminary."""
import hashlib,json,sys
from pathlib import Path
import torch
assert not torch.cuda.is_initialized()
a,b=map(Path,sys.argv[1:3]);reports=[json.loads((p/'measurement/report.json').read_text()) for p in [a,b]]
assert [r['mode'] for r in reports]==['off','on']
for r in reports:
 assert r['status']=='passed'
 assert [c['tokens'] for c in r['cases']]==[1024,4096]
 for c in r['cases']:
  assert [v['name'] for v in c['checks']]==['actual','changed_input_routes','sparse_tail','restored']
  assert c['trace_kernel_names'] and len(c['timing_us'])==3
for field in ['capture_sha256','checkpoint_identity','gate_rel_l2']:assert reports[0][field]==reports[1][field]
assert json.loads((a/'source_manifest.json').read_text())==json.loads((b/'source_manifest.json').read_text())
for p in [a,b]:assert (p/'exit_code.txt').read_text().strip()=='0'
outputs=[torch.load(p/'measurement/outputs.pt',map_location='cpu',weights_only=True) for p in [a,b]]
rows=[]
for ac,bc in zip(reports[0]['cases'],reports[1]['cases'],strict=True):
 assert ac['tokens']==bc['tokens'] and ac['tile_m']==bc['tile_m']==128
 for av,bv in zip(ac['checks'],bc['checks'],strict=True):
  assert av['name']==bv['name'] and av['input_hashes']==bv['input_hashes']
  k=f"{ac['tokens']}_{av['name']}";x,y=[o[k].float() for o in outputs]
  finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all());rel=float((x-y).norm()/x.norm().clamp_min(1e-12));assert finite and rel<=.02,(k,rel)
  rows.append(dict(case=k,rel_l2=rel,max_abs=float((x-y).abs().max()),finite=finite))
result=dict(status='diagnostic_pair_passed',quality_equivalence=False,checks=rows,timings=[dict(tokens=x['tokens'],off_us=[t['median'] for t in x['timing_us']],on_us=[t['median'] for t in y['timing_us']]) for x,y in zip(*[r['cases'] for r in reports])],scope='Separate startup single pair; no selection or speed attribution without repeat, artifact and numerical review')
print(json.dumps(result,indent=2));assert not torch.cuda.is_initialized()
