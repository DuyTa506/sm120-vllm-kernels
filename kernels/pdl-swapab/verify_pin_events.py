#!/usr/bin/env python3
"""CPU validation of guarded constructor and final-launch evidence."""
import argparse
import hashlib
import json
from pathlib import Path

def verify(root):
    files=sorted((root/'serving/pin_control').glob('events.*.jsonl'))
    if not files:raise ValueError('no pin subprocess events')
    events=[json.loads(line) for p in files for line in p.read_text().splitlines()]
    modes={v['mode'] for v in events}
    if len(modes)!=1:raise ValueError('mixed pin modes')
    mode,=modes
    installed=[e for e in events if e['stage']=='installed']
    guards={(e['torch_source_sha256'],e['hook_sha256'],e['core_sha256']) for e in installed}
    if len(guards)!=1:raise ValueError('missing/mixed hook/source guard')
    snapshot=json.loads((root/'pdl_source_manifest.json').read_text())
    torch_hash,hook_hash,core_hash=next(iter(guards))
    if hook_hash!=snapshot['files']['pin_inductor_hook.py'] or core_hash!=snapshot['files']['pin_inductor_core.py']:
        raise ValueError('event hook/core hashes differ from frozen source snapshot')
    if mode!=snapshot['pin_mode']:raise ValueError('event mode differs from requested snapshot')
    constructors={};launched={};compiled=set()
    for e in events:
        key=e.get('identity')
        if e['stage']=='constructor':constructors.setdefault(key,[]).append(e)
        if e['stage']=='first_launch':launched.setdefault(key,[]).append(e['config'])
        if e['stage']=='compiled':compiled.add(key)
    if not constructors or not launched:raise ValueError('missing constructed/launched kernel evidence')
    if not launched.keys()<=constructors.keys():raise ValueError('launch bypassed constructor')
    if mode=='pin':
        data=(root/'serving/pin_reference.json').read_bytes();manifest=json.loads(data)
        if manifest['status']!='canonical_complete':raise ValueError('incomplete manifest')
        refs=manifest['reference']
        for key,entries in constructors.items():
            if key not in refs or len(refs[key])!=1:raise ValueError('unknown/ambiguous constructor identity')
            expected=refs[key][0]['config']
            if any(e['available'].count(expected)!=1 for e in entries):raise ValueError('reference unavailable/ambiguous')
            if any(c!=expected for c in launched.get(key,[])):raise ValueError('launch selection mismatch')
        if not launched.keys()<=compiled:raise ValueError('launch missing compile check')
    return {'status':'passed_events_only','mode':mode,'constructors':len(constructors),'launched':len(launched),
            'constructor_identity_keys':sorted(constructors),'launched_configs':launched,
            'guards':[list(g) for g in guards],'input_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'limitation':'First-launch events only; pair must also match model-bound AOT identities and config/source contracts. No performance or numerical-equivalence assertion.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();x=verify(a.run);a.output.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({k:x[k] for k in ('status','mode','constructors','launched')}))
