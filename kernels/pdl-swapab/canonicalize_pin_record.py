#!/usr/bin/env python3
"""Choose NEW controlled baseline configs from intersection of recorded candidates.
This does not claim to reproduce historical winners. Reject empty intersections.
"""
import argparse
import hashlib
import json
from pathlib import Path

def canonicalize(events):
    refs={};evidence={};torch_hashes=set();counts={}
    for event in events:
        if event['stage']=='installed':torch_hashes.add(event['torch_source_sha256'])
        if event['stage']!='constructor':continue
        key=event['identity'];body=event['identity_evidence']
        if key in evidence and evidence[key]!=body:raise ValueError('identity collision')
        evidence[key]=body
        options={json.dumps(c,sort_keys=True,separators=(',',':')) for c in event['available']}
        if len(options)!=len(event['available']) or not options:raise ValueError('empty/duplicate constructor candidates')
        refs[key]=refs.get(key,options)&options;counts[key]=counts.get(key,0)+1
    if len(torch_hashes)!=1 or not refs:raise ValueError('missing/ambiguous Torch source or constructors')
    empty=[k for k,v in refs.items() if not v]
    if empty:raise ValueError('no common valid config across constructors: '+','.join(empty))
    reference={k:[{'config':json.loads(sorted(v)[0]),'identity_evidence':evidence[k],
                  'constructor_count':counts[k],'intersection_candidates':[json.loads(x) for x in sorted(v)]}] for k,v in refs.items()}
    return {'status':'canonical_complete','torch_source_sha256':next(iter(torch_hashes)),
            'reference':reference,'selection_policy':'Lexicographically first canonical JSON config in intersection across all observed constructors; NEW baseline, not historical winner replay.'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('events',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run=a.events.parents[1]
    if (run/'exit_code.txt').read_text().strip()!='0':raise ValueError('refusing incomplete/failed reference run')
    for name in ('pdl_verify.json','source_verify.log','pin_events_verify.json'):
        if not json.loads((run/name).read_text())['status'].startswith('passed'):raise ValueError('reference gate failed: '+name)
    files=sorted(a.events.glob('events.*.jsonl'));events=[json.loads(line) for f in files for line in f.read_text().splitlines()]
    result=canonicalize(events);result['input_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'identities':len(result['reference'])}))
