"""Fail closed on partial diagnostic workloads even if clients exit zero."""
import json
import re
from pathlib import Path
import sys

def check_boundary(data, *, prompts=32, output_tokens=2304):
    assert (prompts,output_tokens) in ((32,2304),(8,6656)), 'unreviewed boundary workload'
    assert data.get('completed') == prompts, 'boundary requests incomplete'
    lengths=data.get('output_lens', [])
    assert len(lengths)==prompts and all(n==output_tokens for n in lengths), 'boundary output coverage incomplete'
    assert len(data.get('errors', []))==prompts and not any(data['errors']), 'boundary request errors'

def check_stress(text):
    matches=re.findall(r'stress: ok=(\d+) aborted=(\d+) failed=(\d+)',text)
    assert len(matches)==1, 'missing or ambiguous stress summary'
    ok,aborted,failed=map(int,matches[0])
    assert failed==0 and ok+aborted==128 and ok>=64 and aborted>=8, 'stress completion/abort coverage incomplete'

if __name__=='__main__':
    root=Path(sys.argv[1])
    config=json.loads((root/'boundary_config.json').read_text())
    check_boundary(json.loads((root/'serving/boundary.json').read_text()),prompts=config['prompts'],output_tokens=config['output_tokens'])
    check_stress((root/'stress.log').read_text())
    print(f"all{config['prompts']} boundary outputs={config['output_tokens']}; all128 stress requests accounted with completion and abort coverage")
