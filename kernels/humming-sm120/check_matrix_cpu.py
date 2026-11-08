"""Validate available real captures and emit, but never execute, RN screen cases."""
import hashlib,json
from pathlib import Path
import torch
p=Path('/work/kernels/moe-nvfp4/data/moe_inputs.pt')
assert p.stat().st_size>1000000
hidden=torch.load(p,map_location='cpu',weights_only=True)['hidden']
cases=[]
for layer in (19,0,39):
    assert hidden[layer].shape[1]==2048
    for tokens in (1024,4096):
        assert hidden[layer].shape[0]>=2*tokens
        tag=f'humming_rn_l{layer}_m{tokens}_1'
        cases.append(dict(layer=layer,tokens=tokens,capture_offsets=[0,tokens],command=f"sg docker -c 'bash kernels/humming-sm120/run_micro.sh {tag} indexed {tokens} screen_rn {layer}'"))
print(json.dumps(dict(cuda_initialized=torch.cuda.is_initialized(),capture_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),cases=cases,comparison_order=['scalar','cutlass','humming','humming','cutlass','scalar'],limitations='hot-layer screen; disjoint rows from same capture, not independent-prompt validation; each command needs GPU lease'),indent=2))
