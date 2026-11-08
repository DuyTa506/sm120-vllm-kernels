"""CPU fixed-order reference of the exact BF16 contributions sent to atomic add."""
import argparse,json
from pathlib import Path
import torch

def canonical(d):
    width=d['payload'].shape[-1]
    assert width in (128,2048) and d['output'].shape[1]==width
    p=d['payload'].reshape(-1,width);meta=d['metadata'].reshape(-1,4)
    valid=meta[:,0]>=0
    assert (meta[~valid]==-1).all(), 'partial metadata write'
    assert torch.isnan(p[~valid]).all(), 'invalid row unexpectedly written'
    assert torch.isfinite(p[valid]).all(), 'missing column or nonfinite contribution'
    counts=d['counts'].tolist();split=d['split'];base=0;groups={}
    for expert,count in enumerate(counts):
        nt=(count+127)//128
        assert not base<split<base+nt, 'expert straddles split boundary; per-token split contract must be derived from Q0 map'
        groups[expert]=[(0,2),(2,2)] if base<split else [(0,4)]
        base+=nt
    expected={(t,int(e),begin,count) for t,row in enumerate(d['ids'].tolist()) for e in row for begin,count in groups[int(e)]}
    assert sum(len(set(row)) for row in d['ids'].tolist())==d['ids'].numel(), 'duplicate routed expert'
    keys=[tuple(row) for row in meta[valid].tolist()]
    assert len(keys)==len(set(keys)), 'duplicate canonical contribution'
    assert set(keys)==expected, f'missing/extra contributions: {len(expected-set(keys))}/{len(set(keys)-expected)}'
    # Validate physical slot ownership, including all tail/unused-group sentinels.
    base=0
    for expert,count in enumerate(counts):
        for tile in range((count+127)//128):
            rows=min(128,count-tile*128)
            for group in range(2):
                mm=d['metadata'][(base+tile)*2+group]
                active=group==0 or base+tile<split
                assert int((mm[:,0]>=0).sum())==(rows if active else 0)
                if active:
                    assert (mm[:rows,1]==expert).all() and (mm[rows:]==-1).all()
                    assert (mm[:rows,2]==group*2).all()
        base+=(count+127)//128
    order=sorted(range(len(keys)),key=lambda i:keys[i]);values=p[valid][order];keys=[keys[i] for i in order]
    ref=torch.zeros(d['output'].shape,dtype=torch.float64)
    for key,value in zip(keys,values):ref[key[0]]+=value.double()
    return keys,values,ref

def error(x,y):
    delta=x.double()-y.double()
    return dict(rel_l2=float(delta.norm()/y.double().norm()),max_abs=float(delta.abs().max()))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('a',type=Path);ap.add_argument('b',type=Path);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    a,b=[torch.load(p,map_location='cpu',weights_only=True) for p in (args.a,args.b)]
    assert torch.equal(a['ids'],b['ids']) and torch.equal(a['weights'],b['weights'])
    ka,va,ra=canonical(a);kb,vb,rb=canonical(b);assert ka==kb
    result=dict(contributions=len(ka),payload_bit_mismatches=int((va.view(torch.int16)!=vb.view(torch.int16)).sum()),payload_error=error(va,vb),reference_error=error(ra,rb),baseline_atomic_vs_own_fixed_fp64=error(a['output'],ra),candidate_atomic_vs_own_fixed_fp64=error(b['output'],rb),cross_atomic=error(a['output'],b['output']),scope=f'FC2 first {a["output"].shape[1]} columns; diagnostic instrumentation, no performance claim')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
