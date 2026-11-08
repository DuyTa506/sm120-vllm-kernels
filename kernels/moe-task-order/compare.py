"""Report timing and atomic-noise distributions; no automatic promotion."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',type=Path,nargs=2,required=True)
    parser.add_argument('--candidate',type=Path,nargs=2,required=True)
    args=parser.parse_args()
    import torch
    paths=args.baseline+args.candidate
    datasets=[]
    for path in paths:
        rows=[json.loads(s) for s in (path/'measurements.jsonl').read_text().splitlines()]
        datasets.append({(r['layer'],r['tokens'],r['pattern']):r for r in rows})
    assert all(set(d)==set(datasets[0]) for d in datasets)
    def distribution(a,b):
        errors=[]
        for left in a:
            for right in b:
                l,r=left.float(),right.float()
                assert torch.isfinite(l).all() and torch.isfinite(r).all()
                errors.append(float((l-r).norm()/r.norm().clamp_min(1e-12)))
        return dict(min=min(errors),median=statistics.median(errors),max=max(errors),comparisons=len(errors))
    output=[]
    for key in datasets[0]:
        records=[d[key] for d in datasets]
        assert len({r['descriptor_multiset_sha256'] for r in records})==1
        layer,tokens,pattern=key
        tensors=[torch.load(p/f'l{layer}_m{tokens}_{pattern}.pt',map_location='cpu',weights_only=True) for p in paths]
        for value in tensors[1:]:
            assert all(torch.equal(tensors[0][field],value[field]) for field in ('input','ids','weights'))
        output.append(dict(layer=layer,tokens=tokens,pattern=pattern,
                           baseline_us=[r['median_us'] for r in records[:2]],candidate_us=[r['median_us'] for r in records[2:]],
                           baseline_baseline=distribution(tensors[0]['outputs'],tensors[1]['outputs']),
                           candidate_candidate=distribution(tensors[2]['outputs'],tensors[3]['outputs']),
                           baseline_candidate=distribution(tensors[0]['outputs'],tensors[2]['outputs']),
                           nontrivial=all(r['permutation_nontrivial'] for r in records)))
    print(json.dumps(dict(rows=output,status='screen only; distribution report does not grant numerical or serving promotion'),indent=2))

if __name__=='__main__':main()
