#!/usr/bin/env python3
"""Strict PDL pair adapter. Originals remain immutable; every rewrite is reported."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value
P = module('pdl_provenance', HERE/'pdl_provenance.py')
G = module('pdl_greedy', ROOT/'bench/compare_greedy_decode.py')
def require(value, message):
    if not value: raise ValueError(message)
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def normalize(original, target, mode):
    manifest=json.loads((original/'pdl_source_manifest.json').read_text())
    if set(manifest['files']) == {'run_vllm_pdl_focused.sh','pdl_focused_provenance.py'}:
        module('pdl_focused_provenance', HERE/'pdl_focused_provenance.py').run('verify',original,mode)
    else:
        P.run('verify', original, mode)
    require((original/'exit_code.txt').read_text().strip() == '0', 'incomplete run')
    original = original.resolve()
    shutil.copytree(original, target, ignore=shutil.ignore_patterns('pdl_cache'))
    changes = []
    old, new = 'qwen-pdl-'+original.name, 'qwen-exp-'+original.name
    container_path = target/'container.json'
    data = json.loads(container_path.read_text()); c, = data
    require(c['Name'] == '/'+old, 'unexpected container name')
    c['Name'] = '/'+new
    env = dict(v.split('=',1) for v in c['Config']['Env'])
    require(env.pop('TORCHINDUCTOR_ENABLE_PDL') == str(mode), 'wrong PDL mode')
    require(env.pop('TORCHINDUCTOR_CACHE_DIR') == '/bench-results/pdl_cache/inductor', 'wrong Inductor cache')
    require(env.pop('TRITON_CACHE_DIR') == '/bench-results/pdl_cache/triton', 'wrong Triton cache')
    require(env['VLLM_CACHE_ROOT'] == '/bench-results/pdl_cache/vllm', 'wrong vLLM cache')
    env['VLLM_CACHE_ROOT'] = '/root/.cache/vllm/qwen-sequential-scalar'
    c['Config']['Env'] = [k+'='+v for k,v in env.items()]
    mount, = [m for m in c['Mounts'] if m['Destination']=='/bench-results']
    require(Path(mount['Source']).resolve() == original/'serving', 'wrong original result mount')
    mount['Source'] = str(ROOT/'results/qwen_sequential'/original.name/'serving')
    container_path.write_text(json.dumps(data))
    changes.append({'path':'container.json', 'original_sha256':digest(original/'container.json'),
                    'allowed_fields':['Name', 'PDL mode removed after validation', 'three isolated cache variables', '/bench-results host source']})
    directory = target/'serving'
    for path in directory.rglob('*'):
        if path.is_file() and path.suffix in ('.json','.jsonl','.log','.txt'):
            text = path.read_text()
            if old in text:
                path.write_text(text.replace(old,new))
                changes.append({'path':str(path.relative_to(target)), 'original_sha256':digest(original/path.relative_to(target)),
                                'allowed_rewrite':{'from':old,'to':new}})
    return changes

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('baseline',type=Path); ap.add_argument('candidate',type=Path); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--prefill',action='store_true')
    args=ap.parse_args()
    require(args.baseline.resolve()!=args.candidate.resolve(), 'same run twice')
    manifests=[json.loads((p/'pdl_source_manifest.json').read_text()) for p in (args.baseline,args.candidate)]
    require(manifests[0]['files']==manifests[1]['files'], 'PDL driver/helper changed across pair')
    with tempfile.TemporaryDirectory(prefix='pdl-compare-') as td:
        a,b=Path(td)/'base'/args.baseline.name,Path(td)/'candidate'/args.candidate.name
        rewrites=[normalize(args.baseline,a,0),normalize(args.candidate,b,1)]
        report=G.compare(G.load(a),G.load(b))
        if args.prefill:
            S=module('pdl_serving',ROOT/'bench/compare_serving_matrix.py')
            aa,bb=S.load(a,'prefill'),S.load(b,'prefill')
            av,bv=aa[2]['comparison_validation'],bb[2]['comparison_validation']
            changes=S.IDENTITY.compare(av['runtime'],bv['runtime'])
            for key in ('expected_lengths','normalized_commands'):
                require(av[key]==bv[key],'prefill incompatible '+key)
            require(aa[3]['run_nonce']!=bb[3]['run_nonce'],'prefill cache namespace reused')
            require(aa[3]['scenarios']==bb[3]['scenarios'],'prefill workloads differ')
            report['prefill']={'runtime_changes':changes,'baseline':aa[2],'candidate':bb[2],
                'baseline_manifest':aa[3],'candidate_manifest':bb[3],
                'interpretation':'Single paired six-case prefill sweep; repeat reversed order before causal speedup claim.'}
    report['pdl_adapter']={'normalizations':rewrites,'frozen_driver_and_helper':manifests[0]['files'],
        'scope':'Same scalar overlay; only validated PDL mode, fresh cache namespace and container/result naming adapted. Common strict identity, FI and workload gates retained.',
        'script_sha256':digest(Path(__file__))}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(G.markdown(report))
if __name__=='__main__': main()
