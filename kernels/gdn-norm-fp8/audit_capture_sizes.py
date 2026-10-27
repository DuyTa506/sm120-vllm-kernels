#!/usr/bin/env python3
"""CPU-only reproduction of installed MRV2 graph descriptors, without vLLM imports."""
import ast
from collections import defaultdict
from dataclasses import dataclass, replace
import enum
import hashlib
from itertools import groupby, product
import json
from pathlib import Path
import sys
from types import SimpleNamespace

SOURCE_ROOT = Path('/usr/local/lib/python3.12/dist-packages/vllm')


def installed_capture_descriptors(capture_sizes, max_tokens, max_num_seqs, query_len,
                                  source_root=SOURCE_ROOT):
    config_path = source_root / 'config/compilation.py'
    manager_path = source_root / 'v1/worker/gpu/cudagraph_utils.py'
    enum_node = next(n for n in ast.parse(config_path.read_text()).body
                     if isinstance(n, ast.ClassDef) and n.name == 'CUDAGraphMode')
    tree = ast.parse(manager_path.read_text())
    descriptor = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                      and n.name == 'BatchExecutionDescriptor')
    manager = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CudaGraphManager')
    init = next(n for n in manager.body if isinstance(n, ast.FunctionDef) and n.name == '_init_candidates')
    namespace = dict(enum=enum, dataclass=dataclass, replace=replace, defaultdict=defaultdict,
                     groupby=groupby, product=product, round_up=lambda n, d: (n+d-1)//d*d,
                     __name__=__name__)
    exec(compile(ast.Module(body=[enum_node, descriptor, init], type_ignores=[]),
                 str(manager_path), 'exec'), namespace)
    mode = namespace['CUDAGraphMode']
    spec = SimpleNamespace(uses_dynamic_speculative_decoding=lambda: False)
    model_config = SimpleNamespace(speculative_config=spec, num_speculative_tokens=query_len-1,
                                   cache_config=SimpleNamespace(use_kda_recoverssm=False))
    obj = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_capture_sizes=capture_sizes,
                            max_cudagraph_capture_size=max_tokens),
                          max_num_reqs=max_num_seqs, decode_query_len=query_len,
                          cudagraph_mode=mode.FULL_AND_PIECEWISE, varlen_decode=False,
                          lora_capture_cases=[0], vllm_config=model_config,
                          _capture_descs={}, _candidates={}, _maybe_ubatch_twin=lambda desc: None)
    namespace['_init_candidates'](obj)
    return ({m.name:[d.num_tokens for d in ds] for m,ds in obj._capture_descs.items()},
            {str(p.relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (config_path,manager_path)})


def main():
    from test_integration_guards import load_guards
    root=Path(__file__).resolve().parent
    fixture=root.parent.parent/'results/qwen_sequential/gdn_integration_1/source_files/scripts/cudagraph_sizes.txt'
    sizes=[int(x) for x in fixture.read_text().split()]
    descs,hashes=installed_capture_descriptors(sizes,max(sizes),64,3)
    original=sorted({1,2,16384,*sizes})
    report=dict(cuda_imported='torch' in sys.modules,source_sha256=hashes,
                capture_sizes=sizes,decode_query_len=3,max_num_seqs=64,
                original_warmup_count=len(original),
                actual_descriptors=descs,original_warmups=original,
                missing_from_original=sorted(set(descs['FULL'])-set(original)),
                first_two_full_descriptors=descs['FULL'][:2],
                failure_inference='Profile-only FULL capture samples descending largest2;192 is warmed,186 is not. Failed log lacks actual T, so186 is source-derived rather than logged.')
    g=load_guards()
    try:
        report['fixed_warmups']=g.fallback_warmup_sizes(sizes,16384,decode_query_len=3,
                                                       max_num_seqs=64,max_capture_size=16384)
        report['missing_from_fixed']=sorted(set(descs['FULL'])-set(report['fixed_warmups']))
        report['fixed_warmup_count']=len(report['fixed_warmups'])
    except TypeError:
        report['fixed_warmups']=None
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
