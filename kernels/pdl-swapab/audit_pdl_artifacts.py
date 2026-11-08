#!/usr/bin/env python3
"""CPU-only binding of generated PDL wrappers to current saved AOT model artifacts.
Does not execute pickle opcodes; does not claim per-request runtime execution.
"""
import argparse
import ast
import hashlib
import json
import pickletools
import re
from pathlib import Path

def sha(data): return hashlib.sha256(data).hexdigest()
def strings(data):
    for op,arg,pos in pickletools.genops(data):
        if isinstance(arg,str): yield arg
        elif isinstance(arg,bytes) and arg.startswith((b'\x80\x04',b'\x80\x05')):
            yield from strings(arg)
def runtime_calls(source):
    tree=ast.parse(source)
    result=[]
    for fn in ast.walk(tree):
        if isinstance(fn,ast.FunctionDef) and fn.name=='call':
            for node in sorted((n for n in ast.walk(fn) if isinstance(n,ast.Call)),key=lambda n:(n.lineno,n.col_offset)):
                name=ast.unparse(node.func)
                if name.endswith('.run') or name.startswith(('torch.ops.','extern_kernels.')):
                    result.append({'line':node.lineno,'call':ast.unparse(node)})
    return result

def audit(root):
    cache=root/'serving/pdl_cache/vllm/torch_compile_cache/torch_aot_compile'
    log=(root/'startup.log').read_text()
    recorded=re.findall(r'saved AOT compiled function to (/bench-results/[^\s]+/model)',log)
    models=[]
    wrappers={sha(p.read_bytes()):p for p in cache.rglob('*.py')}
    for value in recorded:
        path=root/'serving'/value.removeprefix('/bench-results/')
        if not path.is_file(): raise ValueError('startup model missing: '+value)
        data=path.read_bytes(); linked=[]
        for source in strings(data):
            digest=sha(source.encode())
            if digest not in wrappers: continue
            linked.append({'path':str(wrappers[digest].relative_to(root)), 'sha256':digest,
                'wait':'tl.extra.cuda.gdc_wait()' in source,
                'signal':'tl.extra.cuda.gdc_launch_dependents()' in source,
                'launch_pdl':bool(re.search(r'[\'\"]launch_pdl[\'\"]: True',source)),
                'call_sites':runtime_calls(source)})
        models.append({'path':str(path.relative_to(root)), 'sha256':sha(data),'linked_wrappers':linked})
    if not models or not all(m['linked_wrappers'] for m in models): raise ValueError('current startup AOT wrapper binding incomplete')
    return {'status':'saved_current_startup_artifacts_bound','models':models,'startup_sha256':sha(log.encode()),
        'script_sha256':sha(Path(__file__).read_bytes()),
        'limitation':'Current startup saved AOT artifact linkage only. Per-request execution and overlapping dependent consumer need runtime trace; no overlap claim.'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.write_text(json.dumps(audit(a.run),indent=2)+'\n')
