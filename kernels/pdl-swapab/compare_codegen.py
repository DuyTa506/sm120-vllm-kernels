#!/usr/bin/env python3
"""Compare generated kernel bodies and observed autotune selections, CPU-only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

def sha(data): return hashlib.sha256(data).hexdigest()
class StripPDL(ast.NodeTransformer):
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and ast.unparse(node.value.func) in ('tl.extra.cuda.gdc_wait','tl.extra.cuda.gdc_launch_dependents'):
            if node.value.args or node.value.keywords: raise ValueError('unexpected PDL arguments')
            return None
        return self.generic_visit(node)
def kernel_body(source):
    tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    body=StripPDL().visit(ast.Module(body=fn.body,type_ignores=[]))
    return ast.dump(body,include_attributes=False)
def collect(root):
    audit=json.loads((root/'serving/fi_control/autotune/compiler_config_audit.json').read_text())
    result={}
    for e in audit['files']:
        p=root/'serving/pdl_cache/vllm'/e['path']
        if sha(p.read_bytes())!=e['sha256']: raise ValueError('config hash changed')
        source,=[q for q in p.parent.glob('*.py') if any(isinstance(n,ast.FunctionDef) and n.name.startswith('triton_') for n in ast.parse(q.read_text()).body)];body=kernel_body(source.read_text());key=sha(body.encode())
        result.setdefault(key,[]).append({'config':e['selected'],'config_path':str(p.relative_to(root)),
            'source_path':str(source.relative_to(root)),'source_sha256':sha(source.read_bytes()),'normalized_body':body})
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    a,b=collect(args.baseline),collect(args.candidate)
    common={}
    for key in sorted(a.keys()&b.keys()):
        configs=lambda entries: sorted(json.dumps(e['config'],sort_keys=True) for e in entries)
        common[key]={'baseline':a[key],'candidate':b[key],'observed_selected_configs_equal':configs(a[key])==configs(b[key])}
    out={'matching_kernel_bodies':common,'baseline_only':{k:a[k] for k in a.keys()-b.keys()},'candidate_only':{k:b[k] for k in b.keys()-a.keys()},
      'normalization':'Function-body AST, strip only no-argument tl.extra.cuda.gdc_wait/gdc_launch_dependents expression statements. Function signatures/decorators are outside this body comparison.',
      'limitation':'Saved best_config choices only, not exhaustive executable/binary identity or runtime selection. Differences prevent attributing timing/NLL solely to PDL.',
      'script_sha256':sha(Path(__file__).read_bytes())}
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'common_bodies':len(common),'changed_selected_configs':sum(not x['observed_selected_configs_equal'] for x in common.values()),'baseline_only':len(out['baseline_only']),'candidate_only':len(out['candidate_only'])}))
