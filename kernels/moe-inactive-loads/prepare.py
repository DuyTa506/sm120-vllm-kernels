"""Deterministically wrap exactly the audited operand loads; never edit installed source."""
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'moe-workspace-bands/results/workspace_bands_smoke_1/runtime_sources/0_gated.py'
SHA='993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445'
PRED='valid_rows > warp_m_coord * Int32(16)'
def generate(source):
 if hashlib.sha256(source.encode()).hexdigest()!=SHA:raise ValueError('baseline source mismatch')
 tree=ast.parse(source);patches=[];sites=[]
 for fn in ast.walk(tree):
  if not isinstance(fn,ast.FunctionDef):continue
  if fn.name in ('fc1_gate_up_swiglu_to_sC_tail','fc2_accumulate_slice_tail'):
   for node in ast.walk(fn):
    if not isinstance(node,ast.Expr) or not isinstance(node.value,ast.Call):continue
    call=node.value
    if ast.unparse(call.func)=='cute.copy' and ast.unparse(call.args[0]).startswith('smem_copy_'):
     assert 'cr' in ast.unparse(call.args[-1])
     patches.append((node.lineno,node.end_lineno,node.col_offset,PRED));sites.append(dict(method=fn.name,line=node.lineno,destination=ast.unparse(call.args[-1])))
  if fn.name=='kernel':
   for node in ast.walk(fn):
    if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='self.load_fc2_a_fragments':
     patches.append((node.lineno,node.end_lineno,node.col_offset,PRED));sites.append(dict(method=fn.name,line=node.lineno,destination='load_fc2_a_fragments'))
 assert len(sites)==31,len(sites)
 lines=source.splitlines(keepends=True)
 for start,end,col,pred in sorted(patches,reverse=True):
  lines[start-1:end]=[' '*col+'if '+pred+':\n']+['    '+line for line in lines[start-1:end]]
 result=''.join(lines).replace('class MoEGatedDynamicKernel:','class MoEGatedDynamicKernelInactiveLoads:')
 anchor='        hidden_size = a_input.shape[1]\n';assert result.count(anchor)==1
 result=result.replace(anchor,anchor+'''        if cutlass.const_expr(a_input.shape[1] != 2048 or b_down.shape[1] != 512):
            raise ValueError("inactive-load candidate requires K2048/I512")
''')
 ast.parse(result)
 return result,sites
if __name__=='__main__':
 source=BASE.read_text();candidate,sites=generate(source)
 (HERE/'gated_baseline.py').write_text(source);(HERE/'gated_inactive_loads.py').write_text(candidate)
 (HERE/'source_manifest.json').write_text(json.dumps(dict(baseline_sha256=SHA,candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),sites=sites,scope='M128/I512/K2048; 24 FC1-tail copies, 6 FC2-tail copies, 1 FC2 A/SFA call; no scheduling changes'),indent=2)+'\n')
