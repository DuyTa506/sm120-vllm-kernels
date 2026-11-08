"""CPU-only deterministic FC2-tail defined-destination experiment."""
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SHA='993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445'
PRED='valid_rows > warp_m_coord * Int32(16)'
def generate(source):
 if hashlib.sha256(source.encode()).hexdigest()!=SHA:raise ValueError('baseline SHA mismatch')
 fn=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='fc2_accumulate_slice_tail')
 sites=[]
 for n in ast.walk(fn):
  if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='cute.copy':
   c=n.value;assert ast.unparse(c.args[0]) in ['smem_copy_B','smem_copy_SFB']
   dst=ast.unparse(c.args[-1]);assert dst.startswith(('crB[','fz_crSFB['));sites.append((n.lineno,n.end_lineno,n.col_offset,dst))
 assert len(sites)==6
 lines=source.splitlines(keepends=True)
 for start,end,col,dst in sorted(sites,reverse=True):
  lines[start-1:end]=[' '*col+'cute.recast_tensor('+dst+', cutlass.Uint32).fill(0)\n',' '*col+'if '+PRED+':\n']+['    '+s for s in lines[start-1:end]]
 candidate=''.join(lines).replace('class MoEGatedDynamicKernel:','class MoEGatedDynamicKernelDefinedFC2:')
 anchor='        hidden_size = a_input.shape[1]\n';assert candidate.count(anchor)==1
 candidate=candidate.replace(anchor,anchor+'''        if cutlass.const_expr(a_input.shape[1] != 2048 or b_down.shape[1] != 512):
            raise ValueError("defined-FC2 candidate requires K2048/I512")
''')
 ast.parse(candidate);return candidate,sites
if __name__=='__main__':
 source=(HERE.parent/'moe-inactive-loads/gated_baseline.py').read_text();candidate,sites=generate(source)
 (HERE/'gated_defined_fc2.py').write_text(candidate)
 (HERE/'manifest.json').write_text(json.dumps({'baseline_sha256':SHA,'candidate_sha256':hashlib.sha256(candidate.encode()).hexdigest(),'sites':sites,'scope':'six FC2-tail operand copies only; FC1/common A loads unchanged'},indent=2)+'\n')
