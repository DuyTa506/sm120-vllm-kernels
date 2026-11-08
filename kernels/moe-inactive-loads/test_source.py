import ast,hashlib,json,unittest
from pathlib import Path
from prepare import generate,PRED,SHA
HERE=Path(__file__).resolve().parent
class Undo(ast.NodeTransformer):
 def __init__(self):self.guards=0;self.shape=0
 def visit_ClassDef(self,node):
  if node.name=='MoEGatedDynamicKernelInactiveLoads':node.name='MoEGatedDynamicKernel'
  return self.generic_visit(node)
 def visit_If(self,node):
  if ast.unparse(node.test)==PRED:
   assert len(node.body)==1 and not node.orelse
   call=node.body[0];assert isinstance(call,ast.Expr) and isinstance(call.value,ast.Call)
   assert ast.unparse(call.value.func) in ('cute.copy','self.load_fc2_a_fragments')
   self.guards+=1;return node.body
  if ast.unparse(node.test)=='cutlass.const_expr(a_input.shape[1] != 2048 or b_down.shape[1] != 512)':
   self.shape+=1;return []
  return self.generic_visit(node)
def audit(candidate,baseline):
 t=Undo();restored=t.visit(ast.parse(candidate))
 assert (t.guards,t.shape)==(31,1)
 assert ast.dump(restored,include_attributes=False)==ast.dump(ast.parse(baseline),include_attributes=False),'non-whitelisted AST change'
 return dict(operand_guards=t.guards,geometry_guards=t.shape,all_other_ast_identical=True)
class Tests(unittest.TestCase):
 def test_exact_audit(self):self.assertTrue(audit((HERE/'gated_inactive_loads.py').read_text(),(HERE/'gated_baseline.py').read_text())['all_other_ast_identical'])
 def test_reproducible(self):
  b=(HERE/'gated_baseline.py').read_text();c,sites=generate(b);self.assertEqual(c,(HERE/'gated_inactive_loads.py').read_text());self.assertEqual(len(sites),31)
 def test_unknown_source(self):
  with self.assertRaises(ValueError):generate((HERE/'gated_baseline.py').read_text()+'\n')
 def test_reject_barrier_change(self):
  c=(HERE/'gated_inactive_loads.py').read_text().replace('self.epilog_sync_barrier.arrive_and_wait()','self.epilog_sync_barrier.arrive()',1)
  with self.assertRaises(AssertionError):audit(c,(HERE/'gated_baseline.py').read_text())
 def test_warp_uniform_truth_table(self):
  for rows in range(1,129):
   for warp in range(8):
    predicate=rows>(warp&3)*16
    active=[rows>(mt*64+(warp&3)*16) for mt in range(2)]
    self.assertEqual(predicate,any(active))
if __name__=='__main__':unittest.main(verbosity=2)
