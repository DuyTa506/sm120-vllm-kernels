import ast,unittest
from pathlib import Path
from prepare import generate,PRED
HERE=Path(__file__).resolve().parent
class Undo(ast.NodeTransformer):
 def __init__(self):self.count=0;self.shape=0
 def visit_ClassDef(self,n):
  if n.name=='MoEGatedDynamicKernelDefinedFC2':n.name='MoEGatedDynamicKernel'
  return self.generic_visit(n)
 def generic_visit(self,n):
  for field,value in ast.iter_fields(n):
   if isinstance(value,list):
    result=[];i=0
    while i<len(value):
     item=value[i]
     if i+1<len(value) and isinstance(value[i+1],ast.If) and ast.unparse(value[i+1].test)==PRED:
      guard=value[i+1];assert len(guard.body)==1 and not guard.orelse
      c=guard.body[0].value;assert ast.unparse(c.func)=='cute.copy'
      assert ast.unparse(item)=='cute.recast_tensor('+ast.unparse(c.args[-1])+', cutlass.Uint32).fill(0)'
      self.count+=1;result.extend(guard.body);i+=2;continue
     if isinstance(item,ast.AST):
      item=self.visit(item)
      if item is None:i+=1;continue
     result.append(item);i+=1
    setattr(n,field,result)
   elif isinstance(value,ast.AST):setattr(n,field,self.visit(value))
  return n
 def visit_If(self,n):
  if ast.unparse(n.test)=='cutlass.const_expr(a_input.shape[1] != 2048 or b_down.shape[1] != 512)':self.shape+=1;return None
  return self.generic_visit(n)
def audit(c,b):
 u=Undo();t=u.visit(ast.parse(c));assert (u.count,u.shape)==(6,1)
 assert ast.dump(t,include_attributes=False)==ast.dump(ast.parse(b),include_attributes=False)
class Test(unittest.TestCase):
 def setUp(self):self.b=(HERE.parent/'moe-inactive-loads/gated_baseline.py').read_text();self.c=(HERE/'gated_defined_fc2.py').read_text()
 def test_exact_ast(self):audit(self.c,self.b)
 def test_reproducible(self):self.assertEqual(generate(self.b)[0],self.c)
 def test_wrong_destination_rejected(self):
  with self.assertRaises(AssertionError):audit(self.c.replace('cute.recast_tensor(crB[None, None, 0], cutlass.Uint32).fill(0)','cute.recast_tensor(crB[None, None, 1], cutlass.Uint32).fill(0)',1),self.b)
 def test_changed_wait_rejected(self):
  with self.assertRaises(AssertionError):audit(self.c.replace('phase2_pipeline.consumer_release(phase2_cons_state)','phase2_pipeline.consumer_wait(phase2_cons_state)',1),self.b)
 def test_no_active_fragment_overwritten(self):
  for rows in range(1,129):
   for warp in range(8):
    active=rows>(warp&3)*16
    for mt in range(2):
     consumed=rows>mt*64+(warp&3)*16
     if consumed:self.assertTrue(active)
if __name__=='__main__':unittest.main(verbosity=2)
