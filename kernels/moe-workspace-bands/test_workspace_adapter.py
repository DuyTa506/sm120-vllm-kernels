"""CPU contracts: graph-pointer lifetime, owner/layer isolation, guards and source gates."""
import ast,hashlib,json,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
import workspace_adapter as A
S=types.SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'kernels/moe-nvfp4/scalar/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py'
EXPECTED_SHA='d965685b7b29811f78efc46e0b6906e37efe601d12b21a119ca36d610059cc62'
class Expert:
 def __init__(self):
  self.global_num_experts=self.num_local_experts=256;self.topk=8;self.hidden_dim=2048;self.intermediate_size_per_partition=512;self.max_num_tokens=16384
  self._hybrid_min_tokens=1024;self._activation_str='silu';self.out_dtype='bf16';self.w1_scale=S(device='cuda:0');self._wrapper=None
class Manager:
 def __init__(self):self.owner='a';self.resources={};self.locked=False
 def get_persistent_resource(self,key,make):
  key=(self.owner,key)
  if key not in self.resources:
   if self.locked:raise RuntimeError('warmup locked')
   self.resources[key]=make()
  return self.resources[key]
class Wrapper:
 count=0
 def __init__(self,**kw):
  Wrapper.count+=1;self.capacity=kw['max_num_tokens'];self.weight_views={}
  rows=self.capacity*kw['top_k'];tile=16 if rows<15*256 else 32 if rows<48*256 else 64 if rows<96*256 else 128
  self._static_workspace=kw.get('shared_static_workspace',object())
  self._dynamic_workspace=kw.get('shared_dynamic_workspace',S(tile_m=tile))
  self._moe_output=kw.get('shared_output',object())
class Tests(unittest.TestCase):
 def setUp(self):self.e=Expert();self.m=Manager();Wrapper.count=0
 def get(self,tokens,e=None):return A.ensure_wrapper(e or self.e,tokens,wrapper_cls=Wrapper,manager=self.m)
 def test_band_boundary_and_outer_dispatch(self):
  for tokens,tile,cap in [(1024,32,1535),(1535,32,1535),(1536,64,3071),(2048,64,3071),(3071,64,3071),(3072,128,16384),(4096,128,16384),(16384,128,16384)]:
   w=self.get(tokens);self.assertEqual((w._dynamic_workspace.tile_m,w.capacity),(tile,cap))
  for t in [0,1,16,1023,16385,True,1024.0]:
   with self.assertRaises(ValueError):self.get(t)
 def test_prepare_all_before_lock_and_no_graph_pointer_mutation(self):
  high=self.get(4096);self.assertEqual(Wrapper.count,6)
  captured=(high._static_workspace,high._dynamic_workspace,high._moe_output)
  self.m.locked=True
  small=self.get(1024);middle=self.get(2048)
  self.assertEqual(Wrapper.count,6);self.assertIs(self.get(4096),high)
  self.assertEqual(captured,(high._static_workspace,high._dynamic_workspace,high._moe_output))
  self.assertIsNot(small._dynamic_workspace,middle._dynamic_workspace)
  self.assertIsNot(small._dynamic_workspace,high._dynamic_workspace)
 def test_layer_views_shared_band_scratch_and_owner_isolation(self):
  wa=self.get(1024);other=Expert();wb=self.get(1024,other)
  self.assertIsNot(wa,wb);self.assertIsNot(wa.weight_views,wb.weight_views)
  self.assertIs(wa._dynamic_workspace,wb._dynamic_workspace);self.assertIs(wa._moe_output,wb._moe_output)
  self.m.owner='b';wc=self.get(1024)
  self.assertIsNot(wc,wa);self.assertIsNot(wc._dynamic_workspace,wa._dynamic_workspace)
  self.m.owner='a';self.assertIs(self.get(1024),wa)
 def test_fail_closed_native_tile_change_and_unknown_owner_after_lock(self):
  w=self.get(1024);w._dynamic_workspace.tile_m=128
  with self.assertRaisesRegex(RuntimeError,'allocated tile changed'):self.get(1024)
  self.m.locked=True;self.m.owner='unknown'
  with self.assertRaisesRegex(RuntimeError,'warmup locked'):self.get(4096)
 def test_shape_threshold_and_capacity_guards(self):
  for name,value in [('topk',9),('num_local_experts',128),('_hybrid_min_tokens',0),('max_num_tokens',2048)]:
   e=Expert();setattr(e,name,value)
   with self.assertRaises(ValueError):self.get(1024,e)
 def test_defaultoff_original_wrapper_and_source_identity(self):
  source=SOURCE.read_text();candidate=A.compose_experts_source(source,EXPECTED_SHA)
  with self.assertRaisesRegex(ValueError,'hash mismatch'):A.compose_experts_source(source+'\n',EXPECTED_SHA)
  tree=ast.parse(candidate);fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_ensure_wrapper')
  ns={'Any':object,'_B12X_SHARED_WORKSPACE':False};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<isolated_ensure>','exec'),ns)
  fakev=types.ModuleType('vllm');fakev.envs=S(VLLM_SM120_WORKSPACE_BANDS=False)
  fakefi=types.ModuleType('flashinfer.fused_moe');fakefi.B12xMoEWrapper=Wrapper
  with patch.dict(sys.modules,{'vllm':fakev,'flashinfer.fused_moe':fakefi}):
   off=ns['_ensure_wrapper'](self.e,1024);self.assertEqual(off.capacity,16384);self.assertEqual(off._dynamic_workspace.tile_m,128)
   self.assertIs(off,ns['_ensure_wrapper'](self.e,2048));self.assertEqual(Wrapper.count,1)
   fakev.envs.VLLM_SM120_WORKSPACE_BANDS=True
   with self.assertRaisesRegex(RuntimeError,'shared workspace'):ns['_ensure_wrapper'](self.e,1024)
  apply=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='apply')
  calls=[n for n in ast.walk(apply) if isinstance(n,ast.Call) and ast.unparse(n.func)=='self._ensure_wrapper']
  self.assertEqual(len(calls),1);self.assertEqual(ast.unparse(calls[0].args[0]),'hidden_states.shape[0]')
 def test_env_registration_default_and_toggle(self):
  import os
  env={'environment_variables':{},'os':os};exec(A.compose_envs_source(''),env)
  with patch.dict(os.environ,{},clear=True):self.assertFalse(env['environment_variables'][A.FLAG]())
  with patch.dict(os.environ,{A.FLAG:'1'}):self.assertTrue(env['environment_variables'][A.FLAG]())
if __name__=='__main__':unittest.main(verbosity=2)
