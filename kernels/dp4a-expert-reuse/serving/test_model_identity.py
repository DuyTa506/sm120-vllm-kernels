import unittest
from types import SimpleNamespace as S
from model_identity import bind,require

def config():return S(model_config=S(model='actual-checkpoint',hf_config=S(model_type='qwen3_5_moe',architectures=['Qwen3_5MoeForConditionalGeneration'],text_config=S(hidden_size=2048,moe_intermediate_size=512,num_experts=256,num_experts_per_tok=8))))
class Tests(unittest.TestCase):
 def test_off_never_reads_config(self):
  e=S()
  def forbidden():raise AssertionError('OFF must not require context')
  bind(e,False,forbidden);self.assertFalse(hasattr(e,'_sm120_fi_direct_identity'))
 def test_real_identity_immutable(self):
  e=S();bind(e,True,config);identity=require(e);self.assertEqual(identity['dimensions'],(2048,512,256,8))
  with self.assertRaises(TypeError):identity['model']='other'
 def test_missing_and_wrong_rejected(self):
  with self.assertRaises(RuntimeError):bind(S(),True,lambda:None)
  with self.assertRaises(RuntimeError):require(S())
  for field,value in [('model_type','other'),('architectures',[])]:
   c=config();setattr(c.model_config.hf_config,field,value)
   with self.assertRaises(RuntimeError):bind(S(),True,lambda:c)
  c=config();c.model_config.hf_config.text_config.hidden_size=2560
  with self.assertRaises(RuntimeError):bind(S(),True,lambda:c)
 def test_rebind_different_model_rejected(self):
  e=S();bind(e,True,config);c=config();c.model_config.model='other'
  with self.assertRaises(RuntimeError):bind(e,True,lambda:c)
if __name__=='__main__':unittest.main()
