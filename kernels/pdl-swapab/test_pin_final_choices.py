import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
s=importlib.util.spec_from_file_location('pin',Path(__file__).with_name('pin_inductor_core.py'));p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
def cfg(x):return NS(kwargs={'XBLOCK':x},num_warps=4,num_stages=1,num_ctas=1,pre_hook=None,ir_override=None)
class FinalChoices(unittest.TestCase):
 def test_pin_and_late_change_rejection(self):
  source='def k(x):\n    tl.store(x, 0)\n';c1,c2=cfg(1),cfg(2)
  a=NS(fn=NS(src=source),size_hints={'x':16},triton_meta={'signature':{'x':'*bf16'},'constants':{},'configs':[{(0,):[['tt.divisibility',16]]}]},inductor_meta={},configs=[c1,c2],compile_results=[],launchers=[])
  key,_=p.identity(source,a.size_hints,a.triton_meta['signature'],{},p.compiler_options(a.triton_meta));expected=p.config_dict(c1)
  _,chosen=p.apply_to_constructed_autotuner(a,{key:[{'config':expected}]},lambda _:None)
  self.assertEqual(a.configs,[c1]);self.assertFalse(a.inductor_meta['dynamic_scale_rblock']);self.assertFalse(a.inductor_meta['coordinate_descent_tuning'])
  a.compile_results=[NS(config=c1)];a.launchers=[NS(config=c1)]
  p.assert_final_choices(a,chosen,'launchers',lambda _:None)
  a.launchers=[NS(config=c2)]
  with self.assertRaises(ValueError):p.assert_final_choices(a,chosen,'launchers',lambda _:None)
  a.compile_results=[NS(config=c1),NS(config=c2)]
  with self.assertRaises(ValueError):p.assert_final_choices(a,chosen,'compiled',lambda _:None)
 def test_alignment_participates_in_identity(self):
  key=lambda alignment:p.identity('def k(x):\n    tl.store(x, 0)\n',{'x':16},{'x':'*bf16'},{},{'configs':[{(0,):[['tt.divisibility',alignment]]}]})[0]
  self.assertNotEqual(key(16),key(8))
if __name__=='__main__':unittest.main()
