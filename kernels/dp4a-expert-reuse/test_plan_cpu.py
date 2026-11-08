"""Verify preplanning allocation call boundary with a strict CPU fake ABI."""
import unittest
from types import SimpleNamespace
from fi_plan import Plan
class Tensor:
 def __init__(self):self.shape=(3,2048);self.device='cpu';self.calls=0
 def expand(self,*args):self.calls+=1;return self
 def contiguous(self):self.calls+=1;return self
class API:
 def __init__(self):self._FORCED_BACKEND=None;self.prep=0;self.calls=[]
 def _get_weight_views(self,**kw):self.prep+=1;return object()
 def allocate_sm120_static_workspace(self,**kw):self.prep+=1;return object()
 def launch_sm120_moe(self,**kw):self.calls.append((self._FORCED_BACKEND,kw));return kw['scatter_output']
class Tests(unittest.TestCase):
 def test_allocations_outside_call(self):
  api=API();p={k:Tensor() for k in ['a13','a2','g1','g2','w13','w2']};e=SimpleNamespace(w1_sf_mma=object(),w2_sf_mma=object())
  plan=Plan(api,e,p,Tensor(),Tensor(),Tensor(),Tensor(),'direct_micro');counts={k:v.calls for k,v in p.items()}
  for _ in range(3):plan()
  self.assertEqual(api.prep,2);self.assertEqual(counts,{k:v.calls for k,v in p.items()});self.assertIsNone(api._FORCED_BACKEND)
  for forced,kw in api.calls:
   self.assertEqual(forced,'direct_micro');self.assertTrue(kw['input_scales_are_reciprocal']);self.assertIs(kw['_weight_views'],plan.views);self.assertIs(kw['w1_alpha'],p['g1'])
 def test_multiplier_only_for_forced_direct(self):
  api=API();p={k:Tensor() for k in ['a13','a2','a1_gscale','a2_gscale','g1','g2','w13','w2']};e=SimpleNamespace(w1_sf_mma=object(),w2_sf_mma=object())
  with self.assertRaises(ValueError):Plan(api,e,p,Tensor(),Tensor(),Tensor(),Tensor(),None,direct_multiplier=True)
  plan=Plan(api,e,p,Tensor(),Tensor(),Tensor(),Tensor(),'direct_micro',direct_multiplier=True)
  plan();forced,kw=api.calls[-1]
  self.assertFalse(kw['input_scales_are_reciprocal']);self.assertIs(kw['input_global_scale'],p['a1_gscale']);self.assertIs(kw['fc2_input_scale'],p['a2_gscale']);self.assertIs(kw['w1_alpha'],p['g1'])
if __name__=='__main__':unittest.main()
