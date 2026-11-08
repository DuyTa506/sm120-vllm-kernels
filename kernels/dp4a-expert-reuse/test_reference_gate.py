import unittest
from reference_gate import check
class Tests(unittest.TestCase):
 def fixture(self):
  return {'current_scalar_cutlass':dict(finite=True,rel_l2=.2),'candidate':dict(finite=True,rel_l2=.21)}, {'current_scalar_cutlass':1.,'candidate':1.}
 def test_valid_is_not_quality_promotion(self):
  e,n=self.fixture();self.assertFalse(check(e,n,1.)['quality_equivalence'])
 def test_zero_even_if_mutual_parity_passes(self):
  e,n=self.fixture();n['candidate']=0
  with self.assertRaisesRegex(ValueError,'zero output'):check(e,n,1.)
 def test_m8_auto_collapse(self):
  e,n=self.fixture();e['candidate']['rel_l2']=1.
  with self.assertRaises(ValueError):check(e,n,1.)
 def test_nan_and_baseline_failure(self):
  for value in [float('nan'),.36]:
   e,n=self.fixture();e['current_scalar_cutlass']['rel_l2']=value
   with self.assertRaises(ValueError):check(e,n,1.)
 def test_relative_gate_below_absolute_ceiling(self):
  e,n=self.fixture();e['candidate']['rel_l2']=.3
  with self.assertRaises(ValueError):check(e,n,1.)
if __name__=='__main__':unittest.main()
