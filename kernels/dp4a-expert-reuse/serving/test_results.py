import unittest
from check_results import check
class Tests(unittest.TestCase):
 def fixture(self):return dict(completed=2,failed=0,num_prompts=2,total_input_tokens=256,total_output_tokens=64,input_lens=[128,128],output_lens=[32,32],errors=['',''])
 def test_complete(self):check(self.fixture(),2,128,32)
 def test_partial_and_truncated(self):
  for field,value in [('failed',1),('completed',1),('total_output_tokens',63),('output_lens',[32,31])]:
   d=self.fixture();d[field]=value
   with self.assertRaises(AssertionError):check(d,2,128,32)
if __name__=='__main__':unittest.main()
