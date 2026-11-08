import unittest
from policy import eligible
class Policy(unittest.TestCase):
 def test_exact_bucket(self):
  self.assertTrue(eligible(True,1024,2048,8,256,False,False))
  for m in [1,3,512,1023,1025,4096]:self.assertFalse(eligible(True,m,2048,8,256,False,False))
 def test_contract(self):
  self.assertFalse(eligible(False,1024,2048,8,256,False,False))
  self.assertFalse(eligible(True,1024,2048,8,256,True,False))
  self.assertFalse(eligible(True,1024,2048,8,256,False,True))
if __name__=='__main__':unittest.main()
