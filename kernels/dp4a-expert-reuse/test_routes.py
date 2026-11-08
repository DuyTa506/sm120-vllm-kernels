import unittest
from route_contract import validate
class Routes(unittest.TestCase):
 def test_valid(self):
  for m in [1,2,3,4,8]:
   validate([list(range(8)) for _ in range(m)])
   validate([[-1]*8 for _ in range(m)])
   validate([[0,1,2,3,-1,-1,-1,-1] for _ in range(m)])
 def test_bad(self):
  for rows in [[],[[1]*8],[list(range(7))],[list(range(8))]*9]:
   with self.assertRaises(ValueError):validate(rows)
if __name__=='__main__':unittest.main()
