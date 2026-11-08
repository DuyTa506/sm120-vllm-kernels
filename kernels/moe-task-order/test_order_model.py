import unittest
from order_model import descriptors

class OrderTests(unittest.TestCase):
    def test_exhaustive_descriptor_bijection(self):
        for total in range(41):
            experts=[t%13 for t in range(total)]
            rows=[1+(t*37)%128 for t in range(total)]
            for gate_tiles in range(1,10):
                for split in range(total+1):
                    old=descriptors(experts,rows,gate_tiles,split)
                    new=descriptors(experts,rows,gate_tiles,split,True)
                    self.assertEqual(sorted(old),sorted(new))
                    self.assertEqual(len(new),total+split*((gate_tiles+1)//2-1))
                    if split in (0,total):self.assertEqual(old,new)
    def test_long_tasks_first_and_packed_payload_preserved(self):
        new=descriptors([0,1,2],[128,17,1],4,1,True)
        self.assertEqual([x[0]>>16 for x in new],[1,2,0,0])
        self.assertEqual([x[1]>>20 for x in new],[4,4,2,2])
    def test_invalid_split_rejected(self):
        with self.assertRaises(AssertionError):descriptors([0],[128],4,2)
