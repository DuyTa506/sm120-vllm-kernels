import unittest
from check_workload import check_boundary, check_stress
class Workload(unittest.TestCase):
    def test_boundary(self):
        good=dict(completed=32,output_lens=[2304]*32,errors=['']*32)
        check_boundary(good)
        for change in [dict(completed=31),dict(output_lens=[2304]*31),dict(output_lens=[2304]*31+[1]),dict(errors=['failed']*32)]:
            with self.assertRaises(AssertionError): check_boundary(good|change)
    def test_stress(self):
        check_stress('stress: ok=104 aborted=24 failed=0')
        for text in ['stress: ok=0 aborted=0 failed=0','stress: ok=128 aborted=0 failed=0','stress: ok=103 aborted=24 failed=1','stress: ok=100 aborted=24 failed=0']:
            with self.assertRaises(AssertionError): check_stress(text)

    def test_growth_boundary(self):
        good=dict(completed=8,output_lens=[6656]*8,errors=['']*8)
        check_boundary(good,prompts=8,output_tokens=6656)
        with self.assertRaises(AssertionError): check_boundary(good)
        with self.assertRaises(AssertionError): check_boundary(good|dict(output_lens=[6656]*7+[2304]),prompts=8,output_tokens=6656)
        with self.assertRaises(AssertionError): check_boundary(good,prompts=8,output_tokens=4352)
