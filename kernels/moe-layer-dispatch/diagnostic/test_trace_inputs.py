from types import SimpleNamespace
import unittest
from trace_inputs import describe

class TraceInputTests(unittest.TestCase):
    def test_preserves_optional_positions_and_non_tensor_inputs(self):
        values=[SimpleNamespace(shape=(2,4),dtype='bf16'),None,3,False,'silu',object()]
        result=list(map(describe,values))
        self.assertEqual(result[0],dict(kind='tensor',shape=[2,4],dtype='bf16'))
        self.assertIsNone(result[1])
        self.assertEqual(result[2],dict(kind='int',value=3))
        self.assertEqual(result[3],dict(kind='bool',value=False))
        self.assertEqual(result[4],dict(kind='str',value='silu'))
        self.assertEqual(result[5],dict(kind='object'))
