import json
from pathlib import Path
import random
import struct
import unittest
HERE=Path(__file__).resolve().parent

def fp32(x):return struct.unpack('f',struct.pack('f',x))[0]
def ordered(acc,token,channel):return fp32(fp32(acc*token)*channel)
def swapped(acc,effective_row,effective_col):return fp32(fp32(acc*effective_col)*effective_row)
class Tests(unittest.TestCase):
 def test_scale_order_counterexample(self):
  rng=random.Random(14);witness=None
  for _ in range(20000):
   acc,token,channel=[fp32(rng.uniform(.5,2)*2**rng.randrange(-15,15)) for i in range(3)]
   self.assertEqual(ordered(acc,token,channel),swapped(acc,channel,token))
   wrong=fp32(fp32(acc*channel)*token)
   if wrong!=ordered(acc,token,channel):witness=(acc,token,channel);break
  self.assertIsNotNone(witness,'test did not exercise non-associative FP32 scales')
 def test_original_api_unchanged(self):
  root=HERE/'rowwise_swap_control/src';base=HERE.parent/'deepgemm-sm120/fused_src'
  def api(p):
   text=(p/'csrc/apis/gemm.hpp').read_text();start=text.index('static void fp8_rowwise_gemm_nt_sm120(');end=text.index('\n}\n',start)+3;return text[start:end]
  self.assertEqual(api(root),api(base))
 def test_owned_patch_exact_and_guarded(self):
  import hashlib
  root=HERE/'rowwise_swap_control';m=json.loads((root/'manifest.json').read_text())
  for path,info in m['files'].items():self.assertEqual(hashlib.sha256((root/'src'/path).read_bytes()).hexdigest(),info['patched_sha256'])
  api=(root/'src/csrc/apis/gemm.hpp').read_text();self.assertIn('m >= 1 and m <= 16',api)
  launcher=(root/'src/csrc/jit_kernels/impls/sm120_fp8_fp4_gemm_1d1d.hpp').read_text()
  self.assertIn('config.split_k_factor == 1 and not c.has_value()',launcher)
  self.assertIn('if (swap_ab) DG_HOST_ASSERT(n >= 1 and n <= 16)',launcher)
 def test_actual_skinny_tiny_configuration(self):
  import importlib.util
  spec=importlib.util.spec_from_file_location('replay',HERE/'replay_rowwise_swap.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  pick=module.load_pick(HERE.parent/'deepgemm-sm120/skinny_comparator/reference_dispatch.py')
  for m in (1,3,8,16):self.assertEqual(pick(m,12288,2048),(1,4))
if __name__=='__main__':unittest.main()
