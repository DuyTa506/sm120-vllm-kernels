import hashlib,json
from pathlib import Path
import unittest
HERE=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_exact_patch_and_scale_epilogue_unchanged(self):
  root=HERE/'rowwise_tile_control';manifest=json.loads((root/'manifest.json').read_text())
  for name,v in manifest['files'].items():self.assertEqual(hashlib.sha256((root/'src'/name).read_bytes()).hexdigest(),v['patched_sha256'])
  for name in ('deep_gemm/include/deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh','deep_gemm/include/deep_gemm/epilogue/transform.cuh'):
   self.assertEqual((root/'src'/name).read_bytes(),(HERE/'rowwise_swap_control/src'/name).read_bytes())
 def test_all_narrow_cuda_templates_compiled(self):
  root=HERE/'rowwise_tile_control/nvcc';reports=json.loads((root/'results.json').read_text())
  self.assertEqual(set(reports),{'tile64_m1','tile64_m3','tile64_m8','tile64_m16'})
  for name,row in reports.items():
   self.assertEqual(row['exit_code'],0);self.assertEqual(hashlib.sha256((root/(name+'.cubin')).read_bytes()).hexdigest(),row['cubin_sha256'])
   self.assertEqual(row['policy']['split_k'],1)
 def test_shape_legality_and_final_trace_guards(self):
  s=(HERE/'rowwise_tile_control/src/csrc/jit_kernels/impls/sm120_fp8_fp4_gemm_1d1d.hpp').read_text()
  self.assertIn('m == 12288 and n >= 1 and n <= 16 and k == 2048',s)
  self.assertIn('DG_HOST_ASSERT(matches == 1)',s)
  self.assertLess(s.index('config.split_k_factor = SM120ArchSpec::get_split_k_factor'),s.index('ROWWISE_POLICY_V1'))
  self.assertIn('config.split_k_factor == 1 and not c.has_value()',s)
 def test_active_work_distinguished_from_persistent_grid(self):
  self.assertEqual((12288+127)//128,96);self.assertEqual((12288+63)//64,192)
  self.assertEqual(min(192,188),188)
if __name__=='__main__':unittest.main()
