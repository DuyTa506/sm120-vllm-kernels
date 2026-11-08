import unittest
from pathlib import Path
class PortContract(unittest.TestCase):
 def test_explicit_workspace(self):
  s=Path(__file__).with_name('qf_moe_port.cu').read_text()
  self.assertNotIn('static at::Tensor',s)
  self.assertIn('at::Tensor & part_out',s)
  self.assertIn('cudaMemsetAsync(out.data_ptr()',s)
  self.assertIn('cudaMemsetAsync(done.data_ptr()',s)
 def test_boundary_guards(self):
  s=Path(__file__).with_name('qf_moe_port.cu').read_text()
  self.assertIn('topk_w.dim()==2',s)
  self.assertIn('a0 + a->nbytes() <= b0',s)
  self.assertIn('x.is_cuda() && x.dim()==2',s)
  self.assertIn('w13.scalar_type()==at::kByte',s)
 def test_geometry_and_reuse(self):
  s=Path(__file__).with_name('qf_moe_port.cu').read_text()
  self.assertIn('H = 2048, I = 512, TOPK = 8',s)
  self.assertNotIn('wv[10]',s); self.assertNotIn('wv[5]',s)
  self.assertIn('bool reuse',s)
  self.assertNotIn('at::empty',s)
if __name__=='__main__':unittest.main()
