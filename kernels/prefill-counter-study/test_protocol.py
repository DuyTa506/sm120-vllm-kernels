import ast,unittest
from pathlib import Path
from protocol import command,classify,METRICS,RANGE
class Tests(unittest.TestCase):
 def test_selection(self):
  args=command(Path('/owned'),Path('/output'));self.assertEqual(args[args.index('--nvtx-include')+1],'scalar_L19_M1024/');self.assertEqual(args[args.index('--launch-count')+1],'1')
  for key,value in [('--replay-mode','application'),('--app-replay-mode','strict'),('--cache-control','none'),('--clock-control','none')]:self.assertEqual(args[args.index(key)+1],value)
  self.assertNotIn('--query-metrics',args);self.assertNotIn('--set',args);self.assertEqual(len(METRICS),6)
 def test_failure_not_negative(self):
  self.assertEqual(classify('ERR_NVGPUCTRPERM',1),'counter_permission_unavailable');self.assertEqual(classify('invalid metric',1),'unsupported_metric');self.assertEqual(classify('No kernels were profiled',0),'selection_failed')
 def test_single_range(self):
  tree=ast.parse((Path(__file__).parent/'baseline.py').read_text());calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)];self.assertEqual(calls.count('torch.cuda.nvtx.range_push'),1);self.assertEqual(calls.count('torch.cuda.nvtx.range_pop'),1);self.assertNotIn('torch.cuda.graph',calls)
if __name__=='__main__':unittest.main(verbosity=2)
