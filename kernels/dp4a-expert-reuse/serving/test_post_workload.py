"""Negative fixtures derived from preserved actual ON3 boundary evidence."""
import copy,json,unittest
from pathlib import Path
from verify_post_workload import check_runtime
ROOT=Path(__file__).parent/'results/fi_direct_serving_on3'
class Tests(unittest.TestCase):
 def fixture(self):
  paths=['serving/runtime/startup.json','serving/runtime/post_workload.json','post_workload_rpc.json','serving/fi_control/autotune/verification.json']
  return [json.loads((ROOT/p).read_text()) for p in paths]
 def test_actual_on3_post_passes_without_shutdown(self):check_runtime(*self.fixture())
 def test_missing_post_or_wrong_source_boundary(self):
  for change in [lambda f:f['checks'].pop('server_post_workload'),lambda f:f.update(server_startup_boundary='before_warmup'),lambda f:f['preload'].update(returned=False),lambda f:f['checks']['server_post_workload'].update(changed=['bad'])]:
   s,p,r,f=self.fixture();change(f)
   with self.assertRaises((AssertionError,KeyError)):check_runtime(s,p,r,f)
 def test_rpc_pid_mtp_and_resources(self):
  for change in [lambda p:p.update(pid=999),lambda p:p['speculative_config'].update(num_speculative_tokens=1),lambda p:p['resources'].pop(),lambda p:p['live_graph_replays'].update({'3':0})]:
   s,p,r,f=self.fixture();change(p)
   with self.assertRaises(AssertionError):check_runtime(s,p,r,f)
 def test_forged_matching_rpc_still_rejected(self):
  s,p,r,f=self.fixture();p['pid']=999;r={'results':[p]}
  with self.assertRaises(AssertionError):check_runtime(s,p,r,f)
if __name__=='__main__':unittest.main()
