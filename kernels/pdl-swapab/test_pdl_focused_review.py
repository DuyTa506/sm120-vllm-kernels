import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import pdl_focused_freeze as F
import pin_inductor_core as C
class Review(unittest.TestCase):
 def test_actual_frozen_sources_and_reference(self):
  x=F.verify(HERE/'focused_review.json','pin',HERE/'results/pdl_controlled_reference_v1.json');self.assertEqual(x['files'],30)
  doc=json.loads((HERE/'focused_review.json').read_text());self.assertEqual(len(doc['workload']['prefill']),6);self.assertEqual(len(doc['workload']['greedy']),4);self.assertEqual(doc['workload']['cold_nll_rounds'],5)
 def test_wrong_reference_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'ref.json';p.write_text('{}')
   with self.assertRaisesRegex(ValueError,'reference differs'):F.verify(HERE/'focused_review.json','pin',p)
 def test_changed_reviewed_source_rejected(self):
  doc=json.loads((HERE/'focused_review.json').read_text());doc['files']['kernels/pdl-swapab/run_vllm_pdl_pin_focused.sh']='0'*64
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'review.json';p.write_text(json.dumps(doc))
   with self.assertRaisesRegex(ValueError,'source changed'):F.verify(p,'record',None)
 def test_unknown_lazy_identity_and_config_fail_closed(self):
  reference=json.loads((HERE/'results/pdl_controlled_reference_v1.json').read_text())['reference']
  with self.assertRaisesRegex(ValueError,'unmatched'):C.select(reference,'not_observed_in_focused_workload',[])
  key=next(iter(reference))
  with self.assertRaisesRegex(ValueError,'missing'):C.select(reference,key,[{'XBLOCK':999999}])
if __name__=='__main__':unittest.main()
