import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import compare_pin_pair as C
class ContractTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.root=Path(self.t.name)
  source=C.HERE/'results/pdl_pin0_1'
  for rel in ('pdl_source_manifest.json','pin_events_verify.json','pin_aot_verify.json','serving/pin_reference.json'):
   dst=self.root/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/rel,dst)
  shutil.copytree(source/'serving/pin_control',self.root/'serving/pin_control')
  self.verification={'selected_inductor_config_count':0}
 def test_valid_empty_best_config_with_pinned_evidence(self):C.pin_singleton_compiler_evidence(self.root,self.verification)
 def test_differing_singleton(self):
  p=self.root/'pin_aot_verify.json';doc=json.loads(p.read_text());doc['records'][0]['compiled_configs'][0]['num_warps']=999;p.write_text(json.dumps(doc))
  with self.assertRaisesRegex(ValueError,'singleton config differs'):C.pin_singleton_compiler_evidence(self.root,self.verification)
 def test_missing_binding(self):
  p=self.root/'pin_aot_verify.json';doc=json.loads(p.read_text());doc['records']=[];p.write_text(json.dumps(doc))
  with self.assertRaisesRegex(ValueError,'empty AOT config coverage'):C.pin_singleton_compiler_evidence(self.root,self.verification)
 def test_compiler_audit_failure_not_waived(self):
  self.verification['compiler_config_audit_error']='failed'
  with self.assertRaisesRegex(ValueError,'audit failed'):C.pin_singleton_compiler_evidence(self.root,self.verification)
if __name__=='__main__':unittest.main()
