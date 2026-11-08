import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
s=importlib.util.spec_from_file_location('prov',Path(__file__).with_name('pdl_pin_provenance.py'));p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
class Provenance(unittest.TestCase):
 def test_modes_reference_and_source(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'pin_mode.txt').write_text('pin');p.run('snapshot',root,1)
   (root/'serving').mkdir();ref=root/'serving/pin_reference.json';ref.write_text('{}')
   env={'TORCHINDUCTOR_ENABLE_PDL':'1','VLLM_CACHE_ROOT':'/bench-results/pdl_cache/vllm','TORCHINDUCTOR_CACHE_DIR':'/bench-results/pdl_cache/inductor','TRITON_CACHE_DIR':'/bench-results/pdl_cache/triton','QWEN_INDUCTOR_PIN_MODE':'pin','PYTHONPATH':'/opt/pdl-pin/pin_bootstrap:/opt/qwen-fi-control','QWEN_INDUCTOR_PIN_OUT':'/bench-results/pin_control','QWEN_INDUCTOR_PIN_REFERENCE':'/bench-results/pin_reference.json','QWEN_INDUCTOR_PIN_REFERENCE_SHA256':hashlib.sha256(ref.read_bytes()).hexdigest()}
   def save(): (root/'container.json').write_text(json.dumps([{'Config':{'Env':[k+'='+v for k,v in env.items()]}}]))
   save();self.assertEqual(p.run('verify',root,1)['status'],'passed')
   env['QWEN_INDUCTOR_PIN_MODE']='record';save()
   with self.assertRaises(ValueError):p.run('verify',root,1)
   env['QWEN_INDUCTOR_PIN_MODE']='pin';save();ref.write_text('{"tampered":1}')
   with self.assertRaises(ValueError):p.run('verify',root,1)
if __name__=='__main__':unittest.main()
