import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
s=importlib.util.spec_from_file_location('compare_pdl',Path(__file__).with_name('compare_pdl.py'));c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
class Adapter(unittest.TestCase):
    def test_normalization_and_mode_gate(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'run';root.mkdir();(root/'exit_code.txt').write_text('0');(root/'pdl_source_manifest.json').write_text(json.dumps({'files':{}}))
            data=[{'Name':'/qwen-pdl-run','Config':{'Env':['TORCHINDUCTOR_ENABLE_PDL=0','TORCHINDUCTOR_CACHE_DIR=/bench-results/pdl_cache/inductor','TRITON_CACHE_DIR=/bench-results/pdl_cache/triton','VLLM_CACHE_ROOT=/bench-results/pdl_cache/vllm','KEEP=yes']},'Mounts':[{'Destination':'/bench-results','Source':str(root/'serving')}]}]
            original=json.dumps(data);(root/'container.json').write_text(original)
            with patch.object(c.P,'run'):
                changes=c.normalize(root,Path(td)/'normalized',0)
                out=json.loads((Path(td)/'normalized/container.json').read_text())[0]
                self.assertEqual(out['Name'],'/qwen-exp-run');self.assertIn('KEEP=yes',out['Config']['Env'])
                self.assertEqual((root/'container.json').read_text(),original)
                with self.assertRaises(ValueError):c.normalize(root,Path(td)/'bad',1)
            self.assertEqual(len(changes),1)
if __name__=='__main__':unittest.main()
