import importlib.util
import pickle
from pathlib import Path
import tempfile
import unittest
s=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('audit_pdl_artifacts.py'));a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
class Binding(unittest.TestCase):
    def test_current_models_only(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); cache=root/'serving/pdl_cache/vllm/torch_compile_cache/torch_aot_compile/x/rank_0_0';cache.mkdir(parents=True)
            source="def call(x):\n    tl.extra.cuda.gdc_wait()\n    meta = {'launch_pdl': True}\n    foo.run(x)"
            (cache/'current.py').write_text(source); (cache/'unlinked.py').write_text(source+'\nother.run()')
            (cache/'model').write_bytes(pickle.dumps({'nested':pickle.dumps({'source':source})}))
            (root/'startup.log').write_text('saved AOT compiled function to /bench-results/pdl_cache/vllm/torch_compile_cache/torch_aot_compile/x/rank_0_0/model\n')
            report=a.audit(root);self.assertEqual(len(report['models'][0]['linked_wrappers']),1)
            self.assertTrue(report['models'][0]['linked_wrappers'][0]['launch_pdl'])
            (cache/'current.py').write_text('modified')
            with self.assertRaises(ValueError):a.audit(root)
if __name__=='__main__':unittest.main()
