import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
spec=importlib.util.spec_from_file_location('pdl',Path(__file__).with_name('pdl_provenance.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Provenance(unittest.TestCase):
    def fixture(self,p,mode):
        m.run('snapshot',p,mode)
        env=['TORCHINDUCTOR_ENABLE_PDL='+str(mode),'VLLM_CACHE_ROOT=/bench-results/pdl_cache/vllm','TORCHINDUCTOR_CACHE_DIR=/bench-results/pdl_cache/inductor','TRITON_CACHE_DIR=/bench-results/pdl_cache/triton']
        (p/'container.json').write_text(json.dumps([{'Config':{'Env':env}}]))
        f=p/'serving/pdl_cache/inductor/code.py';f.parent.mkdir(parents=True);f.write_text("# triton\n"+("tl.extra.cuda.gdc_wait()\n# 'launch_pdl': True\n" if mode else ''))
    def test_modes(self):
        for mode in (0,1):
            with tempfile.TemporaryDirectory() as d:
                p=Path(d);self.fixture(p,mode);self.assertEqual(m.run('verify',p,mode)['status'],'passed');self.assertEqual(m.run('coverage',p,mode)['participating_files'],mode)
    def test_wrong_environment(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p,1);f=p/'container.json';f.write_text(f.read_text().replace('TORCHINDUCTOR_ENABLE_PDL=1','TORCHINDUCTOR_ENABLE_PDL=0'))
            with self.assertRaises(ValueError):m.run('verify',p,1)
    def test_snapshot_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p,0);(p/'pdl_source_files/run_vllm_pdl.sh').write_text('changed')
            with self.assertRaises(ValueError):m.run('verify',p,0)
    def test_no_codegen(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p,1);(p/'serving/pdl_cache/inductor/code.py').write_text('# triton only')
            with self.assertRaises(ValueError):m.run('coverage',p,1)
    def test_cache_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p,0);f=p/'container.json';f.write_text(f.read_text().replace('/bench-results/pdl_cache/inductor','/shared/cache'))
            with self.assertRaises(ValueError):m.run('verify',p,0)
if __name__=='__main__':unittest.main()
