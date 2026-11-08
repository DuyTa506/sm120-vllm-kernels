import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pin_bundle_binding as B
import symbolic_pickle as S
SOURCE="""@triton_heuristics.pointwise(size_hints={'x': 16}, triton_meta={'signature': {'p': '*fp32','XBLOCK': 'constexpr'}, 'constants': {}, 'configs': []})
@triton.jit
def kernel(p,XBLOCK:tl.constexpr):
    tl.extra.cuda.gdc_wait()
    tl.store(p,1)
"""
def symbol(tag,state):
 cls=type('Fixture',(S.Symbol,),{'tag':('torch._inductor.triton_bundler',tag)})
 out=cls();out.state=state;return out
class Tests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  p=self.root/'serving/pdl_cache/kernel.py';p.parent.mkdir(parents=True);p.write_text(SOURCE)
  self.key,_,_=B.source_identity(SOURCE);self.cfg={'XBLOCK':16,'num_warps':4,'num_stages':1,'num_ctas':1}
  self.events={'constructor_identity_keys':[self.key],'launched_configs':{self.key:[self.cfg]}}
  self.ref={self.key:[{'config':self.cfg}]}
  meta={k:self.cfg[k] for k in ('num_warps','num_stages','num_ctas')};meta.update(name='kernel',launch_pdl=True)
  payloads={'.source':b'#loc = loc("/bench-results/pdl_cache/kernel.py":1:0)\n', '.json':json.dumps(meta).encode(), '.ptx':b'griddepcontrol.wait;', '.cubin':b'fixture-binary', '.ttir':b'fixture-ir'}
  self.groups=[symbol('TritonKernelArtifacts',{'kernel_hash':'fixture-hash','artifacts':[symbol('TritonKernelArtifact',{'filename':'kernel'+ext,'payload':data}) for ext,data in payloads.items()]})]
  self.wrapper='kernel = async_compile.triton("kernel",'+repr(SOURCE)+')'
  self.wrappers={B.sha(self.wrapper.encode()):self.wrapper}
 def verify(self):return B.verify_bundle(self.root,self.groups,self.wrappers,self.events,self.ref,S)
 def test_complete(self):self.assertEqual(len(self.verify()[0]),1)
 def test_changed_body_same_warps(self):
  (self.root/'serving/pdl_cache/kernel.py').write_text(SOURCE.replace('tl.store(p,1)','tl.store(p,2)'))
  with self.assertRaisesRegex(ValueError,'bypassed constructor'):self.verify()
 def test_wrapper_not_actual_aot_digest(self):
  self.wrappers={'wrong':self.wrapper}
  with self.assertRaisesRegex(ValueError,'actual AOT digest'):self.verify()
 def test_differing_singleton(self):
  self.events=copy.deepcopy(self.events);self.events['launched_configs'][self.key][0]['XBLOCK']=32
  with self.assertRaisesRegex(ValueError,'singleton config differs'):self.verify()
 def test_missing_binding(self):
  self.groups=[]
  with self.assertRaisesRegex(ValueError,'missing model-bound bundle'):self.verify()
if __name__=='__main__':unittest.main()
