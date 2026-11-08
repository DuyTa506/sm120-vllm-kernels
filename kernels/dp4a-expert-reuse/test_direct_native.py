import unittest
from types import SimpleNamespace as S
from direct_native import DirectNative
class Tensor:
 shape=(3,2048);device='cpu';dtype='int32'
 def view(self,*shape):return self
class Tests(unittest.TestCase):
 def test_explicit_launch_no_dispatch_mutation(self):
  calls=[];ids=Tensor();ids.shape=(3,8);x=Tensor();o=Tensor()
  ws=S(**{n:object() for n in ['dm_intermediate','dm_barrier_count','dm_barrier_epoch']})
  wv=S(**{n:object() for n in ['w1_storage','w1_scale_storage','w1_alpha','w2_storage','w2_scale_storage','w2_alpha']})
  dispatch=S(_FORCED_BACKEND='untouched',_get_direct_micro_kernel=lambda *a,**k:('compiled',11,True),MoEDirectMicroKernel=S(launch=lambda c,**k:calls.append((c,k))))
  p=S(direct_multiplier=True,forced='direct_micro',kw=dict(a=x,topk_ids=ids,topk_weights=Tensor(),scatter_output=o),workspace=ws,views=wv,a1=object(),a2=object())
  native=DirectNative(dispatch,p)
  self.assertIs(native(),o);self.assertEqual(dispatch._FORCED_BACKEND,'untouched')
  self.assertIs(calls[0][1]['w1_alphas'],wv.w1_alpha);self.assertIs(calls[0][1]['a1_gscale'],p.a1)
  self.assertIs(calls[0][1]['barrier_epoch'],ws.dm_barrier_epoch)
  other=Tensor();self.assertIs(native(out=other),other)
  self.assertIs(calls[-1][1]['out'],other);self.assertIs(native.kw['out'],o)
  native.kw.pop('out');self.assertIs(native(out=other),other)
  self.assertNotIn('out',native.kw)
 def test_reciprocal_rejected(self):
  with self.assertRaises(ValueError):DirectNative(None,S(direct_multiplier=False,forced='direct_micro'))
if __name__=='__main__':unittest.main()
