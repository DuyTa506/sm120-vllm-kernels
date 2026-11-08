import importlib.util
from pathlib import Path
import pickle
import unittest
HERE=Path(__file__).resolve().parent
def module(name,file):
 s=importlib.util.spec_from_file_location(name,HERE/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=module('canonical','canonicalize_pin_record.py');S=module('symbols','symbolic_pickle.py')
class Protocol(unittest.TestCase):
 def test_intersection(self):
  c1={'XBLOCK':1};c2={'XBLOCK':2};events=[{'stage':'installed','torch_source_sha256':'a'},{'stage':'constructor','identity':'k','identity_evidence':{'x':1},'available':[c1,c2]},{'stage':'constructor','identity':'k','identity_evidence':{'x':1},'available':[c2]}]
  self.assertEqual(C.canonicalize(events)['reference']['k'][0]['config'],c2)
  events[-1]['available']=[{'XBLOCK':3}]
  with self.assertRaises(ValueError):C.canonicalize(events)
 def test_identity_collision(self):
  events=[{'stage':'installed','torch_source_sha256':'a'},{'stage':'constructor','identity':'k','identity_evidence':{'x':1},'available':[{'XBLOCK':1}]},{'stage':'constructor','identity':'k','identity_evidence':{'x':2},'available':[{'XBLOCK':1}]}]
  with self.assertRaises(ValueError):C.canonicalize(events)
 def test_pickle_globals_are_inert(self):
  class Payload:
   def __reduce__(self):return (eval,('1/0',))
  obj=S.load(pickle.dumps(Payload()))
  self.assertIsInstance(obj,S.Symbol);self.assertEqual(obj.tag,('builtins','eval'));self.assertEqual(obj.args,('1/0',))
 def test_external_pickle_reference_rejected(self):
  with self.assertRaises(ValueError):S.load(b'\x80\x02\x82\x01.')
if __name__=='__main__':unittest.main()
