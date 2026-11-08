import copy,unittest
from cache_geometry import validate_preempt_geometry
class Geometry(unittest.TestCase):
    def good(self):
        full=dict(spec_class='FullAttentionSpec',spec_fields=dict(block_size=2176),max_request_resident_blocks=8)
        mamba=dict(spec_class='MambaSpec',spec_fields=dict(block_size=2176,mamba_cache_mode='align',num_speculative_blocks=2,num_prefill_checkpoint_blocks=1),max_request_resident_blocks=5)
        return dict(num_blocks=50,max_model_len=16384,groups=[full]+[copy.deepcopy(mamba) for _ in range(3)])
    def test_expected_geometry(self):validate_preempt_geometry(self.good())
    def test_changed_geometry_fails(self):
        for field,value in [('num_blocks',64),('max_model_len',131072)]:
            data=self.good();data[field]=value
            with self.assertRaises(AssertionError):validate_preempt_geometry(data)
        data=self.good();data['groups'][1]['spec_fields']['num_prefill_checkpoint_blocks']=0
        with self.assertRaises(AssertionError):validate_preempt_geometry(data)
