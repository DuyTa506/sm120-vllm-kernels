"""CPU test of late and already-imported DG initialization policy in both profiles."""
import sys,types
from unittest.mock import patch
from fixed_pdl_policy import install_fixed_pdl
for profile in ('off','on'):
    state={'pdl':True};calls=[]
    dg=types.SimpleNamespace(set_pdl=lambda value:state.update(pdl=value),get_pdl=lambda:state['pdl'])
    def original(module,enable=True):calls.append(enable);module.set_pdl(enable)
    utility=types.SimpleNamespace(_apply_pdl=original)
    vllm_utils=types.ModuleType('vllm.utils');vllm_utils.deep_gemm=utility
    vllm=types.ModuleType('vllm');vllm.utils=vllm_utils
    with patch.dict(sys.modules,{'vllm':vllm,'vllm.utils':vllm_utils,'deep_gemm':dg}):
        install_fixed_pdl();assert not dg.get_pdl()
        first=utility._apply_pdl;install_fixed_pdl();assert utility._apply_pdl is first
        utility._apply_pdl(dg,True);assert calls==[False] and not dg.get_pdl()
print('fixed PDL false survives eager and late initialization; idempotent in both profiles')
