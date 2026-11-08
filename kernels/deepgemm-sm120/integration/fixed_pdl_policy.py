"""Experimental matched PDL=false policy installed before worker model load."""
import sys

def install_fixed_pdl():
    from vllm.utils import deep_gemm as vllm_dg
    if not getattr(vllm_dg._apply_pdl,'_dg_fixed_false',False):
        original=vllm_dg._apply_pdl
        def apply_fixed_false(module,enable=True):
            original(module,False)
            assert module.get_pdl() is False
        apply_fixed_false._dg_fixed_false=True
        vllm_dg._apply_pdl=apply_fixed_false
    # Handle an already imported module without importing/initializing one here.
    module=sys.modules.get('deep_gemm')
    if module is not None:
        module.set_pdl(False)
        assert module.get_pdl() is False
