"""Pass audit_model to llm.apply_model after loading; no new inference/GPU job."""
def audit_model(model):
    import hashlib
    import inspect
    from pathlib import Path
    from vllm import envs
    from vllm.model_executor.kernels.linear.scaled_mm import sm120_dg_rowwise_fp8 as dg
    modules=[]
    for name,layer in model.named_modules():
        if 'linear_attn' not in name or not name.endswith('in_proj_qkvz'):
            continue
        kernel=getattr(getattr(layer,'scheme',None),'fp8_linear',None)
        weight=getattr(layer,'weight',None)
        if weight is None:continue
        scale_name=kernel.layer_param_names[1] if kernel is not None else None
        scale=getattr(layer,scale_name,None) if scale_name is not None else None
        modules.append({'module':name,'class':type(layer).__module__+'.'+type(layer).__name__,
                        'kernel_class':type(kernel).__module__+'.'+type(kernel).__name__,
                        'weight_shape':list(weight.shape),'weight_stride':list(weight.stride()),
                        'weight_dtype':str(weight.dtype),'weight_pointer':weight.data_ptr(),
                        'scale_name':scale_name,'scale_shape':list(scale.shape) if scale is not None else None,
                        'scale_dtype':str(scale.dtype) if scale is not None else None,
                        'dg_enabled':getattr(kernel,'use_sm120_dg_rowwise',False),
                        'skinny_enabled':getattr(kernel,'use_sm120_skinny',False)})
    source=Path(inspect.getsourcefile(dg))
    result={'flag':envs.VLLM_SM120_DG_ROWWISE_FP8,'modules':modules,
            'matched_qkvz_modules':len(modules),'enabled_qkvz_modules':sum(x['dg_enabled'] for x in modules),
            'verified_m':sorted(dg.VERIFIED_M),'dispatch_audit':dg.dispatch_audit(),
            'dispatch_audit_scope':'Python eager/capture dispatch only; live GPU trace needed for graph replay proof',
            'helper_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'prepared':{str(device):{'a_m':sorted(a),'b_shape':list(b.shape),'b_stride':list(b.stride()),
                                      'a_pointers':{str(m):tensor.data_ptr() for m,tensor in a.items()},
                                      'b_pointer':b.data_ptr()} for device,(a,b) in dg._READY.items()}}
    import sys
    module=sys.modules.get('deep_gemm')
    assert module is not None, 'Expected initialized external DG runtime'
    result['deepgemm_pdl']=module.get_pdl()
    assert result['deepgemm_pdl'] is False, 'Unexpected global DeepGEMM PDL policy'
    extension=Path(module._C.__file__)
    result['deepgemm_extension_sha256']=hashlib.sha256(extension.read_bytes()).hexdigest()
    assert len(modules)==30,'Unexpected Qwen GDN module count'
    assert result['enabled_qkvz_modules']==(30 if result['flag'] else 0),'DG layer activation mismatch'
    return result
