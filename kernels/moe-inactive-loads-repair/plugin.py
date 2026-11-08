"""Explicit isolated off/on registration; source-qualified native cache, no queue hooks."""
import hashlib,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
def install(mode='off'):
 if mode not in ('off','on'):raise ValueError('mode must be off/on')
 from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dynamic_kernel as factory,moe_dispatch
 manifest=json.loads((HERE/'manifest.json').read_text())
 original=factory.MoEGatedDynamicKernel
 assert original.__name__=='MoEGatedDynamicKernel','already replaced native factory'
 path=Path(sys.modules[original.__module__].__file__)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['baseline_sha256']
 candidate=HERE/'gated_defined_fc2.py'
 assert hashlib.sha256(candidate.read_bytes()).hexdigest()==manifest['candidate_sha256']
 assert not moe_dispatch._DYNAMIC_KERNEL_CACHE,'native cache must be empty before registration'
 assert moe_dispatch._FORCED_BACKEND is None
 key=moe_dispatch._dynamic_kernel_cache_key
 sha=manifest['candidate_sha256' if mode=='on' else 'baseline_sha256']
 def qualified_key(*args,**kwargs):
  assert (kwargs['k'],kwargs['n'],kwargs['mma_tiler_mn'])==(2048,512,(128,128)),'unsupported experiment geometry'
  assert kwargs['activation']=='silu' and kwargs['quant_mode']=='nvfp4'
  base=key(*args,**kwargs);assert isinstance(base,tuple)
  return base+('defined_fc2_source_sha256',sha)
 moe_dispatch._dynamic_kernel_cache_key=qualified_key
 if mode=='on':
  name=original.__module__.rsplit('.',1)[0]+'.gated_defined_fc2_control'
  spec=importlib.util.spec_from_file_location(name,candidate);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
  factory.MoEGatedDynamicKernel=module.MoEGatedDynamicKernelDefinedFC2
 return dict(mode=mode,source_sha256=sha,class_name=factory.MoEGatedDynamicKernel.__name__)
