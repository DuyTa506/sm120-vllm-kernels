import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,sys
from pathlib import Path
import torch
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dynamic_kernel as factory
for variant in ('baseline','long_first'):
    name=factory.MoEGatedDynamicKernel.__module__.rsplit('.',1)[0]+'.diagnostic_import_'+variant
    path=Path(__file__).with_name('gated_'+variant+'_diagnostic.py')
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    print(json.dumps(dict(variant=variant,source=str(path),imported=True)))
assert not torch.cuda.is_initialized()
