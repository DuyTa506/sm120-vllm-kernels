"""Real package-relative import test with CUDA visibility disabled before imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util
import json
from pathlib import Path
import sys
import torch
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dynamic_kernel as factory

name=factory.MoEGatedDynamicKernel.__module__.rsplit('.',1)[0]+'.gated_long_first_experiment'
spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name('gated_long_first.py'))
module=importlib.util.module_from_spec(spec)
sys.modules[name]=module
spec.loader.exec_module(module)
assert module.MoEGatedDynamicKernelLongFirst.__module__==name
assert not torch.cuda.is_initialized()
print(json.dumps(dict(module=name,package=module.__package__,cuda_initialized=False)))
