"""Validate actual vLLM single-rank context requirement without GPU imports."""
import ast,json
from pathlib import Path
import torch
from vllm.forward_context import ForwardContext,override_forward_context,get_forward_context,is_forward_context_available
source=Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py')
method=next(n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='get_global_valid_shape_m')
ns=dict(get_forward_context=get_forward_context,torch=torch)
exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source),'exec'),ns)
ids=torch.zeros((1024,8),dtype=torch.int32)
assert not is_forward_context_available()
with override_forward_context(ForwardContext(no_compile_layers={},attn_metadata={},slot_mapping={})):
    assert ns['get_global_valid_shape_m'](None,ids)==8192
assert not is_forward_context_available()
try:
    with override_forward_context(ForwardContext(no_compile_layers={},attn_metadata={},slot_mapping={})):
        raise RuntimeError('fixture')
except RuntimeError:pass
assert not is_forward_context_available()
print(json.dumps(dict(valid_shape_m=8192,context_restored=True,cuda_initialized=torch.cuda.is_initialized())))
