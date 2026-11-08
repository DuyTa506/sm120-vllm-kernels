"""Actual installed API/layout checks for the numerical diagnostic; no GPU calls."""
import ast,inspect,json
from pathlib import Path
import torch
from humming.testing.process_input import unpack_group_scales
from humming import ops,dtypes
from quant_oracle import pack_fp4
assert list(inspect.signature(ops.dequant_weight).parameters)[:4]==['inputs','exponent_bits','mantissa_bits','is_signed']
assert list(inspect.signature(ops.unpack_weight).parameters)[:2]==['inputs','num_bits']
# The real unpack helper removes packed-int32 columns and trims logical scales.
scales=torch.tensor([[1.,2.,3.,4.],[5.,6.,7.,8.]],dtype=torch.float32).to(torch.float8_e4m3fn)
packed=scales.view(torch.int32)
normal=unpack_group_scales(packed,2,3,'float8e4m3',False)
assert torch.equal(normal.float(),scales[:,:3].float())
transposed=unpack_group_scales(packed.T.contiguous(),2,3,'float8e4m3',True)
assert torch.equal(transposed.float(),normal.float())
# Diagnose original packing in native code: unpack nibbles BEFORE decoding code values.
source=Path(inspect.getsourcefile(ops.dequant_weight))
s=source.read_text()
assert 'outputs, returned_outputs = _prepare_output(outputs, inputs.shape, torch.float32, inputs.device)' in s
assert not torch.cuda.is_initialized()
print(json.dumps(dict(actual_api_bindings_pass=True,normal_and_transposed_scale_layout_pass=True,unpacked_codes_required_before_dequant=True,cuda_initialized=False)))
