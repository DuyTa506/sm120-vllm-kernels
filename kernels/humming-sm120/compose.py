"""Isolated exact NVFP4 activation schema adapter; no installed edits."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
source=root/'source/nvfp4.py';s=source.read_text()
needle='        convert_to_humming_moe_kernel_format(layer, quant_config=quant_config)'
assert s.count(needle)==1
s=s.replace(needle,'''        # Match this checkpoint's calibrated local-dynamic NVFP4 input scheme.
        # Its reciprocal-then-amax conversion matches the existing FI backend's
        # global activation reduction; do not substitute a unit/dynamic scale.
        if not hasattr(layer, "w13_input_global_scale"):
            raise ValueError("isolated Humming experiment requires CT input globals")
        from humming import dtypes as humming_dtypes
        from humming.schema.compressed_tensors import CompressedTensorsInputSchema
        input_schema = CompressedTensorsInputSchema(
            format="nvfp4-pack-quantized", type="float", num_bits=4,
            dynamic="local", group_size=16, symmetric=True,
        )
        expected_input_scales = {
            name: getattr(layer, name + "_input_global_scale").float().reciprocal().amax().reshape(1)
            for name in ("w13", "w2")
        }
        convert_to_humming_moe_kernel_format(
            layer, quant_config=quant_config, input_schema=input_schema,
            allow_input_schema_fallback=False,
        )
        for name, expected in expected_input_scales.items():
            actual = getattr(layer, name + "_input_scale_2")
            if not torch.equal(actual, expected):
                raise ValueError("Humming activation calibration changed: " + name)
            schema = layer.input_schemas[name]
            if (schema.a_dtype != humming_dtypes.float4e2m1
                    or schema.input_scale_group_size != 16
                    or schema.input_quant_mode.value != "static_tensor_dynamic_group"):
                raise ValueError("Humming activation schema changed: " + name)''')
out=root/'generated/nvfp4.py';out.parent.mkdir(exist_ok=True);out.write_text(s)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(root/'generated/manifest.json').write_text(json.dumps(dict(source_sha256=sha(source),generated_sha256=sha(out),purpose='explicit CT NVFP4 calibrated activation schema; baseline FI global-scale aggregation'),indent=2))
