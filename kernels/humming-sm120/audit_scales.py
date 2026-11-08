"""CPU-only exact checkpoint calibrated-scale coverage; no vLLM/Humming imports."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
p=Path('/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/1c3f884bc99aac2524f6d49bcbac8c88401afd66')
idx=json.loads((p/'model.safetensors.index.json').read_text())['weight_map']
from contextlib import ExitStack
stack=ExitStack()
files={name:stack.enter_context(safe_open(p/name,framework="pt",device="cpu")) for name in set(idx.values())}
# Execute installed scale-conversion function bodies on CPU without importing
# device-initializing backend modules. No quantizer/kernel emulation.
import ast, hashlib
pkg=Path('/usr/local/lib/python3.12/dist-packages')
fi=pkg/'vllm/model_executor/layers/quantization/utils/quant_utils.py'
hm=pkg/'humming/schema/base.py'
def extract(path,name):
    tree=ast.parse(path.read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[]
    module=ast.Module(body=[node],type_ignores=[])
    scope={'torch':torch}
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),scope)
    return scope[name]
fi_amax=extract(fi,'amax_for_moe_activation_quant')
hm_scale=extract(hm,'_convert_static_tensor_scale')
results=[]
contracts=[]
for layer in range(40):
    for kind in ['input_global_scale','weight_global_scale']:
        values={}
        for proj in ['gate_proj','up_proj','down_proj']:
            tensors=[]
            for expert in range(256):
                name=f'model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.{kind}'
                tensors.append(files[idx[name]].get_tensor(name).float().reshape(-1))
            v=torch.cat(tensors);values[proj]=v
            if kind=='input_global_scale':
                baseline=fi_amax(v.reciprocal(),False).reshape(1)
                candidate=hm_scale({'input_global_scale':v},'input_global_scale','input_scale_2',reciprocal=True)['input_scale_2']
                assert torch.equal(baseline,candidate)
                contracts.append(dict(layer=layer,projection=proj,exact=True,effective_dequant_scale=float(baseline.item())))
            results.append(dict(layer=layer,kind=kind,projection=proj,count=v.numel(),unique=v.unique().numel(),min=float(v.min()),max=float(v.max())))
        results.append(dict(layer=layer,kind=kind,gate_up_exact_equal=torch.equal(values['gate_proj'],values['up_proj'])))
print(json.dumps(dict(cuda_initialized=torch.cuda.is_initialized(),results=results,contracts=contracts,source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (fi,hm)}),indent=2))
