"""Quantify E4M3 rounding differences on real captured input, CPU only."""
import json,hashlib
from pathlib import Path
import torch
from safetensors import safe_open
torch.set_num_threads(4)
p=Path('/work/kernels/moe-nvfp4/data/moe_inputs.pt')
x=torch.load(p,map_location='cpu',weights_only=True)['hidden'][19][:1024].float().reshape(1024,-1,16)
model=Path('/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/1c3f884bc99aac2524f6d49bcbac8c88401afd66')
index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];name='model.language_model.layers.19.mlp.experts.0.gate_proj.input_global_scale'
with safe_open(model/index[name],framework='pt',device='cpu') as f:cal=f.get_tensor(name).float()
scale=x.abs().amax(-1)*cal/6
rn=scale.clamp(max=448).to(torch.float8_e4m3fn)
ru_bytes=rn.view(torch.uint8).clone();increment=(rn.float()<scale)&(ru_bytes<126);ru_bytes+=increment.to(torch.uint8)
ru=ru_bytes.view(torch.float8_e4m3fn)
print(json.dumps(dict(layer=19,tokens=1024,blocks=scale.numel(),calibration_multiplier=float(cal.item()),scale_bytes_changed=int(increment.sum()),changed_fraction=float(increment.float().mean()),max_scale_relative_change=float(((ru.float()-rn.float())/rn.float().clamp_min(1e-12)).max()),cuda_initialized=torch.cuda.is_initialized(),capture_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='CPU E4M3 scale RN versus upward increment; not a GPU payload or output accuracy result'),indent=2))
