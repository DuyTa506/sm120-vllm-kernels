"""Validate actual installed Worker/load API plus real checkpoint HF config, CPU only."""
import os,inspect,ast,textwrap,glob,json,hashlib
from pathlib import Path
from types import SimpleNamespace as S
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from vllm.v1.worker.gpu_worker import Worker
from transformers import AutoConfig
from model_identity import bind,require
assert not torch.cuda.is_initialized()
source=textwrap.dedent(inspect.getsource(Worker.load_model));tree=ast.parse(source)
with_nodes=[n for n in ast.walk(tree) if isinstance(n,ast.With)]
assert any(any(isinstance(i.context_expr,ast.Call) and isinstance(i.context_expr.func,ast.Name) and i.context_expr.func.id=='set_current_vllm_config' and ast.unparse(i.context_expr.args[0])=='self.vllm_config' for i in n.items) and any(isinstance(c,ast.Call) and ast.unparse(c.func)=='self.model_runner.load_model' for stmt in n.body for c in ast.walk(stmt)) for n in with_nodes)
path=Path(glob.glob('/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/*/config.json')[0])
hf=AutoConfig.from_pretrained(str(path.parent),local_files_only=True)
e=S();bind(e,True,lambda:S(model_config=S(model=str(path.parent),hf_config=hf)))
identity=dict(require(e));assert identity['dimensions']==(2048,512,256,8)
p=Path(__file__).parent
runtime=ast.parse((p/'generated/vllm/model_executor/layers/fused_moe/experts/sm120_fi_direct/runtime.py').read_text())
forward=next(n for n in runtime.body if isinstance(n,ast.FunctionDef) and n.name=='try_apply')
assert 'get_current_vllm_config' not in ast.unparse(forward)
expert=ast.parse((p/'generated/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py').read_text())
hook=next(n for n in ast.walk(expert) if isinstance(n,ast.FunctionDef) and n.name=='process_weights_after_loading')
assert 'bind_model_identity(self)' in ast.unparse(hook)
assert not torch.cuda.is_initialized()
print(json.dumps(dict(cuda_initialized=False,worker_load_real_context=True,worker_source_sha256=hashlib.sha256(Path(inspect.getfile(Worker)).read_bytes()).hexdigest(),checkpoint_config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),identity=identity,forward_config_lookup=False),indent=2))
