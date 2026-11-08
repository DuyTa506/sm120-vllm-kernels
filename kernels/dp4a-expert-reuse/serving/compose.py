"""Generate reviewable private overlay; never edit installed sources."""
import ast,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent;experiment=root.parent;worktree=root.parents[1]
generated=root/'generated';exp=Path('vllm/model_executor/layers/fused_moe/experts')
inputs={exp/'flashinfer_b12x_moe.py':worktree/'moe-nvfp4/scalar'/exp/'flashinfer_b12x_moe.py',
 Path('vllm/envs.py'):worktree/'humming-sm120/serving/source/envs.py'}
source={}
for dest,path in inputs.items():
 text=path.read_text();source[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 if dest.name=='envs.py':
  text+='\nenvironment_variables.update({"VLLM_SM120_FI_DIRECT_SMALL_M": lambda: os.getenv("VLLM_SM120_FI_DIRECT_SMALL_M", "0") == "1"})\n'
 else:
  load_needle='    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n'
  assert text.count(load_needle)==1
  text=text.replace(load_needle,load_needle+'        from .sm120_fi_direct import runtime as fi_direct_runtime\n        fi_direct_runtime.bind_model_identity(self)\n')
  needle='        assert self.w1_scale is not None and self.w2_scale is not None, ('
  assert text.count(needle)==1
  text=text.replace(needle,'''        from .sm120_fi_direct import runtime as fi_direct_runtime
        if fi_direct_runtime.try_apply(self, output, hidden_states, w1, w2,
                topk_weights, topk_ids, activation, global_num_experts, expert_map,
                apply_router_weight_on_input):
            return
'''+needle)
 target=generated/dest;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text);ast.parse(text)
pkg=generated/exp/'sm120_fi_direct';pkg.mkdir(parents=True,exist_ok=True);(pkg/'__init__.py').write_text('')
for name in ['runtime.py','model_identity.py','fi_plan.py','direct_native.py','direct_sanitize.py']:
 path=root/name if name in ('runtime.py','model_identity.py') else experiment/name
 text=path.read_text();source[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 text=text.replace('from fi_plan import','from .fi_plan import').replace('from direct_native import','from .direct_native import')
 (pkg/name).write_text(text);ast.parse(text)
manifest=dict(scope='CPU-only experimental scalar overlay M1/M4, default off; no serving claim',source=source,
 generated={str(p.relative_to(generated)):hashlib.sha256(p.read_bytes()).hexdigest() for p in generated.rglob('*.py')})
(generated/'composition.json').write_text(json.dumps(manifest,indent=2))
print(generated)
