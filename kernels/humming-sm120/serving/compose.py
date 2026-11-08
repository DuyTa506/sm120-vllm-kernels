"""Generate isolated source overlay, never edit installed vLLM."""
from pathlib import Path
import ast,json,hashlib,shutil
p=Path(__file__).resolve().parent;g=p/'generated';g.mkdir(exist_ok=True)
exp=Path('vllm/model_executor/layers/fused_moe/experts');oracle=Path('vllm/model_executor/layers/fused_moe/oracle/nvfp4.py')
inputs={oracle:p/'source/nvfp4.py',Path('vllm/envs.py'):p/'source/envs.py',exp/'flashinfer_b12x_moe.py':p.parents[1]/'moe-nvfp4/scalar/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py'}
manifest={}
for dest,src in inputs.items():
 text=src.read_text();manifest[str(src)]=hashlib.sha256(src.read_bytes()).hexdigest()
 if dest==oracle:
  needle='    use_a16 = _use_a16(nvfp4_backend, use_a16)';assert text.count(needle)==1
  text=text.replace(needle,'''    from vllm.model_executor.layers.fused_moe.experts import sm120_humming_m1024
    if sm120_humming_m1024.ENABLED:
        assert nvfp4_backend == NvFp4MoeBackend.FLASHINFER_B12X
        assert not use_a16 and is_act_and_mul
        sm120_humming_m1024.prepare(layer, w13, w13_scale, w2, w2_scale)
'''+needle)
 elif dest.name=='envs.py':
  text+='''\n# Both flags participate in vLLM compile_factors; OFF is default.
environment_variables.update({
    "VLLM_SM120_HUMMING_RN_M1024": lambda: os.getenv("VLLM_SM120_HUMMING_RN_M1024", "0") == "1",
    "VLLM_SM120_HUMMING_RN_PROBE": lambda: os.getenv("VLLM_SM120_HUMMING_RN_PROBE", "0") == "1",
})
'''
 else:
  needle='    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n';assert text.count(needle)==1
  text=text.replace(needle,needle+'        self._sm120_humming_state = getattr(layer, "_sm120_humming_state", None)\n')
  needle='        assert self.w1_scale is not None and self.w2_scale is not None, (';assert text.count(needle)==1
  text=text.replace(needle,'''        from . import sm120_humming_m1024
        if sm120_humming_m1024.try_apply(
            getattr(self, "_sm120_humming_state", None), output, hidden_states,
            topk_weights, topk_ids, activation, global_num_experts, expert_map,
            a1q_scale, apply_router_weight_on_input, expert_tokens_meta,
        ):
            return
'''+needle)
 out=g/dest;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(text);ast.parse(text)
for src,name in [('sm120_humming_m1024.py','sm120_humming_m1024.py'),('policy.py','sm120_humming_policy.py')]:shutil.copy2(p/src,g/exp/name)
shutil.copy2(p.parent/'rounding_mode.py',g/exp/'sm120_humming_rounding.py')
shutil.copytree(p.parent/'rounding',g/exp/'rounding',dirs_exist_ok=True)
for f in (g/exp).glob('sm120_humming*.py'):ast.parse(f.read_text())
record=dict(base_source_sha256=manifest,scope='isolated next+scalar overlay; M1024 only; not serving verified',generated_sha256={str(f.relative_to(g)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(g.rglob('*')) if f.is_file() and f.name!='composition.json'})
(g/'composition.json').write_text(json.dumps(record,indent=2)+'\n')
print('Prepared isolated overlay',g)
