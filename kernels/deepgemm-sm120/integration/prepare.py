"""Compose DG with isolated skinny round2 source; does not mutate installed vLLM."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import shutil
D=Path(__file__).resolve().parent
BASE=D.parents[1]/'decode-roofline/round2/generated'
DEST=D/'generated'
DEST.mkdir(exist_ok=True)
rel=Path('vllm/model_executor/kernels/linear/scaled_mm')
files=[Path('vllm/envs.py'),rel/'cutlass.py',rel/'sm120_skinny_fp8.py',rel/'sm120_skinny_fp8.cu']
manifest={}
patch=[]
for name in files:
    source=BASE/name
    raw=source.read_bytes();manifest[str(name)]=hashlib.sha256(raw).hexdigest()
    target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    if name==Path('vllm/envs.py'):
        text=raw.decode()+'''\n# DG rowwise candidate flag participates in compile_factors().
environment_variables.update({
    "VLLM_SM120_DG_ROWWISE_FP8": lambda: os.getenv("VLLM_SM120_DG_ROWWISE_FP8", "0") == "1",
})
'''
    elif name==rel/'cutlass.py':
        text=raw.decode()
        needle='        self.use_sm120_skinny = self._sm120_skinny_ok(layer, weight)'
        assert text.count(needle)==1
        text=text.replace(needle,needle+'''
        from . import sm120_dg_rowwise_fp8
        weight_scale = getattr(layer, weight_scale_name, None)
        self.use_sm120_dg_rowwise = (
            sm120_dg_rowwise_fp8.ENABLED
            and self.config.activation_quant_key.scale.group_shape.is_per_token()
            and not self.config.activation_quant_key.scale.static
            and self.config.out_dtype in (None, torch.bfloat16)
            and sm120_dg_rowwise_fp8.prepare_for_weight(weight, weight_scale)
        )''')
        needle='''        if (
            getattr(self, "use_sm120_skinny", False)'''
        assert text.count(needle)==1
        text=text.replace(needle,'''        if (
            getattr(self, "use_sm120_dg_rowwise", False)
            and bias is None
            and out_dtype == torch.bfloat16
        ):
            output = torch.ops.vllm.sm120_dg_rowwise_fp8_mm(
                A, B, As, Bs, getattr(self, "use_sm120_skinny", False)
            )
            return output.view(*output_shape[:-1], B.shape[1])

'''+needle)
    else:continue
    ast.parse(text);target.write_text(text)
    patch.extend(difflib.unified_diff(raw.decode().splitlines(True),text.splitlines(True),'a/'+str(name),'b/'+str(name)))
shutil.copy2(D/'sm120_dg_rowwise_fp8.py',DEST/rel/'sm120_dg_rowwise_fp8.py')
(D/'source_manifest.json').write_text(json.dumps({'base':'isolated decode-roofline/round2 generated source; not active serving proof','base_sha256':manifest},indent=2)+'\n')
(D/'deepgemm_rowwise_vllm.patch').write_text(''.join(patch))
ast.parse((D/'sm120_dg_rowwise_fp8.py').read_text())
print('Prepared isolated integration at',DEST)

(D/'generated/composition.json').write_text(json.dumps({'generated_sha256':{str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.rglob('*') if p.is_file() and p.name != 'composition.json'}},indent=2)+'\n')
