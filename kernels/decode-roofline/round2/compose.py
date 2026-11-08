"""Isolated skinny compositions; no numerical kernel edits."""
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'generated'
if OUT.exists(): raise SystemExit('preserve existing generated sources; use a new revision')
paths={
 'vllm/envs.py':ROOT/'kernels/gdn-replay/patch/a/vllm/envs.py',
 'vllm/model_executor/layers/utils.py':ROOT/'kernels/decode-roofline/vllm/model_executor/layers/utils.py',
 'vllm/model_executor/layers/skinny_bf16_gemm.py':ROOT/'kernels/decode-roofline/vllm/model_executor/layers/skinny_bf16_gemm.py',
}
fp8=ROOT/'kernels/dense-fp8-gemm/patch/b/vllm/model_executor/kernels/linear/scaled_mm'
for name in ('cutlass.py','sm120_skinny_fp8.py','sm120_skinny_fp8.cu'):
 paths['vllm/model_executor/kernels/linear/scaled_mm/'+name]=fp8/name
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,source in paths.items():
 target=OUT/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
p=OUT/'vllm/envs.py'
s=p.read_text()
assert '"VLLM_SKINNY_BF16_GEMM"' not in s and '"VLLM_SM120_SKINNY_FP8"' not in s
s+='''
# Round2 skinny options: registered defaults participate in compile_factors().
environment_variables.update({
    "VLLM_SKINNY_BF16_GEMM": lambda: os.getenv("VLLM_SKINNY_BF16_GEMM", "0") == "1",
    "VLLM_SM120_SKINNY_FP8": lambda: os.getenv("VLLM_SM120_SKINNY_FP8", "0") == "1",
    "VLLM_SM120_SKINNY_FP8_MAX_M": lambda: int(os.getenv("VLLM_SM120_SKINNY_FP8_MAX_M", "64")),
})
''';p.write_text(s)
def replace(path,before,after):
 s=path.read_text();assert s.count(before)==1,(path,before);path.write_text(s.replace(before,after,1))
replace(OUT/'vllm/model_executor/layers/utils.py','_SKINNY_BF16_GEMM = os.environ.get("VLLM_SKINNY_BF16_GEMM", "0") == "1"','_SKINNY_BF16_GEMM = envs.VLLM_SKINNY_BF16_GEMM')
p=OUT/'vllm/model_executor/kernels/linear/scaled_mm/sm120_skinny_fp8.py'
replace(p,'from vllm.logger import init_logger','from vllm import envs\nfrom vllm.logger import init_logger')
replace(p,'ENABLED = os.environ.get("VLLM_SM120_SKINNY_FP8", "0") == "1"','ENABLED = envs.VLLM_SM120_SKINNY_FP8')
replace(p,'MAX_M = int(os.environ.get("VLLM_SM120_SKINNY_FP8_MAX_M", "64"))','MAX_M = envs.VLLM_SM120_SKINNY_FP8_MAX_M')
manifest=dict(input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths.values()},
 generated_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()},
 scope='cache-key registration only; original BF16/FP8 numerical kernels and dispatch predicates retained')
(OUT/'composition.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
