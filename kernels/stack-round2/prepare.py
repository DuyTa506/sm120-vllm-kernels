"""Build immutable source bundles; no CUDA, installed-package edits or compilation."""
import argparse, ast, hashlib, json, shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FLAGS=('VLLM_SM120_DG_ROWWISE_FP8','VLLM_SM120_SKINNY_FP8','VLLM_SKINNY_BF16_GEMM','VLLM_SM120_TASK_ORDER_LONG_FIRST')
PROFILES={'parent':(0,0,0,0),'task_order':(0,0,0,1),'dense':(1,1,1,0),'combined':(1,1,1,1)}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare(profile,dest):
    assert profile in PROFILES
    assert not dest.exists(), f'Output exists: {dest}'
    dest.mkdir(parents=True)
    inputs={};outputs={}
    def copy(src,rel):
        src=ROOT/src;p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,p)
        inputs[str(src.relative_to(ROOT))]=sha(src)
        return p
    dg=Path('kernels/deepgemm-sm120/integration');base=Path('kernels/decode-roofline/round2/generated')
    comp=json.loads((ROOT/dg/'generated/composition.json').read_text())['generated_sha256']
    for rel,expected in comp.items():
        assert sha(ROOT/dg/'generated'/rel)==expected, f'Stale DG composition: {rel}'
        copy(dg/'generated'/rel,rel)
    for rel in ['vllm/model_executor/layers/utils.py','vllm/model_executor/layers/skinny_bf16_gemm.py']:
        copy(base/rel,rel)
    for rel in ['v1/attention/backends/flashinfer.py','v1/attention/backends/sm120_fp8_fmha_glue.py','model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py','model_executor/layers/mamba/ops/fused_conv_post_conv.py']:
        copy(Path('build/vllm')/rel,Path('vllm')/rel)
    for variant,rel in [('scalar','model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py'),('patched','model_executor/layers/fused_moe/oracle/unquantized.py')]:
        copy(Path('kernels/moe-nvfp4')/variant/'vllm'/rel,Path('vllm')/rel)
    for file in ['gated_long_first.py','native_source_manifest.json']:
        copy(Path('kernels/moe-task-order-serving')/file,Path('plugin')/file)
    copy(dg/'fixed_pdl_policy.py','plugin/fixed_pdl_policy.py')
    copy(Path('kernels/stack-round2/stack_plugin.py'),'plugin/stack_plugin.py')
    for f in ['qwen_fi_control_plugin.py','autotune_import.py','references/next_cpu_audit_autotune.json','qwen_fi_control-0.1.dist-info/METADATA','qwen_fi_control-0.1.dist-info/entry_points.txt']:
        copy(Path('kernels/decode-route-capture')/f,Path('fi')/f)
    copy(Path('scripts/cudagraph_sizes.txt'),'cudagraph_sizes.txt')
    env=dest/'vllm/envs.py'
    text=env.read_text()+'''\n# Stack identity and scheduler policy must participate in compile_factors().
environment_variables.update({
    "VLLM_SM120_TASK_ORDER_LONG_FIRST": lambda: os.getenv("VLLM_SM120_TASK_ORDER_LONG_FIRST", "0") == "1",
    "VLLM_SM120_STACK_SOURCE_ID": lambda: os.getenv("VLLM_SM120_STACK_SOURCE_ID", ""),
})
'''
    for key in ('VLLM_B12X_FIX_SCALES','VLLM_B12X_SHARED_WORKSPACE','VLLM_B12X_SCALAR_INPUT_SCALE','VLLM_SM120_FP8_PREFILL','VLLM_SM120_FP8_PREFILL_NOGLUE','VLLM_FI_KERNEL_PAGES','VLLM_GDN_INTERNAL_CKPT','VLLM_GDN_FUSED_CONV','TORCHINDUCTOR_ENABLE_PDL'):
        text += f'\nenvironment_variables[{key!r}] = lambda: os.getenv({key!r}, "0")\n'
    env.write_text(text)
    for source in ('prepare.py','serve.py','cpu_check.py'):
        copy(Path('kernels/stack-round2')/source,Path('tools')/source)
    info=dest/'plugin/stack_round2-0.1.dist-info';info.mkdir()
    (info/'METADATA').write_text('Metadata-Version: 2.1\nName: stack-round2\nVersion: 0.1\n')
    (info/'entry_points.txt').write_text('[vllm.general_plugins]\nstack_round2 = stack_plugin:register\n')
    for p in dest.rglob('*.py'):ast.parse(p.read_text())
    # Bind the external read-only native DG tree, including binary/JIT headers.
    src=ROOT/'kernels/deepgemm-sm120/fused_src'
    dg_files={str(p.relative_to(src)):sha(p) for p in sorted(src.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pyc') and '.git' not in p.parts}
    for p in sorted(dest.rglob('*')):
        if p.is_file():outputs[str(p.relative_to(dest))]=sha(p)
    manifest={'schema':1,'profile':profile,'config':'next','variant':'scalar','flags':dict(zip(FLAGS,map(str,PROFILES[profile]))),'inputs':inputs,'outputs':outputs,'dg_source_root':str(src.relative_to(ROOT)),'dg_files':dg_files,'dg_pdl':False,'torch_inductor_pdl':'explicit independent launch parameter; default 0','pending':['Humming per-shape adapter','GDN replay preemption gate','dynamic MTP'],'performance_status':'unmeasured experimental composition'}
    manifest['source_id']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'bundle':str(dest),'profile':profile,'source_id':manifest['source_id'],'files':len(outputs),'dg_files':len(dg_files)}))
    return manifest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',choices=PROFILES,default='parent');p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.profile,a.output.resolve())
