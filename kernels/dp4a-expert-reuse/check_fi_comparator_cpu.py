import ast,json,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent
source=p/'sources/moe_dispatch.py';tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='launch_sm120_moe')
assert 'input_scales_are_reciprocal' in [x.arg for x in node.args.kwonlyargs]
s=source.read_text();assert 'workspace.dm_input_gs.copy_(' in s and '1.0 / input_gs' in s
adapter=ast.parse((p/'fi_comparator.py').read_text());call=next(n for n in ast.walk(adapter) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='launch_sm120_moe');kw={x.arg:x.value for x in call.keywords};assert isinstance(kw['input_scales_are_reciprocal'],ast.Constant) and kw['input_scales_are_reciprocal'].value is True
# Exact current effective contract is covered by actual-checkpoint CPU report.
report=json.loads((p/'checkpoint_cpu.json').read_text());assert report['effective_scales_exact'] and report['cuda_initialized'] is False
print(json.dumps(dict(cpu_only=True,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),adapter_sha256=hashlib.sha256((p/'fi_comparator.py').read_bytes()).hexdigest(),explicit_reciprocal_flag=True,no_unit_scale_substitution=True,scope='AST ABI/flag and existing actual-checkpoint algebra audit; not CUDA correctness'),indent=2))
