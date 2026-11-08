"""Run installed sampler bodies on CPU to verify fixed CLI boundary lengths."""
import ast, hashlib, json, logging, math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
p=Path('/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/datasets/utils.py')
tree=ast.parse(p.read_text())
names={'_resolve_range_ratios','get_sampling_params'}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert len(nodes)==2
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
scope=dict(np=np,math=math,logger=logging.getLogger(__name__))
exec(compile(ast.fix_missing_locations(module),str(p),'exec'),scope)
tokenizer=SimpleNamespace(num_special_tokens_to_add=lambda:0,vocab_size=248320)
a,b,_=scope['get_sampling_params'](np.random.default_rng(7731),32,0,128,2304,tokenizer)
assert (a==128).all() and (b==2304).all()
growth_a,growth_b,_=scope['get_sampling_params'](np.random.default_rng(7731),8,0,128,6656,tokenizer)
assert (growth_a==128).all() and (growth_b==6656).all()
assert 128+6656+3<=16384
try:scope['get_sampling_params'](np.random.default_rng(7731),32,1,128,2304,tokenizer)
except ValueError: pass
else:raise AssertionError('ratio1 unexpectedly accepted')
print(json.dumps(dict(input_lengths=a.tolist(),output_lengths=b.tolist(),growth_input_lengths=growth_a.tolist(),growth_output_lengths=growth_b.tolist(),cuda_initialized=torch.cuda.is_initialized(),sampler_sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
