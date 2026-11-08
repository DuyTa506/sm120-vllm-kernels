"""CPU regression using the exact failed QK/rope Triton constructor metadata.
Run inside image with CUDA_VISIBLE_DEVICES= and pin bootstrap in record mode.
"""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import triton
import triton.language as tl
from torch._inductor.runtime.hints import DeviceProperties,HeuristicType
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
import pin_inductor_core as core

source=Path('/work/kernels/pdl-swapab/results/pdl_record0_1/serving/pdl_cache/vllm/torch_compile_cache/torch_aot_compile/19c255ff3d98f916396cff81fba8211228b9ec93cb434d1e259331869617bc63/inductor_cache/24/c247c2ishurycmrdvueznhmcbr5ogwoololuiyq2oaw5sqnlqwvw.py')
assert not torch.cuda.is_initialized()
tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
dec=next(n for n in fn.decorator_list if isinstance(n,ast.Call) and ast.unparse(n.func)=='triton_heuristics.user_autotune')
def literal(node):
    if isinstance(node,ast.Constant):return node.value
    if isinstance(node,ast.Dict):return {literal(k):literal(v) for k,v in zip(node.keys,node.values)}
    if isinstance(node,ast.List):return [literal(v) for v in node.elts]
    if isinstance(node,ast.Tuple):return tuple(literal(v) for v in node.elts)
    if isinstance(node,ast.Attribute) and ast.unparse(node)=='triton.language.bfloat16':return tl.bfloat16
    if isinstance(node,ast.Call) and ast.unparse(node.func)=='DeviceProperties' and not node.args:
        return DeviceProperties(**{k.arg:literal(k.value) for k in node.keywords})
    raise ValueError('unsupported fixture AST: '+ast.dump(node))
kwargs={k.arg:k.value for k in dec.keywords};meta=literal(kwargs['triton_meta']);imeta=literal(kwargs['inductor_meta'])
fn.decorator_list=[];jit=SimpleNamespace(src=ast.unparse(fn),__name__=fn.name)
a=CachingAutotuner(jit,meta,[triton.Config({},num_warps=4,num_stages=2)],None,[],True,
    HeuristicType.USER_AUTOTUNE,size_hints=None,inductor_meta=imeta,custom_kernel=True)
assert hasattr(a,'_qwen_pin_key')
base=core.identity(jit.src,None,meta['signature'],meta['constants'],core.compiler_options(a.triton_meta))[0]
changed=dict(meta['constants'],INPUT_DTYPE=tl.float16)
assert base!=core.identity(jit.src,None,meta['signature'],changed,core.compiler_options(a.triton_meta))[0]
assert core.canonical(torch.bfloat16)!=core.canonical(tl.bfloat16)
assert core.canonical(tl.bfloat16)!=core.canonical('bfloat16')
try:core.canonical(object())
except ValueError:pass
else:raise AssertionError('unknown object accepted')
assert not torch.cuda.is_initialized()
print(json.dumps({'status':'passed','gpu_initialized':False,'fixture_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'identity':base}))
