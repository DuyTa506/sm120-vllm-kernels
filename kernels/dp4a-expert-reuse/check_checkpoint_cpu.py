"""Actual L19 CPU conversion parity; AST-extract installed pure helper bodies.
Swizzle's forced .cuda() is removed explicitly ONLY for this CPU check.
"""
import os,sys,ast,json,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'moe-nvfp4'))
from moe_common import load_layer_raw,get_tensor
from prepare import prepare_raw
pkg=Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils')
paths=[pkg/'nvfp4_utils.py',pkg/'flashinfer_fp4_moe.py',pkg/'quant_utils.py']
def extract(path,name,remove_cuda=False):
 node=next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)
 node.decorator_list=[]
 if remove_cuda:
  calls=[n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='cuda']
  assert len(calls)==1
  class CPU(ast.NodeTransformer):
   def visit_Call(self,n):
    n=self.generic_visit(n)
    return n.func.value if isinstance(n.func,ast.Attribute) and n.func.attr=='cuda' else n
  node=CPU().visit(node)
 scope={'torch':torch,'round_up':lambda n,k:((n+k-1)//k)*k}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),scope)
 return scope[name]
raw=load_layer_raw(19,dev='cpu');p=prepare_raw(raw)
for e in range(256):
 prefix=f'model.language_model.layers.19.mlp.experts.{e}.'
 for kind in ['weight_global_scale','input_global_scale']:
  assert torch.equal(get_tensor(prefix+'gate_proj.'+kind),get_tensor(prefix+'up_proj.'+kind)),(e,kind)
reorder=extract(paths[1],'reorder_w1w3_to_w3w1');swizzle=extract(paths[0],'swizzle_blockscale',True);amax=extract(paths[2],'amax_for_moe_activation_quant')
w,s=reorder(raw['w13'].clone(),raw['s13'].clone())
assert torch.equal(w,p['w13'])
for name,orig,actual in [('fc1',s,p['s13']),('fc2',raw['s2'],p['s2'])]:
 expected=swizzle(orig)
 assert torch.equal(expected.view(torch.uint8),actual.view(torch.uint8)),name
 # Every qf quad address is an exact byte match to original linear scales.
 e,n,k=orig.shape;r=torch.arange(n)[:,None];c=torch.arange(k)[None,:]
 offsets=((r//128)*(k//4)+c//4)*512+(r%32)*16+((r%128)//32)*4+c%4
 decoded=actual.view(torch.uint8).reshape(e,-1)[:,offsets.flatten()].reshape(e,n,k)
 assert torch.equal(decoded,orig.view(torch.uint8)),name
for stage,key,weight in [(1,'igs13','wgs13'),(2,'igs2','wgs2')]:
 effective=amax(raw[key].reciprocal(),False).reshape(1)
 assert torch.equal(p[f'a{stage}'],effective.reciprocal())
 assert torch.equal(p[f'g{stage}'],raw[weight].reciprocal()*effective)
assert not torch.cuda.is_initialized()
print(json.dumps(dict(layer=19,experts=256,cuda_initialized=False,weights_and_scale_bytes_exact=True,effective_scales_exact=True,
 swizzle_cpu_adaptation='removed exactly one forced .cuda() transfer; mathematical body unchanged',
 source_sha256={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}),indent=2))
