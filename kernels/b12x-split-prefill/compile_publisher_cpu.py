"""Compile actual publisher to SM120 PTX with explicit target; never launch."""
import hashlib,json,sys
from pathlib import Path
import torch,triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
import device_publisher as d
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
cases=[(d._count,{'Ids':'*i32','Counts':'*i32','Errors':'*i32'},{'TOPK':8,'E':256}),
       (d._prefix,{'Counts':'*i32','Bases':'*i32'},{'E':256}),
       (d._metadata,dict.fromkeys(['Counts','Bases','TaskExpert','TaskRows'],'*i32'),{'E':256,'NT':4}),
       (d._pack,{'Ids':'*i32','Weights':'*fp32','Q':'*u8','SF':'*u8','Bases':'*i32','WriteRows':'*i32','PA':'*u8','SA':'*u8','TokenMap':'*i32','RouteSlots':'*i32','TokenWeights':'*fp32'},{'TOPK':8,'E':256,'K':2048})]
rows=[]
for fn,sig,const in cases:
 sig.update({k:'constexpr' for k in const})
 c=triton.compile(ASTSource(fn,sig,constexprs=const),target=GPUTarget('cuda',120,32),options={'num_warps':4})
 ptx=c.asm['ptx'];(out/(fn.__name__+'.ptx')).write_text(ptx)
 rows.append({'kernel':fn.__name__,'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(),'num_warps':c.metadata.num_warps,'shared':c.metadata.shared})
assert not torch.cuda.is_initialized()
result={'passed':True,'scope':'Offline SM120 compile only; no launch or GPU validation','cuda_initialized':False,'kernels':rows}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
