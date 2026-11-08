"""Bind isolated measured operator cells to exact generated JIT template/cubin."""
import hashlib
from pathlib import Path
import re

def sha(data):return hashlib.sha256(data).hexdigest()
def bind(root,*,m,n,k,swapped,epilogue,tile):
 expected_n=m if swapped else n;matches=[]
 for path in (root/'jit_cache/cache').glob('sm120_fp8_fp4_gemm_1d1d.*/kernel.cu'):
  source=path.read_text()
  if 'sm120_fp8_fp4_gemm_1d1d_impl<' not in source:continue
  values=[v.strip() for v in source.split('sm120_fp8_fp4_gemm_1d1d_impl<',1)[1].split('>);',1)[0].split(',')]
  if len(values)!=28:raise ValueError('unknown native template signature')
  if values[0]!='0' or values[1:3]!=[str(expected_n),str(k)] or values[19]!=epilogue:continue
  if list(map(int,values[6:9]))!=list(tile):continue
  matches.append((path,values))
 if len(matches)!=1:raise ValueError('missing/ambiguous exact generated native template: '+str((m,n,k,swapped,epilogue,tile,len(matches))))
 path,values=matches[0];base=path.parent
 return {'kernel_source':str(path.relative_to(root)),'kernel_source_sha256':sha(path.read_bytes()),
  'cubin_sha256':sha((base/'kernel.cubin').read_bytes()),'jit_meta_sha256':sha((base/'meta.json').read_bytes()),
  'effective_m':n if swapped else m,'effective_n':expected_n,'k':k,'block_m':int(values[6]),'block_n':int(values[7]),'block_k':int(values[8]),
  'pipeline_stages':int(values[12]),'grid_ctas':int(values[15]),'epilogue':values[19],'split_k':int(values[26]),
  'proof_scope':'Unique exact template/cubin in isolated run cache matched to frozen direct-call dimensions, orientation trace, tile and epilogue. Not an independent per-request profiler trace.'}
