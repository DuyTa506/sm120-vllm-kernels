#!/usr/bin/env python3
"""CPU disassembly inventory; does not establish which consumer vLLM executed."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

def audit(binary,arch):
 proc=subprocess.Popen(['cuobjdump','--dump-sass','--gpu-architecture',arch,str(binary)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 records=[];name=None;lines=[]
 def finish():
  if name is None:return
  text=''.join(lines)
  if 'cutlass' in name.lower() or 'gemm' in name.lower():
   records.append({'function':name,'sass_sha256':hashlib.sha256(text.encode()).hexdigest(),
       'griddepcontrol_instructions':[l.strip() for l in lines if re.search(r'GRIDDEP|GRID_DEP|GDC\.',l,re.I)],'sass_lines':len(lines)})
 for line in proc.stdout:
  m=re.search(r'Function\s*:\s*(\S+)',line)
  if m:finish();name=m[1];lines=[]
  if name is not None:lines.append(line)
 finish();error=proc.stderr.read();rc=proc.wait()
 if rc:raise RuntimeError(f'cuobjdump failed {rc}: {error}')
 return {'arch':arch,'gemm_functions':records,'stderr':error}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--binary',type=Path,default=Path('/usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so'));a=p.parse_args()
 binary=a.binary
 out={'binary':str(binary),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
  'architectures':[audit(binary,arch) for arch in ('sm_120','sm_120a')],
  'limitation':'Static binary inventory only. Runtime symbol selection, actual graph node attributes and dependent overlap remain unverified.'}
 a.output.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps([{'arch':x['arch'],'gemm_functions':len(x['gemm_functions']),'with_gdc':sum(bool(k['griddepcontrol_instructions']) for k in x['gemm_functions'])} for x in out['architectures']]))
