"""Extract complete embedded CUDA ELF and inspect it using host tools only."""
from pathlib import Path
import struct,json,hashlib,subprocess,re,sys
from collections import Counter
source=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(exist_ok=False)
b=source.read_bytes();rows=[]
for i in range(1,len(b)-64):
 if b[i:i+4]!=b'\x7fELF' or struct.unpack_from('<H',b,i+18)[0]!=190:continue
 phoff=struct.unpack_from('<Q',b,i+32)[0];phent,phnum=struct.unpack_from('<HH',b,i+54);shoff=struct.unpack_from('<Q',b,i+40)[0];shent,shnum=struct.unpack_from('<HH',b,i+58);size=max(phoff+phent*phnum,shoff+shent*shnum)
 for n in range(phnum):
  off=i+phoff+n*phent;start=struct.unpack_from('<Q',b,off+8)[0];length=struct.unpack_from('<Q',b,off+32)[0];size=max(size,start+length)
 for n in range(shnum):
  off=i+shoff+n*shent;kind=struct.unpack_from('<I',b,off+4)[0];start,length=struct.unpack_from('<QQ',b,off+24)
  if kind!=8:size=max(size,start+length)
 assert i+size<=len(b);o=output/'kernel.cubin';o.write_bytes(b[i:i+size]);rows.append(dict(offset=i,bytes=size,object_sha256=hashlib.sha256(b).hexdigest(),cubin_sha256=hashlib.sha256(o.read_bytes()).hexdigest()))
assert len(rows)==1
for command,name in [(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(o)],'resource.txt'),(['/usr/local/cuda/bin/nvdisasm',str(o)],'sass.txt')]:
 result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(output/name).write_text(result.stdout);assert result.returncode==0,result.stdout
counts=Counter()
for line in (output/'sass.txt').read_text().splitlines():
 m=re.search(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)',line)
 if m:counts[m.group(1).split('.')[0]]+=1
r=rows[0];r.update(instructions=sum(counts.values()),opcounts=dict(counts),resource=(output/'resource.txt').read_text());(output/'summary.json').write_text(json.dumps(r,indent=2)+'\n');print(r['resource']);print({k:counts[k] for k in ['LDL','STL','LDSM','OMMA','BRA']})
