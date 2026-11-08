import hashlib,json,subprocess
from pathlib import Path
root=Path('/work/nvcc');commands=json.loads((root/'commands.json').read_text());report={}
for name,entry in commands.items():
 p=root/(name+'.cu');assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['source_sha256']
 result=subprocess.run(entry['command'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 (root/(name+'.log')).write_text(result.stdout)
 row={'exit_code':result.returncode,**entry};binary=root/(name+'.cubin')
 if result.returncode==0:row['cubin_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
 report[name]=row;(root/'results.json').write_text(json.dumps(report,indent=2)+'\n')
 print(name,result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
