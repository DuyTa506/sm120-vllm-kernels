import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(sys.argv[1])
root.mkdir(exist_ok=False)
records=[]
for workspace in ('shared','private'):
    command=[sys.executable,str(Path(__file__).with_name('repeat_oracle.py')),'--workspace',workspace,
             '--layer','39','--tokens','1024','--repeats','8','--output',str(root/workspace)]
    started=time.time()
    result=subprocess.run(command)
    records.append(dict(workspace=workspace,command=command,exit_code=result.returncode,started=started,ended=time.time()))
    (root/'terminals.json').write_text(json.dumps(records,indent=2))
    if result.returncode: raise SystemExit(result.returncode)
