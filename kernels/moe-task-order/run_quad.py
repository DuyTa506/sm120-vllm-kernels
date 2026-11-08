"""Serialized ABBA fresh processes, unique caches, identical real inputs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(sys.argv[1]);root.mkdir(exist_ok=False)
tokens=os.environ.get("TASK_ORDER_TOKENS","1024")
assert tokens in ("1024","2048","4096")
if os.environ.get("TASK_ORDER_MODE")=="geometry":
    command=[sys.executable,str(Path(__file__).with_name('micro.py')),'--variant','baseline',
             '--layers','19','--tokens','2048','3072','4096','--patterns','real','--geometry-only','--output',str(root/'geometry')]
    started=time.time();result=subprocess.run(command)
    (root/'terminals.json').write_text(json.dumps([dict(mode='geometry_only',command=command,exit_code=result.returncode,started=started,ended=time.time())],indent=2))
    raise SystemExit(result.returncode)
records=[]
for label,variant in [('a1','baseline'),('b1','long_first'),('b2','long_first'),('a2','baseline')]:
    command=[sys.executable,str(Path(__file__).with_name('micro.py')),'--variant',variant,
             '--layers','19','--tokens',tokens,'--patterns','real','--output',str(root/label)]
    started=time.time();result=subprocess.run(command)
    records.append(dict(label=label,variant=variant,command=command,exit_code=result.returncode,started=started,ended=time.time()))
    (root/'terminals.json').write_text(json.dumps(records,indent=2))
    if result.returncode:raise SystemExit(result.returncode)
command=[sys.executable,str(Path(__file__).with_name('compare.py')),'--baseline',str(root/'a1'),str(root/'a2'),
         '--candidate',str(root/'b1'),str(root/'b2')]
with (root/'comparison.json').open('w') as f: subprocess.run(command,stdout=f,check=True)
