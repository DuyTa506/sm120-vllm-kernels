"""Host-only derived artifact extraction/inspection after the child GPU job exits."""
import json,subprocess,sys
from pathlib import Path
here=Path(__file__).resolve().parent;output=Path(sys.argv[1]);rows=[]
for i,path in enumerate(sorted((output/'measurement').rglob('*.o'))):
 dest=output/f'compiled_artifact_{i}'
 result=subprocess.run([sys.executable,str(here/'inspect_artifact.py'),str(path),str(dest)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 (output/f'compiled_artifact_{i}.log').write_text(result.stdout)
 rows.append({'object':str(path),'inspection':str(dest),'returncode':result.returncode})
(output/'artifact_report.json').write_text(json.dumps(rows,indent=2)+'\n')
assert rows and all(r['returncode']==0 for r in rows),'compiled artifact inspection failed'
