"""Post-run read-only native artifact inspection. Does not execute GPU code."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
output=Path(sys.argv[1]);measurement=output/'measurement'
tool=shutil.which('cuobjdump')
if tool is None:
 for name in ['/usr/local/cuda/bin/cuobjdump','/usr/local/cuda-13.0/bin/cuobjdump']:
  if Path(name).exists():tool=name;break
report={'tool':tool,'artifacts':[],'scope':'Resource/SASS inspection; absence of explicit spill data is not zero-spill proof.'}
for p in sorted(measurement.rglob('*')):
 if not p.is_file() or p.suffix not in ['.o','.cubin']:continue
 row={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 if tool:
  for flag,label in [('--dump-resource-usage','resource'),('--dump-sass','sass')]:
   result=subprocess.run([tool,flag,str(p)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
   dest=output/f'artifact_{len(report["artifacts"])}_{label}.txt';dest.write_text(result.stdout)
   row[label]={'returncode':result.returncode,'path':str(dest)}
 report['artifacts'].append(row)
(output/'artifact_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'artifacts':len(report['artifacts']),'cuobjdump':tool}))
