"""Import-only report validation; no device query or CUDA allocation."""
import csv,io,json,subprocess,sys
from pathlib import Path
from protocol import METRICS
here=Path(__file__).resolve().parent;output=Path(sys.argv[1]);reports=list(output.glob('*.ncu-rep'));assert len(reports)==1,reports
cmd=[str(here/'tools/linux-desktop-glibc_2_11_3-x64/ncu'),'--import',str(reports[0]),'--page','raw','--csv']
r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);(output/'counter_table.csv').write_text(r.stdout);(output/'counter_import.stderr').write_text(r.stderr);assert r.returncode==0
lines=r.stdout.splitlines();start=next(i for i,l in enumerate(lines) if 'Metric Name' in l);rows=list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))))
assert all(x.get('Metric Value','').strip().lower() not in ('','n/a','nan','not supported') for x in rows if x.get('Metric Name') in METRICS)
metrics={x['Metric Name'] for x in rows};assert set(METRICS)<=metrics,(set(METRICS)-metrics)
assert all('gatedMoEGatedDynamicKernel' in x.get('Kernel Name','') for x in rows if x.get('Metric Name') in METRICS)
assert len({x['ID'] for x in rows if x.get('Metric Name') in METRICS})==1
passes=[json.loads(p.read_text()) for p in sorted((output/'passes').glob('*/report.json'))];assert passes and all(p['status']=='passed' and p['tokens']==1024 and p['tile_m']==128 for p in passes)
for p in passes:assert p['capture_sha256']==passes[0]['capture_sha256'] and p['checkpoint_identity']==passes[0]['checkpoint_identity']
result=dict(status='counters_collected',native_kernel_records=1,application_passes=len(passes),requested_metrics=list(METRICS),performance_conclusion=None)
(output/'report_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
