import json,sys
from pathlib import Path
from protocol import classify
output=Path(sys.argv[1]);code=int(sys.argv[2]);log=output/'run.log';text=log.read_text() if log.exists() else ''
r=dict(process_exit=code,status=classify(text,code),performance_conclusion=None)
if (output/'report_validation.json').exists() and code==0:r['status']='counters_collected_needs_analysis'
(output/'classification.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
