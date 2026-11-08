import hashlib,json,shutil,sys
from pathlib import Path
root,target=map(Path,sys.argv[1:3]);names=['protocol.py','test_protocol.py','baseline.py','process_guard.py','run.sh','provenance.py','capture_manifest.json','validate_report.py','classify_result.py']
files=[root/name for name in names]+sorted(p for p in (root/'tools').rglob('*') if p.is_file())
data={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
if '--verify' in sys.argv:
 assert json.loads(target.read_text())==data,'sources or portable tool payload changed'
else:
 target.write_text(json.dumps(data,indent=2)+'\n');snapshot=target.parent/'source_snapshot';snapshot.mkdir()
 for name in names:shutil.copy2(root/name,snapshot/name)
