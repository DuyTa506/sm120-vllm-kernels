import hashlib
import json
import shutil
from pathlib import Path
import sys

root, target = map(Path, sys.argv[1:3])
names=['workspace_adapter.py','test_workspace_adapter.py','micrograph.py','process_guard.py','run.sh','provenance.py','capture_manifest.json']
files=[root/name for name in names]
data = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
if "--verify" in sys.argv:
    if json.loads(target.read_text()) != data:
        raise SystemExit("experiment sources changed during run")
else:
    target.write_text(json.dumps(data, indent=2))
    archive = target.parent / (target.stem + "_snapshot")
    archive.mkdir(exist_ok=False)
    for source in files:
        destination = archive / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
