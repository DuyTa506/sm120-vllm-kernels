import hashlib
import json
import shutil
from pathlib import Path
import sys

root, target = map(Path, sys.argv[1:3])
files = sorted(p for p in root.rglob("*") if p.is_file() and "results" not in p.parts
               and "__pycache__" not in p.parts and p.suffix != ".pyc")
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
