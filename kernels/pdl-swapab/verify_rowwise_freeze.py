#!/usr/bin/env python3
"""Verify reviewed tiny-rowwise experiment inputs without imports or GPU access."""
import hashlib
import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parent
manifest=json.loads(Path(sys.argv[1]).read_text())
if manifest['status']!='CPU_ready_GPU_not_run':raise ValueError('unexpected frozen protocol status')
for relative,want in manifest['files'].items():
 path=(root/relative).resolve()
 if not path.is_relative_to(root.parents[1]):raise ValueError('frozen source path escapes worktree')
 if hashlib.sha256(path.read_bytes()).hexdigest()!=want:raise ValueError('reviewed source changed: '+relative)
print(json.dumps({'status':'verified_reviewed_rowwise_sources','files':len(manifest['files'])}))
