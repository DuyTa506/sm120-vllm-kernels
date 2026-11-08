#!/usr/bin/env python3
"""CPU-only pre-launch guard for the reviewed focused PDL experiment."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify(manifest_path,mode,reference):
 manifest=json.loads(manifest_path.read_text())
 if manifest['status']!='reviewed_focused_pdl' or mode not in ('pin','record'):raise ValueError('unsupported focused protocol')
 for relative,digest in manifest['files'].items():
  path=(ROOT/relative).resolve()
  if not path.is_relative_to(ROOT) or sha(path)!=digest:raise ValueError('focused reviewed source changed: '+relative)
 if mode=='pin':
  if not reference or sha(Path(reference))!=manifest['reference_sha256']:raise ValueError('focused reference differs from frozen review')
  ref=json.loads(Path(reference).read_text())
  if ref['status']!='canonical_complete' or len(ref['reference'])!=manifest['reference_identity_count']:raise ValueError('invalid reviewed pin reference')
 return {'status':'passed_focused_review','review_sha256':sha(manifest_path),'files':len(manifest['files']),'mode':mode,
  'reference_sha256':manifest['reference_sha256'] if mode=='pin' else None,'coverage':manifest['coverage']}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('mode',choices=('pin','record'));p.add_argument('reference',nargs='?');a=p.parse_args();print(json.dumps(verify(a.manifest,a.mode,a.reference)))
