#!/usr/bin/env python3
"""Use measured native policy template arguments to instantiate rowwise epilogues CPU-only."""
import hashlib
import json
from pathlib import Path
import re
import shlex
HERE=Path(__file__).resolve().parent
cache=HERE/'results/swap_native_matrix_1/jit_cache/cache';out=HERE/'rowwise_swap_control/nvcc';out.mkdir(exist_ok=True)
selected={}
for path in cache.glob('sm120*/kernel.cu'):
 source=path.read_text();m=re.search(r'impl<\s*(\d+), (\d+), (\d+)',source)
 if not m or int(m[3])!=2048:continue
 n=int(m[2])
 if n not in (1,3,8,16,12288):continue
 label='off' if n==12288 else f'on_m{n}'
 if label in selected:raise ValueError('ambiguous measured template')
 marker='epilogue::transform::EpilogueIdentity'
 if source.count(marker)!=1:raise ValueError('unexpected source marker')
 new=source.replace(marker,'epilogue::transform::EpilogueRowChannel' + ('' if label=='off' else 'Swapped'))
 target=out/(label+'.cu');target.write_text(new)
 cmd=shlex.split(json.loads((path.parent/'meta.json').read_text())['command']);cmd[1]='/work/nvcc/'+target.name;cmd[cmd.index('--output-file')+1]='/work/nvcc/'+label+'.cubin'
 selected[label]={'command':cmd,'original_source':str(path.relative_to(HERE)),'original_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'source_sha256':hashlib.sha256(new.encode()).hexdigest()}
if set(selected)!={'off','on_m1','on_m3','on_m8','on_m16'}:raise ValueError('missing measured tiny templates')
(out/'commands.json').write_text(json.dumps(selected,indent=2)+'\n');print('prepared',sorted(selected))
