"""Import isolated native phases without CUDA initialization or compilation."""
import hashlib,json,sys
from pathlib import Path
import torch
here=Path(__file__).resolve().parent
manifest=json.loads((here/'adapter_manifest.json').read_text())
for name,record in manifest.items():
 assert hashlib.sha256((here/'adapter'/name).read_bytes()).hexdigest()==record['adapter_sha256']
sys.path.insert(0,str(here/'adapter'))
from nvfp4_phase1 import Nvfp4MaterializedPhase1Kernel
from nvfp4_phase2 import Nvfp4MaterializedPhase2Kernel
p1=Nvfp4MaterializedPhase1Kernel(source_tile_m=128)
p2=Nvfp4MaterializedPhase2Kernel(source_tile_m=128,deterministic_output=False)
assert (p1.tile_m,p1.tile_n,p1.source_tile_m)==(64,128,128)
assert not torch.cuda.is_initialized()
print(json.dumps(dict(passed=True,phase1_shared_bytes=p1.shared_bytes,phase2_shared_bytes=p2.shared_bytes,cuda_initialized=False,scope='Import and host constructor only; no native compile or execution')))
