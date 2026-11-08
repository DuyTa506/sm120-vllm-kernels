import hashlib
import json
from pathlib import Path
import os
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','12.0a')
from torch.utils.cpp_extension import load
p=Path(__file__).resolve().parent
flags=['-O3','-std=c++17','-gencode=arch=compute_120a,code=sm_120a']
(p/'build').mkdir(exist_ok=True)
m=load(name='sm120_skinny_fp8',sources=[str(p/'sm120_skinny_fp8.cu')],build_directory=str(p/'build'),extra_cuda_cflags=flags,verbose=True)
manifest={'module_name':'sm120_skinny_fp8','compile_flags':flags,'TORCH_CUDA_ARCH_LIST':os.environ['TORCH_CUDA_ARCH_LIST'],
          'scope':'existing repository candidate comparator; not proof of active CONFIGnext dispatch',
          'files':{f:hashlib.sha256((p/f).read_bytes()).hexdigest() for f in ('sm120_skinny_fp8.cu','reference_dispatch.py','build/sm120_skinny_fp8.so')}}
(p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
