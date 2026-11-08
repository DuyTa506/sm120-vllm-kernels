"""Compile/import local port without device access; never invokes its CUDA API."""
import os,json,hashlib,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TORCH_CUDA_ARCH_LIST']='12.0a'
os.environ['MAX_JOBS']='2'
import torch
from torch.utils.cpp_extension import load
p=Path(__file__).resolve().parent;diag='--diagnostic' in sys.argv;name='dp4a_expert_reuse_diag' if diag else 'dp4a_expert_reuse_sm120';source='qf_moe_diag.cu' if diag else 'qf_moe_port.cu';b=p/('build_diag' if diag else 'build');b.mkdir(exist_ok=True)
m=load(name=name,sources=[str(p/source)],build_directory=str(b),extra_cuda_cflags=['-O3','-std=c++17'],extra_cflags=['-O3'],extra_include_paths=[str(x) for x in Path('/usr/local/lib/python3.12/dist-packages/nvidia').glob('*/include') if x.parent.name in ('cusparse','cublas','cusolver','curand')],verbose=True)
assert not torch.cuda.is_initialized()
print(json.dumps(dict(cuda_initialized=False,source_sha256=hashlib.sha256((p/source).read_bytes()).hexdigest(),artifact=m.__file__)))
