"""Run inside the current vLLM image with CUDA hidden; no weight loading."""
import hashlib,inspect,json
from pathlib import Path
import torch
assert not torch.cuda.is_available()
from vllm.model_executor.layers.fused_moe.oracle import nvfp4
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static,kNvfp4Dynamic
import vllm
root=Path(vllm.__file__).parent
files=['model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py','model_executor/layers/fused_moe/oracle/nvfp4.py','model_executor/layers/fused_moe/experts/marlin_moe.py','model_executor/layers/quantization/utils/marlin_utils_fp4.py','config/speculative.py']
result={'scope':'CPU source/quant-scheme selection only; no full deployment eligibility, repack or numerical validation','vllm_version':vllm.__version__,'cuda_available':False,'cuda_initialized':torch.cuda.is_initialized(),'sources':{}}
for name in files:
 p=root/name;raw=p.read_bytes();result['sources'][name]={'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()}
backend=nvfp4.map_nvfp4_backend('marlin');assert backend==nvfp4.NvFp4MoeBackend.MARLIN
assert nvfp4.backend_to_kernel_cls(backend)==[MarlinExperts]
assert MarlinExperts._supports_quant_scheme(kNvfp4Static,None)
result['selection']={'requested_target_backend':'marlin','mapped_backend':backend.value,'class':'MarlinExperts','nvfp4_weight_a16_scheme_supported':True,'draft_backend':'flashinfer_b12x','draft_tokens':2}
assert not torch.cuda.is_initialized()
print(json.dumps(result,indent=2))
