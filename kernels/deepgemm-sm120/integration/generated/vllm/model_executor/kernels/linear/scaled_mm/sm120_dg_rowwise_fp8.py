# SPDX-License-Identifier: Apache-2.0
"""Opt-in, narrow SM120 QKVZ FP8 epilogue candidate; cold setup precedes graphs."""
import torch
from vllm import envs
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)
ENABLED = envs.VLLM_SM120_DG_ROWWISE_FP8
# Only actual-operand-validated shapes. Expand only after separate evidence.
VERIFIED_M = frozenset((32, 33, 48, 63, 64, 65, 80, 96, 112, 128))
_READY = {}
_DG = None
_AUDIT = {}


def supported_weight(weight, scale):
    return (weight.is_cuda and weight.ndim == 2 and tuple(weight.shape) == (2048,12288)
            and weight.dtype == torch.float8_e4m3fn and tuple(weight.stride()) == (1,2048)
            and scale is not None and scale.is_cuda and scale.device == weight.device
            and scale.dtype == torch.float32 and scale.numel() == 12288 and scale.is_contiguous())


def prepare_for_weight(weight, scale):
    """Called only during process_weights_after_loading, before any graph capture."""
    global _DG
    if not ENABLED or not supported_weight(weight,scale): return False
    if torch.cuda.get_device_capability(weight.device) != (12,0): return False
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('DeepGEMM rowwise preparation must precede CUDA graph capture')
    key=weight.device.index
    if key in _READY: return True
    import deep_gemm
    assert hasattr(deep_gemm._C,'fp8_rowwise_gemm_nt_sm120'), 'Isolated fused-scale extension required'
    assert deep_gemm.get_pdl() is False, 'Serving experiment requires fixed PDL off before preparation'
    _DG=deep_gemm
    with torch.cuda.device(weight.device):
        b_units=deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(torch.ones(12288,16,device=weight.device,dtype=torch.float32))
        a_units={}
        # Unit metadata is shape-dependent, not tied to model weight contents.
        for m in sorted(VERIFIED_M):
            a_units[m]=deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(torch.ones(m,16,device=weight.device,dtype=torch.float32))
            a=torch.zeros(m,2048,device=weight.device,dtype=torch.float8_e4m3fn)
            out=torch.empty(m,12288,device=weight.device,dtype=torch.bfloat16)
            sa=torch.ones(m,1,device=weight.device,dtype=torch.float32)
            deep_gemm._C.fp8_rowwise_gemm_nt_sm120((a,a_units[m]),(weight.T,b_units),out,sa,scale)
        torch.cuda.synchronize(weight.device)
    _READY[key]=(a_units,b_units)
    logger.info('DG rowwise FP8 prepared device=%s QKVZ M=%s before graphs',key,sorted(VERIFIED_M))
    return True


def _eligible(a,b,sa,sb):
    return (a.is_cuda and a.ndim==2 and a.shape[0] in VERIFIED_M and a.shape[1]==2048
            and a.dtype==torch.float8_e4m3fn and a.is_contiguous() and a.data_ptr()%16==0
            and supported_weight(b,sb) and b.device==a.device and sa.device==a.device
            and sa.dtype==torch.float32 and sa.numel()==a.shape[0] and sa.is_contiguous()
            and a.device.index in _READY)


def _audit(m,path):
    # Counts Python dispatch/graph construction, not CUDA graph replay executions.
    key=(int(m),path)
    _AUDIT[key]=_AUDIT.get(key,0)+1
    if _AUDIT[key]==1:
        logger.info('DG rowwise FP8 dispatch M=%d path=%s (eager/capture dispatch)',m,path)


def dispatch_audit():
    return [{'m':m,'path':path,'python_dispatch_calls':count} for (m,path),count in sorted(_AUDIT.items())]


def _run(a:torch.Tensor,b:torch.Tensor,sa:torch.Tensor,sb:torch.Tensor,use_skinny:bool) -> torch.Tensor:
    if _eligible(a,b,sa,sb):
        a_units,b_units=_READY[a.device.index]
        out=torch.empty((a.shape[0],12288),device=a.device,dtype=torch.bfloat16)
        _DG._C.fp8_rowwise_gemm_nt_sm120((a,a_units[a.shape[0]]),(b.T,b_units),out,sa,sb)
        _audit(a.shape[0],'deepgemm_fused')
        return out
    if use_skinny:
        _audit(a.shape[0],'existing_skinny_dispatch')
        return torch.ops.vllm.sm120_fp8_scaled_mm(a,b,sa,sb)
    _audit(a.shape[0],'cutlass_fallback')
    out=torch.empty((a.shape[0],b.shape[1]),device=a.device,dtype=torch.bfloat16)
    torch.ops._C.cutlass_scaled_mm(out,a,b,sa,sb,None)
    return out


def _fake(a:torch.Tensor,b:torch.Tensor,sa:torch.Tensor,sb:torch.Tensor,use_skinny:bool) -> torch.Tensor:
    return torch.empty((a.shape[0],b.shape[1]),device=a.device,dtype=torch.bfloat16)


if ENABLED:
    direct_register_custom_op(op_name='sm120_dg_rowwise_fp8_mm',op_func=_run,fake_impl=_fake)
