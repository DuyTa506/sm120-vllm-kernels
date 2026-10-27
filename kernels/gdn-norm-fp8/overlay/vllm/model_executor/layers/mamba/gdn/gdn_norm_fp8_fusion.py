"""Default-off packed GDN experiment; no model-level symbolic shape branch.

Original packed core + compiled native QuantFP8-to-output handle small batches.
Larger prefills use the validated explicit-layout kernel. All paths retain
RowParallelLinear and its CUTLASS GEMM through QuantizedActivation.
"""
import os
import weakref

import torch

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import CutlassFP8ScaledMMLinearKernel
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation, expose_input_quant_key, get_input_quant_key,
)
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsLinearMethod
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import CompressedTensorsW8A8Fp8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape, kFp8DynamicTokenSym, kFp8StaticChannelSym,
)
from vllm.platforms import current_platform

from .gdn_norm_fp8_guards import (
    OP_NAME, MIN_FUSED_TOKENS, SCALE_FLOOR, configuration_rejection, fallback_warmup_sizes,
    packed_route, reference_rows_per_block,
)
from .gdn_norm_fp8_kernel import fused_norm_gate_fp8_explicit
from .gdn_norm_fp8_quant_out import make_raw_quant_out

logger = init_logger(__name__)


def _is_compiled_native_quant(quant):
    if type(quant) is not QuantFP8:
        return False
    original = getattr(quant._forward_method, "_torchdynamo_orig_callable", None)
    return (getattr(original, "__func__", None) is QuantFP8.forward_native
            and quant.static is False and quant.group_shape == GroupShape.PER_TOKEN
            and quant.num_token_padding is None and not quant.is_group_quant
            and not quant.column_major_scales and not quant.tma_aligned_scales)


class _GdnDynamicTokenCutlass(CutlassFP8ScaledMMLinearKernel):
    """Advertise QA after original weight processing and startup checks."""
    def __init__(self, original, owner, warmup_sizes):
        if type(original) is not CutlassFP8ScaledMMLinearKernel:
            raise TypeError("Only the original CUTLASS FP8 kernel can be adapted")
        expected = {"logical_output_size", "quant_fp8", "fp8_dtype", "config", "layer_param_names"}
        if set(vars(original)) != expected:
            raise ValueError("CUTLASS state changed; refuse an unreviewed adapter")
        for name in expected:
            setattr(self, name, getattr(original, name))
        self._owner = weakref.ref(owner)
        self._warmup_sizes = tuple(warmup_sizes)
        self._quant_out = None
        self._ready = False

    def input_quant_key(self):
        return kFp8DynamicTokenSym if self._ready else None

    def process_weights_after_loading(self, layer):
        # Existing loader lifecycle, before model tracing/capture.
        super().process_weights_after_loading(layer)
        owner = self._owner()
        if owner is None:
            raise RuntimeError("GDN owner disappeared during weight processing")
        w, ws, xs, xsub = self._get_layer_params(layer)
        nw = owner.norm.weight
        eligible = (
            w.shape == (4096, 2048) and w.dtype == torch.float8_e4m3fn
            and w.stride() == (1, 4096) and w.is_cuda
            and ws.numel() == 2048 and ws.dtype == torch.float32 and ws.is_contiguous()
            and ws.device == w.device and self.logical_output_size == 2048
            and xs is None and xsub is None
            and nw.shape == (128,) and nw.dtype == torch.bfloat16
            and nw.is_contiguous() and nw.device == w.device
            and _is_compiled_native_quant(self.quant_fp8)
            and not getattr(layer, "requires_unquantized_input", False)
        )
        if not eligible:
            logger.info("GDN norm-FP8 disabled after loading %s: tensor/quantizer contract", owner.prefix)
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("GDN fallback compilation must precede graph capture")
        # Match the validated standalone compilation options exactly. The raw
        # method is the retained quantizer's original native implementation;
        # self.quant_fp8 stays intact as oracle and ordinary Tensor fallback.
        self._quant_out = torch.compile(
            make_raw_quant_out(self.quant_fp8), dynamic=True, fullgraph=True)
        # Includes size 1, dynamic >=2, max T and every configured graph bucket.
        with torch.inference_mode():
            for tokens in self._warmup_sizes:
                x = torch.zeros((tokens, 4096), device=w.device, dtype=torch.bfloat16)
                q = torch.empty((tokens, 4096), device=w.device, dtype=torch.float8_e4m3fn)
                scale = torch.empty((tokens, 1), device=w.device, dtype=torch.float32)
                self._quant_out(x, q, scale)
            # The fused route has one layout/stride specialization at T>=1024.
            # Compile and launch it before capture as well; grid size is dynamic.
            if reference_rows_per_block(MIN_FUSED_TOKENS, owner._gdn_norm_fp8_sm_count) != 4:
                raise RuntimeError("Unexpected SM count for prefill FLA specialization")
            x = torch.zeros((1, 32, 128), device=w.device, dtype=torch.bfloat16)
            packed_z = torch.zeros((1, 12288), device=w.device, dtype=torch.bfloat16)
            z = packed_z.as_strided((1, 32, 128), (12288, 128, 1), 8192)
            fused_norm_gate_fp8_explicit(x, z, nw, num_warps=8, reference_rows_per_block=4)
        self._ready = True
        expose_input_quant_key(layer, self)
        if get_input_quant_key(layer) != kFp8DynamicTokenSym:
            raise RuntimeError("GDN external quantization capability was hidden")
        owner._gdn_norm_fp8_enabled = True
        logger.info("GDN norm-FP8 ready for %s: explicit prefill + original packed/compiled-quant-out fallback", owner.prefix)


def configure_gdn_norm_fp8(layer, vllm_config):
    """Install pending adapter at construction; finalize only after loading."""
    if os.environ.get("VLLM_GDN_NORM_FP8_FUSION", "0") != "1" or not current_platform.is_cuda():
        return
    projection, norm = layer.out_proj, layer.norm
    scheme = getattr(projection, "scheme", None)
    kernel = getattr(scheme, "fp8_linear", None)
    config = getattr(kernel, "config", None)
    compilation = vllm_config.compilation_config
    splits = set(compilation.splitting_ops or ())
    spec = dict(
        flag=True, sm120=torch.cuda.get_device_capability() == (12, 0),
        tp1=layer.tp_size == 1 and vllm_config.parallel_config.tensor_parallel_size == 1,
        pp1=vllm_config.parallel_config.pipeline_parallel_size == 1,
        compiled_inductor=(not vllm_config.model_config.enforce_eager and int(compilation.mode) == 3
                           and compilation.backend == "inductor"
                           and not compilation.is_custom_op_enabled("quant_fp8")),
        packed_path=layer.enable_fused_gdn_spec_decode and not layer.gqa_interleaved_layout,
        norm_contract=(norm.eps == 1e-6 and norm.group_size is None and norm.norm_before_gate
                       and norm.activation == "silu" and norm.bias is None),
        gdn_shape=(layer.num_v_heads == 32 and layer.head_v_dim == 128
                   and layer.value_dim == 4096 and layer.hidden_size == 2048
                   and layer.key_dim == 2048),
        projection_type=(type(projection) is RowParallelLinear
                         and type(projection.quant_method) is CompressedTensorsLinearMethod),
        scheme_type=type(scheme) is CompressedTensorsW8A8Fp8,
        kernel_type=type(kernel) is CutlassFP8ScaledMMLinearKernel,
        dynamic_token=(getattr(config, "activation_quant_key", None) == kFp8DynamicTokenSym
                       and kernel.input_quant_key() is None) if kernel is not None else False,
        static_channel=getattr(config, "weight_quant_key", None) == kFp8StaticChannelSym,
        dtype_bf16=(getattr(config, "input_dtype", None) == torch.bfloat16
                    and getattr(config, "out_dtype", None) == torch.bfloat16),
        weight_shape=getattr(config, "weight_shape", None) == (2048, 4096),
        no_padding=kernel.get_output_padding() is None if kernel is not None else False,
        no_lora=vllm_config.lora_config is None,
        no_unquantized_branch=not getattr(projection, "requires_unquantized_input", False),
        input_parallel=projection.input_is_parallel,
        no_bias=projection.bias is None, return_bias=projection.return_bias,
        compiled_native_quant=_is_compiled_native_quant(getattr(kernel, "quant_fp8", None)),
        split_boundary=f"vllm::{OP_NAME}" in splits,
        standard_splits=set(compilation._attention_ops).issubset(splits),
        no_partition=not compilation.use_inductor_graph_partition,
    )
    rejection = configuration_rejection(spec)
    if rejection is not None:
        logger.info("GDN norm-FP8 disabled for %s: %s", layer.prefix, rejection)
        return
    warmups = fallback_warmup_sizes(compilation.cudagraph_capture_sizes or (),
                                   vllm_config.scheduler_config.max_num_batched_tokens)
    try:
        adapted = _GdnDynamicTokenCutlass(kernel, layer, warmups)
    except (TypeError, ValueError) as exc:
        logger.info("GDN norm-FP8 disabled for %s: %s", layer.prefix, str(exc))
        return
    scheme.fp8_linear = adapted
    layer._gdn_norm_fp8_sm_count = int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)


def packed_norm_fp8(layer, mixed_qkvz, ba, core, q, scale, metadata):
    """Opaque-op implementation; metadata decisions occur outside Dynamo."""
    tokens = core.shape[0]
    if (not layer._gdn_norm_fp8_enabled or core.shape != (tokens, 32, 128)
            or mixed_qkvz.shape != (tokens, 12288) or ba.shape != (tokens, 64)
            or core.dtype != torch.bfloat16 or mixed_qkvz.dtype != core.dtype
            or ba.dtype != core.dtype or not core.is_contiguous()
            or mixed_qkvz.stride() != (12288, 1)
            or q.shape != (tokens, 4096) or q.dtype != torch.float8_e4m3fn
            or scale.shape != (tokens, 1) or scale.dtype != torch.float32
            or not q.is_contiguous() or not scale.is_contiguous()
            or core.device != q.device or scale.device != q.device
            or mixed_qkvz.device != q.device or ba.device != q.device):
        raise RuntimeError("GDN packed FP8 runtime contract changed")
    actual = None if metadata is None else metadata.num_actual_tokens
    route = packed_route(tokens, actual, 0 if metadata is None else metadata.num_prefills)
    if route == "profile":
        layer._warmup_prefill_kernels(mixed_qkvz[:, :8192], 0)
        q.zero_()
        scale.fill_(SCALE_FLOOR)
        return
    if route == "original":
        # Exact fused MTP decode core, state updates and small prefill.
        layer._forward_core_fused_norm_packed(mixed_qkvz, ba, core)
        kernel = layer.out_proj.scheme.fp8_linear
        if not kernel._ready or kernel._quant_out is None:
            raise RuntimeError("Compiled QuantFP8 was not warmed before capture")
        if torch.cuda.is_current_stream_capturing() and tokens not in kernel._warmup_sizes:
            raise RuntimeError("Unwarmed GDN fallback CUDA graph shape")
        # Baseline quantizes all padded rows, after normalizing actual rows.
        kernel._quant_out(core.flatten(-2), q, scale)
        return
    mixed_qkv, z_flat = mixed_qkvz.split([8192, 4096], dim=-1)
    z = z_flat.view(tokens, 32, 128)
    b, a = layer.split_ba(ba)
    layer._forward_core(mixed_qkv=mixed_qkv, b=b.contiguous(), a=a.contiguous(), core_attn_out=core)
    # Never feed padded Z into SiLU: unused projection rows may contain garbage.
    if actual < tokens:
        q[actual:].zero_()
        scale[actual:].fill_(SCALE_FLOOR)
    fused_norm_gate_fp8_explicit(
        core[:actual], z[:actual], layer.norm.weight, out=(q[:actual], scale[:actual]),
        num_warps=8, reference_rows_per_block=reference_rows_per_block(actual, layer._gdn_norm_fp8_sm_count))


def quantized_activation(q, scale):
    return QuantizedActivation(data=q, scale=scale, orig_dtype=torch.bfloat16,
                               orig_shape=torch.Size([q.shape[0], 4096]),
                               quant_key=kFp8DynamicTokenSym)
