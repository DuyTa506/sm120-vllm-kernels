# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_convert_sf_to_mma_layout,
    has_flashinfer_b12x_moe,
)


# bench patch (VLLM_B12X_FIX_SCALES=1): keep the checkpoint's e4m3 block scales and apply the per-expert
# weight global scale (~3e-5) in fp32 via w1_alpha/w2_alpha, instead of baking it into the e4m3 block
# scales. The bake pushes every block scale into the e4m3 subnormal range: 13-47% mean scale error and
# up to 36% of blocks flushed to zero on real weights, ~15-20% relative error per expert matmul.
_B12X_FIX_SCALES = os.environ.get("VLLM_B12X_FIX_SCALES", "0") == "1"
# Masking of negative expert ids (needed for b12x to run at all in vLLM) without the scale fix,
# to reproduce the original b12x numerics.
_B12X_MASK_NEG_IDS = _B12X_FIX_SCALES or os.environ.get("VLLM_B12X_NEGFIX_ONLY", "0") == "1"
# Experimental scratch sharing on top of the scale-corrected hybrid backend.
_B12X_SHARED_WORKSPACE = os.environ.get("VLLM_B12X_SHARED_WORKSPACE", "0") == "1"


class FlashInferB12xExperts(mk.FusedMoEExpertsModular):
    """FlashInfer CuteDSL fused MoE expert for SM12x (SM120/SM121,
    RTX Pro 6000 / DGX Spark).

    Uses ``b12x_fused_moe`` from FlashInfer PR #3080 which fuses token
    dispatch, two GEMMs, SwiGLU activation, and topk-weight reduction into a
    single kernel call.  Input quantization (BF16→FP4) is performed inside the
    kernel so BF16 hidden states are passed directly.

    Weight scale factors are converted to the MMA layout produced by
    ``convert_sf_to_mma_layout`` once during ``process_weights_after_loading``
    and cached as ``w1_sf_mma`` / ``w2_sf_mma``.

    Only NVFP4 (kNvfp4Static/kNvfp4Dynamic) quantization is supported.
    """

    _ACTIVATION_MAP: dict[MoEActivation, str] = {
        MoEActivation.SILU: "silu",
        MoEActivation.GELU_TANH: "gelu_tanh",
        MoEActivation.RELU2_NO_MUL: "relu2",
    }

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.quant_dtype == "nvfp4", (
            "FlashInferB12xExperts only supports nvfp4 quantization."
        )
        self.out_dtype = moe_config.in_dtype
        self.num_local_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        # FC2 input scale tensor bound in process_weights_after_loading: the
        # calibrated (now-zeroed) a2_gscale for static-quant checkpoints, or
        # a synthesized uniform-1.0 tensor for W4A16 checkpoints that lack
        # one. Holding it on the instance keeps apply() alloc-free.
        self._fc2_input_scale: torch.Tensor | None = None

        # Shape params for B12xMoEWrapper construction.
        self.global_num_experts = moe_config.num_experts
        self.topk = moe_config.experts_per_token
        self.hidden_dim = moe_config.hidden_dim
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.max_num_tokens = moe_config.max_num_tokens
        self.local_expert_offset = self.ep_rank * self.num_local_experts

        activation = moe_config.activation
        if activation not in self._ACTIVATION_MAP:
            raise ValueError(
                f"FlashInferB12xExperts does not support "
                f"activation {activation!r}. "
                f"Supported: {list(self._ACTIVATION_MAP.keys())}"
            )
        self._activation_str = self._ACTIVATION_MAP[activation]

        # Lazily created on first apply() call.
        self._wrapper: Any | None = None
        self.w1_sf_mma: torch.Tensor | None = None
        self.w2_sf_mma: torch.Tensor | None = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _B12X_FIX_SCALES:
            self._process_weights_fixed_scales(layer)
            return
        # Normalise block scales to absorb the per-expert weight global scale
        # (w_gs).  vLLM's NVFP4 convention stores:
        #   block_scale = max_abs * w_gs / fp4_max,  g1_alphas = 1/w_gs
        # The SM12x kernel treats w1_alpha (= g1_alphas) as a per-expert weight
        # dequant multiplier separate from input_gs (activation scale).  We bake
        # w_gs into the block scales so that w1_alpha = 1.0 and the kernel sees
        # the simpler form:
        #   block_scale = max_abs / fp4_max,  w1_alpha = 1.0
        # The FP4-packed values and dequantised results are identical in both
        # representations.  We set scale_2 = 1.0 to signal that the bake-in is
        # already done.
        layer.w13_weight_scale.data = (
            layer.w13_weight_scale.float() * layer.w13_weight_scale_2.view(-1, 1, 1)
        ).to(layer.w13_weight_scale.dtype)
        layer.w13_weight_scale_2.data.fill_(1.0)

        layer.w2_weight_scale.data = (
            layer.w2_weight_scale.float() * layer.w2_weight_scale_2.view(-1, 1, 1)
        ).to(layer.w2_weight_scale.dtype)
        layer.w2_weight_scale_2.data.fill_(1.0)

        # The SM12x kernel uses dynamic per-block quantization for FC2 input
        # activations (the SwiGLU output before the down projection).  The
        # calibrated a2_gscale from the modelopt checkpoint (~tens to hundreds)
        # is intended for static-quantisation backends (TRTLLM/CUTLASS) and
        # causes every intermediate activation to saturate at max FP4 when
        # multiplied by values that large.  Force to 1.0 so the kernel uses
        # its own per-block dynamic scale.
        if self.a2_gscale is not None:
            self.a2_gscale.fill_(1.0)
            self._fc2_input_scale = self.a2_gscale
        else:
            # W4A16 NVFP4 checkpoints have no calibrated a2_gscale; b12x
            # performs dynamic per-block FC2-input quantization, so a uniform
            # 1.0 scale per expert is equivalent to the bake-in above for
            # static-quant checkpoints. Allocate once here so apply() stays
            # alloc-free.
            self._fc2_input_scale = torch.ones(
                self.num_local_experts,
                device=layer.w13_weight.device,
                dtype=torch.float32,
            )

        # Precompute MMA-layout views of the weight scale factors once here
        # rather than recomputing on every forward pass.
        assert self.w1_scale is not None
        num_experts_w1, m1, k1_sf = self.w1_scale.shape
        k1 = k1_sf * 16
        self.w1_sf_mma = flashinfer_convert_sf_to_mma_layout(
            self.w1_scale.reshape(num_experts_w1 * m1, k1_sf),
            m=m1,
            k=k1,
            num_groups=num_experts_w1,
        )

        assert self.w2_scale is not None
        num_experts_w2, m2, k2_sf = self.w2_scale.shape
        k2 = k2_sf * 16
        self.w2_sf_mma = flashinfer_convert_sf_to_mma_layout(
            self.w2_scale.reshape(num_experts_w2 * m2, k2_sf),
            m=m2,
            k=k2,
            num_groups=num_experts_w2,
        )

    def _process_weights_fixed_scales(self, layer: torch.nn.Module) -> None:
        # Keep the checkpoint e4m3 block scales untouched; apply the per-expert weight global scale
        # (w*_weight_scale_2, ~3e-5) in fp32 through the kernel alphas.
        # b12x interprets input_global_scale / fc2_input_scale as reciprocal (dequant-style) scales,
        # i.e. activations are quantized with multiplier 1/scale, and launch folds
        # w1_alpha *= input_global_scale (FC2 is not folded). So, with the CUTLASS path's calibrated
        # static scales a13 = w13_input_scale = 1/a1_gscale and a2 = 1/a2_gscale:
        #   input_global_scale = a13, w1_alpha = w13_weight_scale_2
        #   fc2_input_scale    = a2,  w2_alpha = w2_weight_scale_2 * a2
        # VLLM_B12X_ACT_SCALES=dynamic uses unit activation scales instead (pure per-16 dynamic
        # block scaling; equally accurate vs an fp32 reference, but further from the CUTLASS path).
        dev = layer.w13_weight.device
        w13_s2 = layer.w13_weight_scale_2.to(torch.float32).view(-1)
        w2_s2 = layer.w2_weight_scale_2.to(torch.float32).view(-1)
        hybrid_min = int(os.environ.get("VLLM_B12X_HYBRID_MIN_TOKENS", "1024"))
        # b12x's direct-micro kernel (<= 3 tokens) reads the activation scales in multiplier form,
        # its MMA kernels in reciprocal form: static scales are only safe when <= 3-token batches
        # never reach b12x (hybrid threshold >= 4); otherwise fall back to unit scales.
        if (
            os.environ.get("VLLM_B12X_ACT_SCALES", "static") == "dynamic"
            or self.a1_gscale is None
            or hybrid_min < 4
        ):
            a13 = torch.ones(self.num_local_experts, dtype=torch.float32, device=dev)
            a2 = torch.ones_like(a13)
        else:
            a13 = (1.0 / self.a1_gscale.to(torch.float32)).expand(self.num_local_experts)
            a2 = (1.0 / self.a2_gscale.to(torch.float32)).expand(self.num_local_experts)
        self._input_global_scale = a13.contiguous()
        self._fc2_input_scale = a2.contiguous()
        self._w1_alpha = w13_s2.contiguous()
        self._w2_alpha = (w2_s2 * a2).contiguous()
        # Hybrid dispatch: batches with fewer than this many tokens (decode, MTP verify, small
        # chunks) run the FlashInfer CUTLASS fused MoE exactly like the FLASHINFER_CUTLASS backend
        # (same weights, same swizzled block scales, same static scales), larger ones run b12x.
        # b12x's direct-micro kernel (<= 3 tokens) reads input/fc2 scales in multiplier form while
        # its MMA kernels read them in reciprocal form, so static scales are only valid when the
        # tiny-batch band never reaches b12x: the default threshold keeps b12x on its dynamic
        # (>= 1024 routed rows) kernel.
        self._hybrid_min_tokens = hybrid_min
        if self.a1_gscale is not None and self.a2_gscale is not None:
            a1g = self.a1_gscale.to(torch.float32)
            a2g = self.a2_gscale.to(torch.float32)
            self._cutlass_g1 = (w13_s2 / a1g).contiguous()   # w13_weight_scale_2 * a13
            self._cutlass_g2 = (w2_s2 / a2g).contiguous()    # w2_weight_scale_2 * a2
        else:
            self._hybrid_min_tokens = 0

        assert self.w1_scale is not None and self.w2_scale is not None
        num_experts_w1, m1, k1_sf = self.w1_scale.shape
        self.w1_sf_mma = flashinfer_convert_sf_to_mma_layout(
            self.w1_scale.reshape(num_experts_w1 * m1, k1_sf),
            m=m1,
            k=k1_sf * 16,
            num_groups=num_experts_w1,
        )
        num_experts_w2, m2, k2_sf = self.w2_scale.shape
        self.w2_sf_mma = flashinfer_convert_sf_to_mma_layout(
            self.w2_scale.reshape(num_experts_w2 * m2, k2_sf),
            m=m2,
            k=k2_sf * 16,
            num_groups=num_experts_w2,
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(120)
            and has_flashinfer_b12x_moe()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # b12x performs in-kernel BF16->FP4 activation quant, so W4A16
        # NVFP4 checkpoints (activation_key=None, e.g. mixed-precision
        # compressed-tensors layouts) are runtime-compatible.
        return (weight_key, activation_key) in (
            (kNvfp4Static, kNvfp4Dynamic),
            (kNvfp4Static, None),
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SILU,
            MoEActivation.GELU_TANH,
            MoEActivation.RELU2_NO_MUL,
        )

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # B12xMoEWrapper does not yet support expert parallelism: its local
        # expert count must equal the global expert count.
        return not moe_parallel_config.use_ep

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # b12x_fused_moe applies topk weights internally.
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # b12x_fused_moe manages its own internal workspace.
        workspace1 = (1,)
        workspace2 = (0,)
        output_shape = (M, K)
        return (workspace1, workspace2, output_shape)

    @property
    def expects_unquantized_inputs(self) -> bool:
        # B12xMoEWrapper expects BF16 hidden states and performs its own FP4
        # quantization internally.  Returning True prevents the modular kernel
        # from pre-quantizing activations.
        return True

    def _ensure_wrapper(self) -> Any:
        """Resolve a layer wrapper in the current workspace execution owner.

        Scratch/output can be shared by sequential layers, but the wrapper's
        cached weight views and folded scales must remain layer-specific.
        Resolve the owner on every call: caching one wrapper on self would
        accidentally alias different ubatches/lanes or a reset manager.
        """
        from flashinfer.fused_moe import B12xMoEWrapper

        config = dict(
            num_experts=self.global_num_experts,
            top_k=self.topk,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_size_per_partition,
            use_cuda_graph=True,
            max_num_tokens=self.max_num_tokens,
            num_local_experts=self.num_local_experts,
            activation=self._activation_str,
            output_dtype=self.out_dtype,
            device=self.w1_scale.device,
        )
        if not _B12X_SHARED_WORKSPACE:
            if self._wrapper is None:
                self._wrapper = B12xMoEWrapper(**config)
            return self._wrapper

        from vllm.v1.worker.workspace import current_workspace_manager

        manager = current_workspace_manager()
        if not hasattr(manager, "get_persistent_resource"):
            raise RuntimeError(
                "VLLM_B12X_SHARED_WORKSPACE requires a WorkspaceManager with "
                "get_persistent_resource support."
            )

        def create_layer_wrapper():
            # Let FlashInfer size both workspace families; do not duplicate its
            # architecture-specific allocation logic. This seed never runs on
            # weights. The pinned FlashInfer exposes these buffers through its
            # shared_* constructor arguments and retains their storage.
            buffers = manager.get_persistent_resource(
                (B12xMoEWrapper, "buffers", tuple(config.items())),
                lambda: B12xMoEWrapper(**config),
            )
            return B12xMoEWrapper(
                **config,
                shared_static_workspace=buffers._static_workspace,
                shared_dynamic_workspace=buffers._dynamic_workspace,
                shared_output=buffers._moe_output,
            )

        # WorkspaceManager separates ubatches and execution lanes and forbids
        # new resources after warmup is locked. Within one owner calls must be
        # serialized, as for its other scratch. Concurrent forward streams
        # must use separate owners; CUDA graph capture itself can use a
        # different stream while retaining the same sequential owner.
        return manager.get_persistent_resource(
            (FlashInferB12xExperts, "wrapper", self), create_layer_wrapper
        )

    def _apply_cutlass(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation):
        """Same call as FlashInferExperts.apply for NVFP4 (FLASHINFER_CUTLASS backend)."""
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
            activation_to_flashinfer_type,
        )
        from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe

        a1q, a1q_scale = ops.scaled_fp4_quant(
            hidden_states, self.a1_gscale, is_sf_swizzled_layout=True
        )
        par = self.moe_config.moe_parallel_config
        flashinfer_cutlass_fused_moe(
            input=a1q,
            token_selected_experts=topk_ids.to(torch.int),
            token_final_scales=topk_weights,
            fc1_expert_weights=w1.view(torch.long),
            fc2_expert_weights=w2.view(torch.long),
            output=output,
            output_dtype=self.out_dtype,
            quant_scales=[
                self.a1_gscale,
                self.w1_scale.view(torch.int32),
                self._cutlass_g1,
                self.a2_gscale,
                self.w2_scale.view(torch.int32),
                self._cutlass_g2,
            ],
            input_sf=a1q_scale,
            tp_size=par.tp_size,
            tp_rank=par.tp_rank,
            ep_size=par.ep_size,
            ep_rank=par.ep_rank,
            activation_type=activation_to_flashinfer_type(activation),
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        assert self.w1_scale is not None and self.w2_scale is not None, (
            "w1_scale and w2_scale must not be None for FlashInferB12xExperts"
        )
        assert self.g1_alphas is not None and self.g2_alphas is not None, (
            "g1_alphas and g2_alphas must not be None for FlashInferB12xExperts"
        )
        assert self._fc2_input_scale is not None, (
            "_fc2_input_scale must be set by process_weights_after_loading"
        )
        assert self.w1_sf_mma is not None and self.w2_sf_mma is not None, (
            "process_weights_after_loading must run before FlashInferB12xExperts.apply"
        )

        if _B12X_FIX_SCALES and getattr(self, "_hybrid_min_tokens", 0) > 0:
            from flashinfer.autotuner import AutoTuner

            small = hidden_states.shape[0] < self._hybrid_min_tokens
            if small or AutoTuner.get().is_tuning_mode:
                # In the autotune warmup pass (max tokens) also run CUTLASS so its tactics
                # get tuned for all token buckets; b12x then overwrites the output.
                self._apply_cutlass(output, hidden_states, w1, w2, topk_weights, topk_ids,
                                    activation)
                if small:
                    return
        wrapper = self._ensure_wrapper()
        if _B12X_MASK_NEG_IDS:
            # vLLM marks skipped tokens with expert id -1 (dummy/profile runs); the b12x dynamic
            # kernel reads out of bounds on negative ids, so route them to expert 0 with weight 0.
            skip = topk_ids < 0
            topk_ids = topk_ids.masked_fill(skip, 0)
            topk_weights = topk_weights.masked_fill(skip, 0.0)
        if _B12X_FIX_SCALES and getattr(self, "_w1_alpha", None) is not None:
            wrapper_output = wrapper.run(
                x=hidden_states,
                w1_weight=w1,
                w1_weight_sf=self.w1_sf_mma,
                w1_alpha=self._w1_alpha,
                fc2_input_scale=self._fc2_input_scale,
                w2_weight=w2,
                w2_weight_sf=self.w2_sf_mma,
                w2_alpha=self._w2_alpha,
                token_selected_experts=topk_ids.to(torch.int32),
                token_final_scales=topk_weights,
                input_global_scale=self._input_global_scale,
            )
            output.copy_(wrapper_output)
            return

        wrapper_output = wrapper.run(
            x=hidden_states,
            w1_weight=w1,
            w1_weight_sf=self.w1_sf_mma,
            w1_alpha=self.g1_alphas,
            fc2_input_scale=self._fc2_input_scale,
            w2_weight=w2,
            w2_weight_sf=self.w2_sf_mma,
            w2_alpha=self.g2_alphas,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights,
        )
        output.copy_(wrapper_output)
