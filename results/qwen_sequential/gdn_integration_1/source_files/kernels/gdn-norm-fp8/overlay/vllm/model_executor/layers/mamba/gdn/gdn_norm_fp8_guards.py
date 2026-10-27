"""Pure CPU dispatch rules for the default-off packed GDN experiment."""
OP_NAME = "gdn_attention_core_norm_fp8_packed_experimental"
MIN_FUSED_TOKENS = 1024
SCALE_FLOOR = 1.0 / (448.0 * 512.0)
CONFIGURATION_KEYS = (
    "flag", "sm120", "tp1", "pp1", "compiled_inductor", "packed_path",
    "norm_contract", "gdn_shape", "projection_type", "scheme_type", "kernel_type",
    "dynamic_token", "static_channel", "dtype_bf16", "weight_shape", "no_padding",
    "no_lora", "no_unquantized_branch", "input_parallel", "no_bias", "return_bias",
    "compiled_native_quant", "split_boundary", "standard_splits", "no_partition",
)


def configuration_rejection(spec):
    for key in CONFIGURATION_KEYS:
        if spec.get(key) is not True:
            return key
    return None


def packed_route(total_tokens, actual_tokens, num_prefills):
    if total_tokens < 1:
        raise ValueError("Positive padded token count required")
    if actual_tokens is None:
        return "profile"
    if not 0 <= actual_tokens <= total_tokens:
        raise ValueError("Actual token count exceeds output storage")
    if actual_tokens >= MIN_FUSED_TOKENS and num_prefills > 0:
        return "fused"
    return "original"


def reference_rows_per_block(tokens, sm_count):
    if tokens < 1 or sm_count < 1:
        raise ValueError("Positive token and SM counts required")
    needed = (tokens * 32 + 2 * sm_count - 1) // (2 * sm_count)
    return min(1 << (needed - 1).bit_length(), 4)


def fallback_warmup_sizes(capture_sizes, max_tokens):
    if max_tokens < 2 or any(t < 1 or t > max_tokens for t in capture_sizes):
        raise ValueError("Invalid capture sizes")
    # Torch specializes size 1 separately. Size 2 starts the dynamic >=2 path.
    return sorted({1, 2, max_tokens, *capture_sizes})
