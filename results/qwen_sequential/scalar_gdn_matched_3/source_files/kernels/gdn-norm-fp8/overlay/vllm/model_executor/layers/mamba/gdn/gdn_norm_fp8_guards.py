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
    "fixed_decode_capture",
    "baseline_passes",
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


def fallback_warmup_sizes(capture_sizes, max_tokens, *, decode_query_len=1,
                          max_num_seqs=None, max_capture_size=None):
    if max_tokens < 2 or any(t < 1 or t > max_tokens for t in capture_sizes):
        raise ValueError("Invalid capture sizes")
    if decode_query_len < 1 or (max_num_seqs is not None and max_num_seqs < 1):
        raise ValueError("Invalid decode capture configuration")
    # Torch specializes size 1 separately. Size 2 starts the dynamic >=2 path.
    warmups = {1, 2, max_tokens, *capture_sizes}
    if decode_query_len > 1:
        if max_num_seqs is None:
            raise ValueError("Speculative decode requires max_num_seqs")
        capture_limit = max(capture_sizes, default=0) if max_capture_size is None else max_capture_size
        if not 0 <= capture_limit <= max_tokens:
            raise ValueError("Invalid maximum capture size")
        limit = min(max_num_seqs * decode_query_len, capture_limit, max_tokens)
        # MRV2 CudaGraphManager creates FULL descriptors from rounded token
        # counts without mutating compilation_config.cudagraph_capture_sizes.
        # Keep the original counts too: PIECEWISE and varlen decode use them.
        for tokens in capture_sizes:
            rounded = ((tokens + decode_query_len - 1) // decode_query_len) * decode_query_len
            if rounded <= limit:
                warmups.add(rounded)
    return sorted(warmups)
