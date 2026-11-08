
#include <deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm120_fp8_fp4_gemm_1d1d_impl<
        0,
        16,
        2048,
        128,
        128,
        1,
        64,
        16,
        64,
        64,
        64,
        0,
        16,
        128,
        256,
        188,
        GemmType::Normal,
        false,
        cutlass::bfloat16_t,
        epilogue::transform::EpilogueRowChannelSwapped,
        false,
        false,
        false,
        true,
        false,
        64,
        1,
        128
    >);
};
