#include "../gemm_template.cuh"
void run_v_32x64x128_pp(torch::Tensor& o, torch::Tensor const& a, torch::Tensor const& b, torch::Tensor const& sa,
             torch::Tensor const& sb, int raster, int swizzle, int splits, int decomp) {
  Sm120Fp8Gemm<Shape<Int<32>, Int<64>, Int<128>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, 0, void, Shape<_32,_32>>::run(
      o, a, b, sa, sb, raster, swizzle, splits, decomp);
}
