#include "../gemm_template.cuh"
void run_16x32x128_pp(torch::Tensor& o, torch::Tensor const& a, torch::Tensor const& b, torch::Tensor const& sa,
             torch::Tensor const& sb, int raster, int swizzle, int splits, int decomp) {
  Sm120Fp8Gemm<Shape<Int<16>, Int<32>, Int<128>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, 0, void, Shape<_16,_32>>::run(
      o, a, b, sa, sb, raster, swizzle, splits, decomp);
}
