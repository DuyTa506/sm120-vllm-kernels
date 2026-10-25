#include "../gemm_template.cuh"
void run_64x256x64_pp(torch::Tensor& o, torch::Tensor const& a, torch::Tensor const& b, torch::Tensor const& sa,
             torch::Tensor const& sb, int raster, int swizzle, int splits, int decomp) {
  Sm120Fp8Gemm<Shape<Int<64>, Int<256>, Int<64>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, 0, void, cutlass::epilogue::collective::EpilogueTileAuto>::run(
      o, a, b, sa, sb, raster, swizzle, splits, decomp);
}
