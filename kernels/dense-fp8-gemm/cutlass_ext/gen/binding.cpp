#include <torch/extension.h>
void run_v_128x128x128_coop(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, int, int, int, int);
void run_v_64x64x128_pp(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, int, int, int, int);
void run_v_32x64x128_pp(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, int, int, int, int);
void run_v_16x64x128_pp(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, int, int, int, int);
void run_128x128x64_coop(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, int, int, int, int);
void run_128x256x64_coop(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, int, int, int, int);
void run_256x128x64_coop(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, int, int, int, int);
void run_128x128x128_pp(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, int, int, int, int);
void run_128x128x64_pp(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, int, int, int, int);
void run_64x128x128_pp(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, int, int, int, int);
void run_128x64x128_pp(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, int, int, int, int);
void run_64x256x64_pp(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, int, int, int, int);
void run_sk_128x64x128_coop(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, int, int, int, int);
void run_sk_128x128x128_coop(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, int, int, int, int);
void run_16x128x128_pp(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, int, int, int, int);
void run_16x32x128_pp(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, int, int, int, int);
void run(std::string name, torch::Tensor o, torch::Tensor a, torch::Tensor b, torch::Tensor sa, torch::Tensor sb,
         int64_t raster, int64_t swizzle, int64_t splits, int64_t decomp) {
  if (name == "v_128x128x128_coop") return run_v_128x128x128_coop(o, a, b, sa, sb, raster, swizzle, splits, decomp);
  if (name == "v_64x64x128_pp") return run_v_64x64x128_pp(o, a, b, sa, sb, raster, swizzle, splits, decomp);
  if (name == "v_32x64x128_pp") return run_v_32x64x128_pp(o, a, b, sa, sb, raster, swizzle, splits, decomp);
  if (name == "v_16x64x128_pp") return run_v_16x64x128_pp(o, a, b, sa, sb, raster, swizzle, splits, decomp);
  if (name == "128x128x64_coop") return run_128x128x64_coop(o, a, b, sa, sb, raster, swizzle, splits, decomp);
  if (name == "128x256x64_coop") return run_128x256x64_coop(o, a, b, sa, sb, raster, swizzle, splits, decomp);
  if (name == "256x128x64_coop") return run_256x128x64_coop(o, a, b, sa, sb, raster, swizzle, splits, decomp);
  if (name == "128x128x128_pp") return run_128x128x128_pp(o, a, b, sa, sb, raster, swizzle, splits, decomp);
  if (name == "128x128x64_pp") return run_128x128x64_pp(o, a, b, sa, sb, raster, swizzle, splits, decomp);
  if (name == "64x128x128_pp") return run_64x128x128_pp(o, a, b, sa, sb, raster, swizzle, splits, decomp);
  if (name == "128x64x128_pp") return run_128x64x128_pp(o, a, b, sa, sb, raster, swizzle, splits, decomp);
  if (name == "64x256x64_pp") return run_64x256x64_pp(o, a, b, sa, sb, raster, swizzle, splits, decomp);
  if (name == "sk_128x64x128_coop") return run_sk_128x64x128_coop(o, a, b, sa, sb, raster, swizzle, splits, decomp);
  if (name == "sk_128x128x128_coop") return run_sk_128x128x128_coop(o, a, b, sa, sb, raster, swizzle, splits, decomp);
  if (name == "16x128x128_pp") return run_16x128x128_pp(o, a, b, sa, sb, raster, swizzle, splits, decomp);
  if (name == "16x32x128_pp") return run_16x32x128_pp(o, a, b, sa, sb, raster, swizzle, splits, decomp);
  TORCH_CHECK(false, "unknown config ", name);
}
std::vector<std::string> configs() { return {"v_128x128x128_coop", "v_64x64x128_pp", "v_32x64x128_pp", "v_16x64x128_pp", "128x128x64_coop", "128x256x64_coop", "256x128x64_coop", "128x128x128_pp", "128x128x64_pp", "64x128x128_pp", "128x64x128_pp", "64x256x64_pp", "sk_128x64x128_coop", "sk_128x128x128_coop", "16x128x128_pp", "16x32x128_pp"}; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run);
  m.def("configs", &configs);
}
