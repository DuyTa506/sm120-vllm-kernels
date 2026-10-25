"""JIT-build the SM120 FP8 CUTLASS config sweep extension (one .cu per config, parallel ninja build).
import cutlass_ext.build as b; ext = b.load(); ext.run(name, out, a, bT, sa, sb, raster, swizzle, splits, decomp)"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass"
COOP = "cutlass::gemm::KernelTmaWarpSpecializedCooperative"
PP = "cutlass::gemm::KernelTmaWarpSpecializedPingpong"
SK = "cutlass::gemm::StreamKScheduler"

# name: (M, N, K tile, schedule, stages (0=auto), scheduler, epilogue tile or None)
CONFIGS = {
    # replicas of vLLM's built-in sm120 configs (should be bitwise identical + same speed)
    "v_128x128x128_coop": (128, 128, 128, COOP, 0, "void", None),
    "v_64x64x128_pp": (64, 64, 128, PP, 0, "void", None),
    "v_32x64x128_pp": (32, 64, 128, PP, 0, "void", "Shape<_32,_32>"),
    "v_16x64x128_pp": (16, 64, 128, PP, 0, "void", "Shape<_16,_32>"),
    # prefill candidates
    "128x128x64_coop": (128, 128, 64, COOP, 0, "void", None),
    "128x256x64_coop": (128, 256, 64, COOP, 0, "void", None),
    "256x128x64_coop": (256, 128, 64, COOP, 0, "void", None),
    "128x128x128_pp": (128, 128, 128, PP, 0, "void", None),
    "128x128x64_pp": (128, 128, 64, PP, 0, "void", None),
    "64x128x128_pp": (64, 128, 128, PP, 0, "void", None),
    "128x64x128_pp": (128, 64, 128, PP, 0, "void", None),
    "64x256x64_pp": (64, 256, 64, PP, 0, "void", None),
    # decode candidates: split-K / stream-K over the small-M pingpong tiles
    "sk_128x64x128_coop": (128, 64, 128, COOP, 0, SK, None),
    "sk_128x128x128_coop": (128, 128, 128, COOP, 0, SK, None),
    "16x128x128_pp": (16, 128, 128, PP, 0, "void", "Shape<_16,_32>"),
    "16x32x128_pp": (16, 32, 128, PP, 0, "void", "Shape<_16,_32>"),
}


def _gen(only=None):
    src_dir = os.path.join(HERE, "gen")
    os.makedirs(src_dir, exist_ok=True)
    names = [n for n in CONFIGS if only is None or n in only]
    files = []
    for n in names:
        tm, tn, tk, sched, st, scheduler, epi = CONFIGS[n]
        epi_t = epi if epi else "cutlass::epilogue::collective::EpilogueTileAuto"
        body = f"""#include "../gemm_template.cuh"
void run_{n}(torch::Tensor& o, torch::Tensor const& a, torch::Tensor const& b, torch::Tensor const& sa,
             torch::Tensor const& sb, int raster, int swizzle, int splits, int decomp) {{
  Sm120Fp8Gemm<Shape<Int<{tm}>, Int<{tn}>, Int<{tk}>>, {sched}, {st}, {scheduler}, {epi_t}>::run(
      o, a, b, sa, sb, raster, swizzle, splits, decomp);
}}
"""
        p = os.path.join(src_dir, f"{n}.cu")
        if not os.path.exists(p) or open(p).read() != body:
            open(p, "w").write(body)
        files.append(p)
    decl = "\n".join(
        f"void run_{n}(torch::Tensor&, torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, "
        f"torch::Tensor const&, int, int, int, int);" for n in names)
    table = "\n".join(f'  if (name == "{n}") return run_{n}(o, a, b, sa, sb, raster, swizzle, splits, decomp);'
                      for n in names)
    lst = ", ".join(f'"{n}"' for n in names)
    binding = f"""#include <torch/extension.h>
{decl}
void run(std::string name, torch::Tensor o, torch::Tensor a, torch::Tensor b, torch::Tensor sa, torch::Tensor sb,
         int64_t raster, int64_t swizzle, int64_t splits, int64_t decomp) {{
{table}
  TORCH_CHECK(false, "unknown config ", name);
}}
std::vector<std::string> configs() {{ return {{{lst}}}; }}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
  m.def("run", &run);
  m.def("configs", &configs);
}}
"""
    p = os.path.join(src_dir, "binding.cpp")
    if not os.path.exists(p) or open(p).read() != binding:
        open(p, "w").write(binding)
    return [p] + files


def load(only=None, name="sm120_fp8_sweep", verbose=False):
    from torch.utils.cpp_extension import load as _load
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
    os.environ.setdefault("MAX_JOBS", str(min(16, os.cpu_count() or 8)))
    build_dir = os.path.join(HERE, ".build", name)
    os.makedirs(build_dir, exist_ok=True)
    return _load(
        name=name,
        sources=_gen(only),
        build_directory=build_dir,
        extra_include_paths=[f"{CUTLASS}/include", f"{CUTLASS}/tools/util/include"],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-std=c++20", "-DNDEBUG", "--expt-relaxed-constexpr", "--expt-extended-lambda",
                           "-gencode=arch=compute_120a,code=sm_120a", "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
                           "-Xcompiler=-fno-strict-aliasing"],
        verbose=verbose,
    )


if __name__ == "__main__":
    import sys
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    m = load(only, verbose=True)
    print(m.configs())
