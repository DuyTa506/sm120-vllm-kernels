"""Prepare an isolated DG source copy and a reviewable rowwise epilogue patch; CPU only."""
from pathlib import Path
import shutil
import difflib
import hashlib
import json

D = Path(__file__).resolve().parent
SOURCE = Path('/home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/deepgemm-sm120/src')
DEST = D / 'fused_src'
if DEST.exists():
    raise SystemExit('Refusing to overwrite isolated source')
shutil.copytree(SOURCE, DEST, symlinks=True, ignore=shutil.ignore_patterns('.git', 'build', 'dist', '__pycache__', '*.egg-info', '*.so'))
changes = []
def change(name, old, new, count=1):
    p = DEST/name
    before = p.read_text()
    assert before.count(old) >= count, name
    after = before.replace(old, new, count)
    p.write_text(after)
    changes.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), 'a/'+name, 'b/'+name))

change('deep_gemm/include/deep_gemm/common/types.cuh', '    float alpha = 1.0f;', '    float alpha = 1.0f;\n    const float* row_scale = nullptr;\n    const float* col_scale = nullptr;')
change('deep_gemm/include/deep_gemm/epilogue/transform.cuh', '// Scale only the product term', '// Marker: coordinate-aware application is performed by the SM120 normal epilogue.\nstruct EpilogueRowChannel: EpilogueIdentity {};\n\n// Scale only the product term')
change('csrc/jit_kernels/impls/sm120_fp8_fp4_gemm_1d1d.hpp',
'''                                    const bool swap_ab = false,
                                    const std::optional<float>& alpha = std::nullopt) {''',
'''                                    const bool swap_ab = false,
                                    const std::optional<float>& alpha = std::nullopt,
                                    const std::optional<torch::Tensor>& row_scale = std::nullopt,
                                    const std::optional<torch::Tensor>& col_scale = std::nullopt) {''')
change('csrc/jit_kernels/impls/sm120_fp8_fp4_gemm_1d1d.hpp',
'''    config.split_k_factor = SM120ArchSpec::get_split_k_factor(desc, config.layout);
''',
'''    config.split_k_factor = SM120ArchSpec::get_split_k_factor(desc, config.layout);
    DG_HOST_ASSERT(row_scale.has_value() == col_scale.has_value());
    if (row_scale.has_value()) {
        // Experimental path deliberately rejects unverified reduction/swap contracts.
        DG_HOST_ASSERT(config.split_k_factor == 1 and not swap_ab and not c.has_value());
        DG_HOST_ASSERT(not alpha.has_value() and not epilogue_type.has_value());
    }
''')
change('csrc/jit_kernels/impls/sm120_fp8_fp4_gemm_1d1d.hpp',
'''    const auto epilogue = make_epilogue_input(m, n, epilogue_type, split_k > 1 ? std::nullopt : alpha);''',
'''    auto epilogue = make_epilogue_input(m, n, epilogue_type, split_k > 1 ? std::nullopt : alpha);
    if (row_scale.has_value()) {
        epilogue.type = "epilogue::transform::EpilogueRowChannel";
        epilogue.args.row_scale = row_scale->data_ptr<float>();
        epilogue.args.col_scale = col_scale->data_ptr<float>();
    }''')
change('deep_gemm/include/deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh',
'''            auto read_cd = [&](const cd_dtype_t& x) -> float {''',
'''            if constexpr (cute::is_same_v<epilogue_type_t, epilogue::transform::EpilogueRowChannel>) {
                DG_STATIC_ASSERT(kSplitKFactor == 1 and kGemmType == GemmType::Normal,
                                 "Row/channel epilogue supports normal non-split GEMM only");
                #pragma unroll
                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                    const uint32_t row0 = m_base + (m_tile_base + mt) * MMA_M + group_id;
                    const uint32_t row1 = row0 + 8;
                    const float sa0 = row0 < shape_m ? epilogue.row_scale[row0] : 0.0f;
                    const float sa1 = row1 < shape_m ? epilogue.row_scale[row1] : 0.0f;
                    #pragma unroll
                    for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                        const uint32_t ai = (mt * kNTilesPerWarp + nt) * MMA_ACCUM;
                        const uint32_t col = n_base + (n_tile_base + nt) * MMA_N + thread_id * 2;
                        const float sb0 = col < shape_n ? epilogue.col_scale[col] : 0.0f;
                        const float sb1 = col + 1 < shape_n ? epilogue.col_scale[col + 1] : 0.0f;
                        // Preserve FP32 (acc * token_scale) * channel_scale, round to BF16 only at store.
                        accum[ai + 0] = __fmul_rn(__fmul_rn(accum[ai + 0], sa0), sb0);
                        accum[ai + 1] = __fmul_rn(__fmul_rn(accum[ai + 1], sa0), sb1);
                        accum[ai + 2] = __fmul_rn(__fmul_rn(accum[ai + 2], sa1), sb0);
                        accum[ai + 3] = __fmul_rn(__fmul_rn(accum[ai + 3], sa1), sb1);
                    }
                }
            }

            auto read_cd = [&](const cd_dtype_t& x) -> float {''')
api = '''// Experimental narrow API: unit block scales plus original row/channel FP32 scales.
static void fp8_rowwise_gemm_nt_sm120(const std::pair<torch::Tensor, torch::Tensor>& a,
                                     const std::pair<torch::Tensor, torch::Tensor>& b,
                                     const torch::Tensor& d,
                                     const torch::Tensor& sa,
                                     const torch::Tensor& sb) {
    DG_HOST_ASSERT(a.first.is_cuda());
    const c10::cuda::CUDAGuard device_guard(a.first.device());
    DG_HOST_ASSERT(at::cuda::getDeviceProperties(a.first.get_device())->major == 12);
    DG_HOST_ASSERT(a.first.dim() == 2 and b.first.dim() == 2 and d.dim() == 2);
    DG_HOST_ASSERT(a.first.is_contiguous() and b.first.is_contiguous() and d.is_contiguous());
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn and b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    const int m = a.first.size(0), n = b.first.size(0), k = a.first.size(1);
    DG_HOST_ASSERT(m >= 32 and n > 0 and k > 0 and k % 128 == 0);
    DG_HOST_ASSERT(b.first.size(1) == k and d.size(0) == m and d.size(1) == n);
    DG_HOST_ASSERT(sa.scalar_type() == torch::kFloat and sb.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(sa.is_contiguous() and sb.is_contiguous() and sa.numel() == m and sb.numel() == n);
    for (const auto& tensor : {a.second, b.first, b.second, d, sa, sb})
        DG_HOST_ASSERT(tensor.is_cuda() and tensor.device() == a.first.device());
    auto recipe = std::make_optional(std::make_tuple(1, 1, 128));
    const auto [sfa, sfb, gran_a, gran_b] = layout::transform_sf_pair_into_required_layout(
        a.second, b.second, m, n, k, recipe, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, false);
    constexpr auto major = cute::UMMA::Major::K;
    sm120_fp8_fp4_gemm_1d1d(a.first, sfa, b.first, sfb, std::nullopt, d,
        m, n, k, gran_a, gran_b, major, major, "nk", std::nullopt, false, std::nullopt, sa, sb);
}

'''
change('csrc/apis/gemm.hpp', 'static void register_apis(pybind11::module_& m) {', api + 'static void register_apis(pybind11::module_& m) {\n    m.def("fp8_rowwise_gemm_nt_sm120", &fp8_rowwise_gemm_nt_sm120,\n          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("sa"), py::arg("sb"));')
(D/'patches'/'rowwise_epilogue_round2.patch').write_text(''.join(changes))
# Fingerprint actual original files, including the existing extension target.
manifest = {}
for p in sorted(SOURCE.rglob('*')):
    if p.is_file() and '.git' not in p.parts and 'build' not in p.parts and '__pycache__' not in p.parts:
        try: manifest[str(p.relative_to(SOURCE))] = hashlib.sha256(p.read_bytes()).hexdigest()
        except FileNotFoundError: pass
(D/'fused_source_manifest.json').write_text(json.dumps({'intended_pin':'b64107f2b9599ca76445b7f62eedf66bae1d095b','pin_tree_verified':False,'actual_original_files_sha256':manifest},indent=2)+'\n')
print('Prepared isolated source and patch:', DEST)
