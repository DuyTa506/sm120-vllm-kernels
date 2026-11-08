#!/usr/bin/env python3
"""Patch only an owned exact-fingerprint fused DG snapshot for tiny rowwise swap."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
BASE_FP='bbffbea38eb18e6abb84f7b860ae297e1b57b06d99729fbf6c8be0cbe5d112e2'
def sha(data):return hashlib.sha256(data).hexdigest()
def once(text,old,new):
 if text.count(old)!=1:raise ValueError('unsupported source fragment: '+old[:100])
 return text.replace(old,new)
def main():
 p=argparse.ArgumentParser();p.add_argument('owned',type=Path);args=p.parse_args();owned=args.owned.resolve();root=owned/'src'
 data=(owned/'base_fingerprint.json').read_bytes()
 if sha(data)!=BASE_FP:raise ValueError('not peer-confirmed frozen source fingerprint')
 expected=json.loads(data)['actual_files_sha256'];changes={}
 def read(rel):
  data=(root/rel).read_bytes()
  if sha(data)!=expected[rel]:raise ValueError('base file differs: '+rel)
  return data.decode()
 rel='csrc/apis/gemm.hpp';old=read(rel)
 begin=old.index('static void fp8_rowwise_gemm_nt_sm120(');end=old.index('\nstatic void register_apis(',begin)
 original=old[begin:end]
 tiny=once(original,'fp8_rowwise_gemm_nt_sm120(','fp8_rowwise_gemm_nt_sm120_tiny(')
 tiny=once(tiny,'const torch::Tensor& sb) {','const torch::Tensor& sb,\n                                     const bool swap_ab) {')
 tiny=once(tiny,'m >= 32 and n > 0','m >= 1 and m <= 16 and n > 0')
 call='''    sm120_fp8_fp4_gemm_1d1d(a.first, sfa, b.first, sfb, std::nullopt, d,
        m, n, k, gran_a, gran_b, major, major, "nk", std::nullopt, false, std::nullopt, sa, sb);'''
 replace='''    if (std::getenv("DG_ROWWISE_SWAP_TRACE")) {
        std::printf("ROWWISE_SWAP_V1 m=%d n=%d k=%d selected=%d\\n", m, n, k, static_cast<int>(swap_ab));
        std::fflush(stdout);
    }
    if (swap_ab) {
        // Effective row is original output channel; effective column is token.
        // Swapped epilogue retains original (acc * token_scale) * channel_scale.
        sm120_fp8_fp4_gemm_1d1d(b.first, sfb, a.first, sfa, std::nullopt, d,
            n, m, k, gran_b, gran_a, major, major, "nk", std::nullopt, true, std::nullopt, sb, sa);
    } else {
        sm120_fp8_fp4_gemm_1d1d(a.first, sfa, b.first, sfb, std::nullopt, d,
            m, n, k, gran_a, gran_b, major, major, "nk", std::nullopt, false, std::nullopt, sa, sb);
    }'''
 tiny=once(tiny,call,replace)
 new=old[:end]+'\n'+tiny+old[end:]
 new=once(new,'static void register_apis(pybind11::module_& m) {','''static void register_apis(pybind11::module_& m) {
    m.def("fp8_rowwise_gemm_nt_sm120_tiny", &fp8_rowwise_gemm_nt_sm120_tiny,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("sa"), py::arg("sb"), py::arg("swap_ab"));''')
 new='#include <cstdio>\n#include <cstdlib>\n'+new;changes[rel]=(old,new)
 rel='csrc/jit_kernels/impls/sm120_fp8_fp4_gemm_1d1d.hpp';old=read(rel)
 new=once(old,'DG_HOST_ASSERT(config.split_k_factor == 1 and not swap_ab and not c.has_value());','''DG_HOST_ASSERT(config.split_k_factor == 1 and not c.has_value());
        // Only the explicitly tiny API may use the transposed row/channel path.
        if (swap_ab) DG_HOST_ASSERT(n >= 1 and n <= 16);''')
 new=once(new,'epilogue.type = "epilogue::transform::EpilogueRowChannel";','''epilogue.type = swap_ab ? "epilogue::transform::EpilogueRowChannelSwapped"
                                : "epilogue::transform::EpilogueRowChannel";''');changes[rel]=(old,new)
 rel='deep_gemm/include/deep_gemm/epilogue/transform.cuh';old=read(rel)
 new=once(old,'struct EpilogueRowChannel: EpilogueIdentity {};','''struct EpilogueRowChannel: EpilogueIdentity {};
// Effective row is channel, column is token; multiply token scale first.
struct EpilogueRowChannelSwapped: EpilogueIdentity {};''');changes[rel]=(old,new)
 rel='deep_gemm/include/deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh';old=read(rel)
 new=once(old,'if constexpr (cute::is_same_v<epilogue_type_t, epilogue::transform::EpilogueRowChannel>) {','''if constexpr (cute::is_same_v<epilogue_type_t, epilogue::transform::EpilogueRowChannel> or
                          cute::is_same_v<epilogue_type_t, epilogue::transform::EpilogueRowChannelSwapped>) {''')
 comment='// Preserve FP32 (acc * token_scale) * channel_scale, round to BF16 only at store.'
 new=once(new,comment,'''// Preserve FP32 (acc * token_scale) * channel_scale in BOTH orientations.
                        const auto scale = [](float value, float row, float col) {
                            if constexpr (cute::is_same_v<epilogue_type_t, epilogue::transform::EpilogueRowChannelSwapped>)
                                return __fmul_rn(__fmul_rn(value, col), row);
                            else
                                return __fmul_rn(__fmul_rn(value, row), col);
                        };''')
 for i,row,col in ((0,'sa0','sb0'),(1,'sa0','sb1'),(2,'sa1','sb0'),(3,'sa1','sb1')):
  new=once(new,f'__fmul_rn(__fmul_rn(accum[ai + {i}], {row}), {col})',f'scale(accum[ai + {i}], {row}, {col})')
 changes[rel]=(old,new)
 patch=''.join(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/'+p,tofile='b/'+p)) for p,(a,b) in changes.items())
 for rel,(_,new) in changes.items():(root/rel).write_text(new)
 (owned/'rowwise_swap.patch').write_text(patch)
 manifest={'status':'CPU_prepared_not_built','base_fingerprint_sha256':BASE_FP,'files':{p:{'base_sha256':sha(a.encode()),'patched_sha256':sha(b.encode())} for p,(a,b) in changes.items()},'patch_sha256':sha(patch.encode()),
  'api':'fp8_rowwise_gemm_nt_sm120_tiny','scope':'M1..16, same FP8 bytes, FP32 row/channel scales, BF16 output; no C/alpha/split-K. Distinct swapped marker preserves token then channel multiplication.'}
 (owned/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
if __name__=='__main__':main()
