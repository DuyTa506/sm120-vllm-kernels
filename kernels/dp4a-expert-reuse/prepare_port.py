"""Text-only reviewed port; does not execute upstream source."""
from pathlib import Path
import hashlib
p=Path(__file__).resolve().parent
s=(p/'sources/qf_moe.cu').read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='158649bd31de55e60bcb170ccd1ac491e691e579ef8e4e35857099c1a4a75d08'
def replace(old,new,n=1):
 global s
 assert s.count(old)==n,(old,s.count(old),n)
 s=s.replace(old,new)
replace('#include <ATen/cuda/CUDAContext.h>', '#include <c10/cuda/CUDAStream.h>')
replace('at::cuda::getCurrentCUDAStream()', 'c10::cuda::getCurrentCUDAStream()')
replace('H = 2560, I = 640, TOPK = 10','H = 2048, I = 512, TOPK = 8')
replace('wv[10][2]','wv[H / 256][2]');replace('sv[10]','sv[H / 256]')
replace('i < 10','i < H / 256',2)
replace('wv[5][2]','wv[I / 128][2]');replace('sv[5]','sv[I / 128]')
replace('i < 5','i < I / 128',2)
replace('uint32_t * __restrict__ hq, float * __restrict__ hsf) {','uint32_t * __restrict__ hq, float * __restrict__ hsf, bool reuse) {')
replace('__restrict__ out, int n_experts) {','__restrict__ out, int n_experts, bool reuse) {')
replace('if (__syncthreads_or(tid < pos && s_ids[tid] == e)) return;','if (__syncthreads_or(reuse && tid < pos && s_ids[tid] == e)) return;',2)
replace('if (tid < n && s_ids[tid] == e) atomicOr','if (tid < n && s_ids[tid] == e && (reuse || tid == pos)) atomicOr')
replace('for (int k = 0; k < tid; ++k)\n                if (s_ids[k] == ej)','for (int k = 0; reuse && k < tid; ++k)\n                if (s_ids[k] == ej)')
replace('if (ej == e) {','if (ej == e && (reuse || tid == pos)) {')
replace('const at::Tensor & g2_alpha, const at::Tensor & a2_gs, bool pdl) {','const at::Tensor & g2_alpha, const at::Tensor & a2_gs,\n                            at::Tensor & hq, at::Tensor & hsf, at::Tensor & part_out,\n                            at::Tensor & done, at::Tensor & out, bool reuse) {')
replace('    const int T = x.size(0);', (p/'guards.txt').read_text()+'    const int T = x.size(0);')
start=s.index('    auto opts = x.options();');end=s.index('    const int g1_smem',start)
s=s[:start]+'''    // Per-invocation explicit workspace; same stream owns it until completion.
    TORCH_CHECK(hq.scalar_type()==at::kInt && hq.numel()>=MAXSLOT*MAXT*(I/8));
    TORCH_CHECK(hsf.scalar_type()==at::kFloat && hsf.numel()>=MAXSLOT*MAXT*IB);
    TORCH_CHECK(part_out.scalar_type()==at::kFloat && part_out.numel()>=MAXSLOT*MAXT*H);
    TORCH_CHECK(done.scalar_type()==at::kInt && done.numel()>=G2_TILES);
    TORCH_CHECK(out.scalar_type()==at::kBFloat16 && out.sizes()==x.sizes());
    for (const auto* t : {&topk_ids,&topk_w,&w13,&w13_sf,&g1_alpha,&a1_gs,&w2,&w2_sf,&g2_alpha,&a2_gs,
                           static_cast<const at::Tensor*>(&hq),static_cast<const at::Tensor*>(&hsf),
                           static_cast<const at::Tensor*>(&part_out),static_cast<const at::Tensor*>(&done),static_cast<const at::Tensor*>(&out)}) {
        TORCH_CHECK(t->is_contiguous() && t->device()==x.device());
    }
    // Explicit memset nodes are graph-capturable and included in full-call timing.
    // All-invalid batches have no final owner, so pre-zero output is mandatory.
    C10_CUDA_CHECK(cudaMemsetAsync(out.data_ptr(),0,out.nbytes(),s));
    C10_CUDA_CHECK(cudaMemsetAsync(done.data_ptr(),0,done.nbytes(),s));
''' +s[end:]
start=s.index('    static bool attr = false;');end=s.index('    cudaLaunchAttribute lattr',start)
s=s[:start]+s[end:]
replace('pdl ? 1 : 0','0')
replace('hsf.data_ptr<float>()));','hsf.data_ptr<float>(), reuse));')
replace('out_p, (int) w2.size(0)));','out_p, (int) w2.size(0), reuse));')
# Update inherited shape-specific comments without changing instructions.
for old,new in [('Qwen3.8-Flash-Next on SM120: 1..8 tokens, top-10 of 512','Qwen3.6 on SM120: 1..8 tokens, top-8 of 256'),('160, 40 scale blocks per row','128, 32 scale blocks per row'),('40, 10 quads','32, 8 quads'),('positions x 40 blocks','positions x 32 blocks'),('5 quads (64 elements)','I/128 quads (64 elements)'),('x [T, 2560]','x [T, 2048]'),('[T, 10]','[T, 8]'),('[E, 1280, 1280]','[E, 1024, 1024]'),('[E, 1280, 160]','[E, 1024, 128]'),('[E, 2560, 320]','[E, 2048, 256]'),('[E, 2560, 40]','[E, 2048, 32]')]:
 s=s.replace(old,new)
s='''// Derived from SSHdotCodes/qwen-3.8-flash-next-pro6000, Apache-2.0.
// Upstream commit 8e5be4440eeea37d31994b0845d1344201de7741; see LICENSE.upstream.
// Local experimental H2048/I512/top8 port, explicit workspace, reuse switch, PDL disabled.
'''+s+'\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("apply", &nvfp4_moe_decode); }\n'
(p/'qf_moe_port.cu').write_text(s)
