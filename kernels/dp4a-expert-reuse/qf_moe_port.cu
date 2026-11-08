// Derived from SSHdotCodes/qwen-3.8-flash-next-pro6000, Apache-2.0.
// Upstream commit 8e5be4440eeea37d31994b0845d1344201de7741; see LICENSE.upstream.
// Local experimental H2048/I512/top8 port, explicit workspace, reuse switch, PDL disabled.
// Decode-size NVFP4 (W4A4) mixture-of-experts for Qwen3.6 on SM120: 1..8 tokens, top-8 of 256.
// Same quantization recipe as FlashInfer's CUTLASS MoE (TensorRT-LLM kernels): activations are quantized
// to e2m1 with e4m3 block scales (16 elements) using the fast-math reciprocal recipe; gemm1 output is
// rounded to bf16, SwiGLU silu(gate) * up is rounded to bf16 and re-quantized before gemm2.
// Products of two e2m1 codes are small integers (after x2 scaling), so each 16-element block sum is
// computed exactly with dp4a and scaled by both e4m3 block scales, as the block-scaled MMA does.
//
//   gemm1 : one CTA per (top-k position, 16-row block); positions that repeat an earlier expert exit,
//           so every selected expert is read once; x -> FP4 for the routed tokens; SwiGLU; h -> FP4
//   gemm2 : one CTA per (expert, 128 output rows); the last CTA per row tile sums in top-k order
// h is stored per (first top-k position of the expert, token), so no routing pass is needed between.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

constexpr int H = 2048, I = 512, TOPK = 8, MAXT = 8, MAXSLOT = MAXT * TOPK;
constexpr int HB = H / 16, IB = I / 16;          // 128, 32 scale blocks per row
constexpr int HQ = HB / 4, IQ = IB / 4;          // 32, 8 quads (64 elements) per row
constexpr int W13_ROWS = 2 * I;                  // [up; gate]


__device__ __forceinline__ float rcp_approx(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// e4m3 (round-to-nearest, satfinite) of a positive value; returns the code and its float value.
__device__ __forceinline__ uint8_t to_e4m3(float v, float & back) {
    const __nv_fp8_e4m3 t = __nv_fp8_e4m3(v);
    back = static_cast<float>(t);
    return t.__x;
}

__device__ __forceinline__ float e4m3_to_float(uint32_t code) {
    __nv_fp8_e4m3 t;
    t.__x = (uint8_t) code;
    return static_cast<float>(t);
}

// 16 values -> 8 bytes of e2m1 codes (low nibble = even element) + e4m3 scale, FlashInfer fast-math recipe.
__device__ __forceinline__ uint2 quant16(const float (&v)[16], float gs, float & sf_val) {
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
    float sf;
    const uint8_t code = to_e4m3(gs * (amax * rcp_approx(6.0f)), sf);
    (void) code;
    const float out_scale = amax != 0.f ? rcp_approx(sf * rcp_approx(gs)) : 0.f;
    sf_val = sf;
    uint32_t w[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        uint32_t packed = 0;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float a = v[h * 8 + 2 * p] * out_scale, b = v[h * 8 + 2 * p + 1] * out_scale;
            uint16_t r;
            asm("{\n.reg .b8 t;\ncvt.rn.satfinite.e2m1x2.f32 t, %2, %1;\ncvt.u16.u8 %0, t;\n}" : "=h"(r) : "f"(a), "f"(b));
            packed |= (uint32_t) (r & 0xFF) << (8 * p);
        }
        w[h] = packed;
    }
    return make_uint2(w[0], w[1]);
}

// 8 e2m1 codes in one word -> int8 values x2 ({0,1,2,3,4,6,8,12} with sign), split even/odd elements.
__device__ __forceinline__ int2 fp4x8_to_i8(uint32_t q) {
    constexpr uint32_t t0 = 0x03020100u, t1 = 0x0C080604u, t2 = 0xFDFEFF00u, t3 = 0xF4F8FAFCu;
    const uint32_t sel = 0x32103210u | ((q & 0x88888888u) >> 1);
    uint32_t tmp[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t s = q >> (16 * i);
        const uint32_t lo = __byte_perm(t0, t1, s);
        const uint32_t hi = __byte_perm(t2, t3, s);
        tmp[i] = __byte_perm(lo, hi, sel >> (16 * i));
    }
    return make_int2((int) __byte_perm(tmp[0], tmp[1], 0x6420), (int) __byte_perm(tmp[0], tmp[1], 0x7531));
}

// swizzled 128x4 block-scale layout (sglang swizzle_blockscale): 4 consecutive blocks of one row
__device__ __forceinline__ size_t sf_quad_offset(int row, int quad, int kq) {
    return ((size_t) (row >> 7) * kq + quad) * 512 + (row & 31) * 16 + ((row & 127) >> 5) * 4;
}


// ------------------------------------------------------------------------------------------------
constexpr int G1_THREADS = 128;

__device__ __forceinline__ void quant_x_block(const __nv_bfloat16 * __restrict__ src, float gs, int2 * dst8, float & sf) {
    const uint4 u0 = __ldg(reinterpret_cast<const uint4 *>(src)), u1 = __ldg(reinterpret_cast<const uint4 *>(src) + 1);
    float v[16];
    const __nv_bfloat162 * h0 = reinterpret_cast<const __nv_bfloat162 *>(&u0);
    const __nv_bfloat162 * h1 = reinterpret_cast<const __nv_bfloat162 *>(&u1);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 a = __bfloat1622float2(h0[i]), c = __bfloat1622float2(h1[i]);
        v[2 * i] = a.x; v[2 * i + 1] = a.y; v[8 + 2 * i] = c.x; v[8 + 2 * i + 1] = c.y;
    }
    const uint2 q = quant16(v, gs, sf);
    dst8[0] = fp4x8_to_i8(q.x);
    dst8[1] = fp4x8_to_i8(q.y);
}

// grid: T*TOPK positions x 32 blocks of 16 intermediate rows. Local rows 0-15 = up, 16-31 = gate.
__global__ void __launch_bounds__(G1_THREADS) moe_gemm1_kernel(
        const __nv_bfloat16 * __restrict__ x, int T, const int * __restrict__ ids,
        const uint8_t * __restrict__ w13, const uint8_t * __restrict__ w13_sf, const float * __restrict__ g1_alpha,
        const float * __restrict__ a1_gs_p, const float * __restrict__ a2_gs_p, int n_experts,
        uint32_t * __restrict__ hq, float * __restrict__ hsf, bool reuse) {
    extern __shared__ int2 s_x[];                // [T_e][H / 8] int8 x2 split, then [T_e][HB] floats
    __shared__ int s_ids[MAXSLOT];
    __shared__ unsigned s_mask;
    __shared__ float s_dot[32][MAXT];
    __shared__ float s_act[MAXT][16];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n = T * TOPK, pos = blockIdx.x / (I / 16), jt = blockIdx.x % (I / 16);
    asm volatile("griddepcontrol.wait;" ::: "memory");   // router outputs (programmatic dependent launch)
    if (tid < n) s_ids[tid] = ids[tid];
    if (tid == 0) s_mask = 0;
    __syncthreads();
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int e = s_ids[pos];
    if (e < 0 || e >= n_experts) return;                          // padded token
    if (__syncthreads_or(reuse && tid < pos && s_ids[tid] == e)) return;   // an earlier position owns this expert

    // weight loads first: they depend only on the expert
    const int r = warp * 8 + (lane >> 2), q0 = lane & 3;
    const int grow = (r < 16 ? 0 : I) + jt * 16 + (r & 15);
    const uint8_t * wrow = w13 + ((size_t) e * W13_ROWS + grow) * (H / 2);
    const uint8_t * sfe = w13_sf + (size_t) e * W13_ROWS * HB;
    uint4 wv[H / 256][2];
    uint32_t sv[H / 256];
#pragma unroll
    for (int i = 0; i < H / 256; ++i) {
        const int q = q0 + 4 * i;
        const uint4 * p = reinterpret_cast<const uint4 *>(wrow + q * 32);
        wv[i][0] = __ldcs(p);
        wv[i][1] = __ldcs(p + 1);
        sv[i] = __ldg(reinterpret_cast<const unsigned int *>(sfe + sf_quad_offset(grow, q, HQ)));
    }
    if (tid < n && s_ids[tid] == e && (reuse || tid == pos)) atomicOr(&s_mask, 1u << (tid / TOPK));
    __syncthreads();
    const unsigned mask = s_mask;
    const int te = __popc(mask);
    float * s_xsf = reinterpret_cast<float *>(s_x + te * (H / 8));
    const float a1_gs = *a1_gs_p;
    for (int c = tid; c < te * HB; c += G1_THREADS) {
        const int rk = c / HB, b = c % HB;
        unsigned m = mask;
        for (int k = 0; k < rk; ++k) m &= m - 1;
        const int t = __ffs(m) - 1;
        quant_x_block(x + (size_t) t * H + b * 16, a1_gs, s_x + rk * (H / 8) + b * 2, s_xsf[rk * HB + b]);
    }
    __syncthreads();

    float acc[MAXT];
#pragma unroll
    for (int t = 0; t < MAXT; ++t) acc[t] = 0.f;
#pragma unroll
    for (int i = 0; i < H / 256; ++i) {
        const int q = q0 + 4 * i;
        int2 wi[8];
        const uint32_t * ww = reinterpret_cast<const uint32_t *>(&wv[i][0]);
#pragma unroll
        for (int k = 0; k < 8; ++k) wi[k] = fp4x8_to_i8(ww[k]);
        float wsf[4];
#pragma unroll
        for (int b = 0; b < 4; ++b) wsf[b] = e4m3_to_float((sv[i] >> (8 * b)) & 0xFF);
#pragma unroll
        for (int rk = 0; rk < MAXT; ++rk) {
            if (rk >= te) break;
            float part = 0.f;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                int s = 0;
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const int2 xv = s_x[rk * (H / 8) + q * 8 + b * 2 + k];
                    s = __dp4a(wi[b * 2 + k].x, xv.x, s);
                    s = __dp4a(wi[b * 2 + k].y, xv.y, s);
                }
                part = fmaf((float) s, wsf[b] * s_xsf[rk * HB + q * 4 + b], part);
            }
            acc[rk] += part;
        }
    }
    const float alpha = g1_alpha[e] * 0.25f;
#pragma unroll
    for (int rk = 0; rk < MAXT; ++rk) {
        if (rk >= te) break;
        float v = acc[rk];
        v += __shfl_xor_sync(0xffffffff, v, 1);
        v += __shfl_xor_sync(0xffffffff, v, 2);
        if (q0 == 0) s_dot[r][rk] = v * alpha;
    }
    __syncthreads();
    // SwiGLU on bf16-rounded gemm1 outputs; result rounded to bf16 before quantization
    for (int c = tid; c < 16 * te; c += G1_THREADS) {
        const int j = c & 15, rk = c >> 4;
        const float up = __bfloat162float(__float2bfloat16(s_dot[j][rk]));
        const float gate = __bfloat162float(__float2bfloat16(s_dot[16 + j][rk]));
        s_act[rk][j] = __bfloat162float(__float2bfloat16(gate * (1.f / (1.f + __expf(-gate))) * up));
    }
    __syncthreads();
    if (tid < te) {
        const int rk = tid;
        unsigned m = mask;
        for (int k = 0; k < rk; ++k) m &= m - 1;
        const int t = __ffs(m) - 1;
        float v[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) v[j] = s_act[rk][j];
        float sf;
        const uint2 q = quant16(v, *a2_gs_p, sf);
        const size_t pair = (size_t) pos * MAXT + t;
        hq[pair * (I / 8) + 2 * jt] = q.x;
        hq[pair * (I / 8) + 2 * jt + 1] = q.y;
        hsf[pair * IB + jt] = sf;
    }
}

// ------------------------------------------------------------------------------------------------
// gemm2: one CTA per (top-k position, 128 output rows); positions that repeat an earlier expert exit.
// 8 warps x 16 rows, 2 lanes per row, I/128 quads (64 elements) each. Each CTA writes its weighted partial
// rows; the last CTA of a row tile adds them in top-k order (deterministic) and writes bf16.
constexpr int G2_THREADS = 256, G2_ROWS = 128, G2_TILES = H / G2_ROWS;

__global__ void __launch_bounds__(G2_THREADS) moe_gemm2_kernel(
        int T, const int * __restrict__ ids, const float * __restrict__ tw,
        const uint32_t * __restrict__ hq, const float * __restrict__ hsf,
        const uint8_t * __restrict__ w2, const uint8_t * __restrict__ w2_sf, const float * __restrict__ g2_alpha,
        float * __restrict__ part_out, unsigned * __restrict__ done, __nv_bfloat16 * __restrict__ out, int n_experts, bool reuse) {
    __shared__ int s_ids[MAXSLOT], s_fpos[MAXSLOT];
    __shared__ unsigned s_mask;
    __shared__ float s_w[MAXT];
    __shared__ int2 s_h[MAXT][I / 8];
    __shared__ float s_hsf[MAXT][IB];
    __shared__ bool s_last;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n = T * TOPK, pos = blockIdx.x / G2_TILES, tile = blockIdx.x % G2_TILES;
    if (tid < n) s_ids[tid] = ids[tid];
    if (tid == 0) s_mask = 0;
    __syncthreads();
    const int e = s_ids[pos];
    if (e < 0 || e >= n_experts) return;
    if (__syncthreads_or(reuse && tid < pos && s_ids[tid] == e)) return;

    const int row = warp * 16 + (lane >> 1), q0 = lane & 1;
    const int grow = tile * G2_ROWS + row;
    const uint8_t * wrow = w2 + ((size_t) e * H + grow) * (I / 2);
    const uint8_t * sfe = w2_sf + (size_t) e * H * IB;
    uint4 wv[I / 128][2];
    uint32_t sv[I / 128];
#pragma unroll
    for (int i = 0; i < I / 128; ++i) {
        const int q = q0 + 2 * i;
        const uint4 * p = reinterpret_cast<const uint4 *>(wrow + q * 32);
        wv[i][0] = __ldcs(p);
        wv[i][1] = __ldcs(p + 1);
        sv[i] = __ldg(reinterpret_cast<const unsigned int *>(sfe + sf_quad_offset(grow, q, IQ)));
    }
    // first position of every selection (for the tile reduction) and this expert's tokens
    int fp = -1;
    if (tid < n) {
        const int ej = s_ids[tid];
        if (ej >= 0 && ej < n_experts) {
            fp = tid;
            for (int k = 0; reuse && k < tid; ++k)
                if (s_ids[k] == ej) { fp = k; break; }
        }
        s_fpos[tid] = fp;
        if (ej == e && (reuse || tid == pos)) {
            atomicOr(&s_mask, 1u << (tid / TOPK));
            s_w[tid / TOPK] = tw[tid];
        }
    }
    const int n_active = __syncthreads_count(tid < n && fp == tid);
    asm volatile("griddepcontrol.wait;" ::: "memory");   // gemm1's h (programmatic dependent launch)
    const unsigned mask = s_mask;
    const int te = __popc(mask);
    for (int c = tid; c < te * (I / 8); c += G2_THREADS) {
        const int rk = c / (I / 8), k = c % (I / 8);
        unsigned m = mask;
        for (int j = 0; j < rk; ++j) m &= m - 1;
        const int t = __ffs(m) - 1;
        s_h[rk][k] = fp4x8_to_i8(__ldcg(hq + ((size_t) pos * MAXT + t) * (I / 8) + k));
    }
    for (int c = tid; c < te * IB; c += G2_THREADS) {
        const int rk = c / IB, k = c % IB;
        unsigned m = mask;
        for (int j = 0; j < rk; ++j) m &= m - 1;
        const int t = __ffs(m) - 1;
        s_hsf[rk][k] = __ldcg(hsf + ((size_t) pos * MAXT + t) * IB + k);
    }
    __syncthreads();

    float part[MAXT];
#pragma unroll
    for (int rk = 0; rk < MAXT; ++rk) part[rk] = 0.f;
#pragma unroll
    for (int i = 0; i < I / 128; ++i) {
        const int q = q0 + 2 * i;
        int2 wi[8];
        const uint32_t * ww = reinterpret_cast<const uint32_t *>(&wv[i][0]);
#pragma unroll
        for (int k = 0; k < 8; ++k) wi[k] = fp4x8_to_i8(ww[k]);
        float wsf[4];
#pragma unroll
        for (int b = 0; b < 4; ++b) wsf[b] = e4m3_to_float((sv[i] >> (8 * b)) & 0xFF);
#pragma unroll
        for (int rk = 0; rk < MAXT; ++rk) {
            if (rk >= te) break;
            float p = 0.f;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                int s = 0;
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const int2 hv = s_h[rk][q * 8 + b * 2 + k];
                    s = __dp4a(wi[b * 2 + k].x, hv.x, s);
                    s = __dp4a(wi[b * 2 + k].y, hv.y, s);
                }
                p = fmaf((float) s, wsf[b] * s_hsf[rk][q * 4 + b], p);
            }
            part[rk] += p;
        }
    }
    const float alpha = g2_alpha[e] * 0.25f;
    unsigned m = mask;
#pragma unroll
    for (int rk = 0; rk < MAXT; ++rk) {
        if (rk >= te) break;
        const float v = part[rk] + __shfl_xor_sync(0xffffffff, part[rk], 1);
        const int t = __ffs(m) - 1;
        m &= m - 1;
        if (q0 == 0) part_out[((size_t) pos * MAXT + t) * H + grow] = v * alpha * s_w[t];
    }
    // last CTA of this row tile: sum the partials in top-k order, write bf16, re-arm the counter
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(done + tile, 1u) == (unsigned) n_active - 1;
    __syncthreads();
    if (s_last) {
        __threadfence();
        for (int c = tid; c < T * (G2_ROWS / 4); c += G2_THREADS) {
            const int t = c / (G2_ROWS / 4), r4 = (c % (G2_ROWS / 4)) * 4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            for (int k = 0; k < TOPK; ++k) {
                const int p = s_fpos[t * TOPK + k];
                if (p < 0) continue;
                const float4 a = __ldcg(reinterpret_cast<const float4 *>(part_out + ((size_t) p * MAXT + t) * H + tile * G2_ROWS + r4));
                v.x += a.x; v.y += a.y; v.z += a.z; v.w += a.w;
            }
            __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
            uint2 packed;
            packed.x = *reinterpret_cast<uint32_t *>(&lo);
            packed.y = *reinterpret_cast<uint32_t *>(&hi);
            *reinterpret_cast<uint2 *>(out + (size_t) t * H + tile * G2_ROWS + r4) = packed;
        }
        if (tid == 0) done[tile] = 0;
    }
}

}  // namespace

// x [T, 2048] bf16 (T <= 8); topk_ids [T, 8] int32; topk_w [T, 8] fp32;
// w13 [E, 1024, 1024] uint8 ([up; gate]); w13_sf swizzled e4m3 [E, 1024, 128]; g1_alpha [E];
// w2 [E, 2048, 256] uint8; w2_sf swizzled e4m3 [E, 2048, 32]; g2_alpha [E]; a1_gs / a2_gs fp32 scalars (device).
at::Tensor nvfp4_moe_decode(const at::Tensor & x, const at::Tensor & topk_ids, const at::Tensor & topk_w,
                            const at::Tensor & w13, const at::Tensor & w13_sf, const at::Tensor & g1_alpha,
                            const at::Tensor & a1_gs, const at::Tensor & w2, const at::Tensor & w2_sf,
                            const at::Tensor & g2_alpha, const at::Tensor & a2_gs,
                            at::Tensor & hq, at::Tensor & hsf, at::Tensor & part_out,
                            at::Tensor & done, at::Tensor & out, bool reuse) {
    TORCH_CHECK(x.is_cuda() && x.dim()==2 && x.size(1)==H);
    TORCH_CHECK(topk_ids.dim()==2 && topk_w.dim()==2);
    TORCH_CHECK(topk_w.size(0)==x.size(0) && topk_w.size(1)==TOPK);
    TORCH_CHECK(w13.dim()==3 && w2.dim()==3 && w13.size(0)==256 && w2.size(0)==256);
    TORCH_CHECK(w13.scalar_type()==at::kByte && w2.scalar_type()==at::kByte);
    TORCH_CHECK(w13_sf.dim()==3 && w2_sf.dim()==3);
    TORCH_CHECK(w13_sf.scalar_type()==at::kByte && w2_sf.scalar_type()==at::kByte);
    TORCH_CHECK(w13_sf.size(0)==256 && w13_sf.size(1)==2*I && w13_sf.size(2)==HB);
    TORCH_CHECK(w2_sf.size(0)==256 && w2_sf.size(1)==H && w2_sf.size(2)==IB);
    TORCH_CHECK(g1_alpha.dim()==1 && g2_alpha.dim()==1 && g1_alpha.scalar_type()==at::kFloat && g2_alpha.scalar_type()==at::kFloat);
    TORCH_CHECK(a1_gs.dim()==1 && a2_gs.dim()==1);
    TORCH_CHECK(hq.dim()==1 && hsf.dim()==1 && part_out.dim()==1 && done.dim()==1 && out.dim()==2);
    // Contiguous byte intervals, not only equal pointers: reject shifted aliases.
    const at::Tensor* writes[]={&hq,&hsf,&part_out,&done,&out};
    const at::Tensor* reads[]={&x,&topk_ids,&topk_w,&w13,&w13_sf,&g1_alpha,&a1_gs,&w2,&w2_sf,&g2_alpha,&a2_gs};
    auto disjoint=[](const at::Tensor* a,const at::Tensor* b) {
        const auto a0=reinterpret_cast<uintptr_t>(a->data_ptr());
        const auto b0=reinterpret_cast<uintptr_t>(b->data_ptr());
        TORCH_CHECK(a0 + a->nbytes() <= b0 || b0 + b->nbytes() <= a0,"overlapping writable/input storage");
    };
    for (int j=0;j<5;++j) {
        for (const auto* r:reads) disjoint(writes[j],r);
        for (int k=0;k<j;++k) disjoint(writes[j],writes[k]);
    }
    const int T = x.size(0);
    TORCH_CHECK(T >= 1 && T <= MAXT && x.size(1) == H && x.scalar_type() == at::kBFloat16 && x.is_contiguous());
    TORCH_CHECK(topk_ids.scalar_type() == at::kInt && topk_ids.size(0) == T && topk_ids.size(1) == TOPK && topk_ids.is_contiguous());
    TORCH_CHECK(topk_w.scalar_type() == at::kFloat && topk_w.is_contiguous());
    TORCH_CHECK(w13.size(1) == W13_ROWS && w13.size(2) == H / 2 && w2.size(1) == H && w2.size(2) == I / 2);
    TORCH_CHECK(w13_sf.numel() == w13.size(0) * W13_ROWS * HB && w2_sf.numel() == w2.size(0) * H * IB);
    TORCH_CHECK(w13.size(0) == w2.size(0) && g1_alpha.numel() == w13.size(0) && g2_alpha.numel() == w13.size(0));
    TORCH_CHECK(a1_gs.scalar_type() == at::kFloat && a2_gs.scalar_type() == at::kFloat && a1_gs.numel() == 1 && a2_gs.numel() == 1);
    const c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t s = c10::cuda::getCurrentCUDAStream();
    // Per-invocation explicit workspace; same stream owns it until completion.
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
    const int g1_smem = T * ((H / 8) * (int) sizeof(int2) + HB * (int) sizeof(float));
    cudaLaunchAttribute lattr[1];
    lattr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    lattr[0].val.programmaticStreamSerializationAllowed = 0;
    {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(T * TOPK * (I / 16));
        cfg.blockDim = dim3(G1_THREADS);
        cfg.dynamicSmemBytes = g1_smem;
        cfg.stream = s;
        cfg.attrs = lattr;
        cfg.numAttrs = 1;
        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, moe_gemm1_kernel, (const __nv_bfloat16 *) x.data_ptr(), T,
                                          (const int *) topk_ids.data_ptr<int>(), (const uint8_t *) w13.data_ptr(),
                                          (const uint8_t *) w13_sf.data_ptr(), (const float *) g1_alpha.data_ptr<float>(),
                                          (const float *) a1_gs.data_ptr<float>(), (const float *) a2_gs.data_ptr<float>(),
                                          (int) w13.size(0), reinterpret_cast<uint32_t *>(hq.data_ptr<int>()),
                                          hsf.data_ptr<float>(), reuse));
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(T * TOPK * G2_TILES);
    cfg.blockDim = dim3(G2_THREADS);
    cfg.stream = s;
    cfg.attrs = lattr;
    cfg.numAttrs = 1;
    const uint32_t * hq_p = reinterpret_cast<const uint32_t *>(hq.data_ptr<int>());
    const float * hsf_p = hsf.data_ptr<float>();
    const uint8_t * w2_p = reinterpret_cast<const uint8_t *>(w2.data_ptr());
    const uint8_t * w2sf_p = reinterpret_cast<const uint8_t *>(w2_sf.data_ptr());
    unsigned * done_p = reinterpret_cast<unsigned *>(done.data_ptr<int>());
    __nv_bfloat16 * out_p = reinterpret_cast<__nv_bfloat16 *>(out.data_ptr());
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, moe_gemm2_kernel, T, (const int *) topk_ids.data_ptr<int>(),
                                      (const float *) topk_w.data_ptr<float>(), hq_p, hsf_p, w2_p, w2sf_p,
                                      (const float *) g2_alpha.data_ptr<float>(), part_out.data_ptr<float>(), done_p,
                                      out_p, (int) w2.size(0), reuse));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("apply", &nvfp4_moe_decode); }
