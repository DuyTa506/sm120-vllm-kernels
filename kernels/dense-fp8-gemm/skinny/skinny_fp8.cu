// Skinny-M FP8 GEMM for SM120 decode: out[M,N] = bf16( sa[m] * ( sb[n] * sum_k a[m,k] * w[n,k] ) )
// (same epilogue op order and rounding as vLLM's CUTLASS ScaledEpilogue; only the fp32 summation order
// of the K reduction differs).
//
// "Swap-AB" tensor-core GEMV: the weight is the MMA A operand (16 rows of N per warp), the activations
// are the MMA B operand (8 tokens per n8 tile, MT tiles -> M <= 8*MT). Fragments are loaded straight
// from global memory into registers (no smem staging): since the K reduction is order independent we
// permute K inside each 128-wide k-block so every thread issues full 16-byte loads, applying the same
// permutation to weights and activations. The block's KW warps split K; partial sums are reduced in
// shared memory in a fixed order (deterministic, batch-size independent per row).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

__device__ __forceinline__ uint4 ldg_stream(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint4 ldg_keep(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ void mma_e4m3(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One block = 16 output rows (n) x all M tokens; KW warps split K; each warp owns K/KW contiguous k.
// KB = 128-wide k-blocks a warp keeps in flight per iteration.
template <int MT, int KW, int KB>
__global__ void __launch_bounds__(KW * 32) skinny_fp8_kernel(
    __nv_bfloat16* __restrict__ out, const uint8_t* __restrict__ a, const uint8_t* __restrict__ w,
    const float* __restrict__ sa, const float* __restrict__ sb, int M, int N, int K) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, q = lane & 3;
  const int n0 = blockIdx.x * 16;
  const int kper = K / KW;  // multiple of 128 (host checks)
  const int kbeg = warp * kper;

  const uint8_t* w0 = w + (size_t)(n0 + g) * K + kbeg + 16 * q;
  const uint8_t* w1 = w0 + (size_t)8 * K;
  const uint8_t* ap[MT];
#pragma unroll
  for (int t = 0; t < MT; ++t) {
    int m = 8 * t + g;
    ap[t] = (m < M) ? a + (size_t)m * K + kbeg + 16 * q : nullptr;
  }

  float acc[MT][4];
#pragma unroll
  for (int t = 0; t < MT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;

  for (int k = 0; k < kper; k += 128 * KB) {
    uint4 wa[KB][2], wb[KB][2];
#pragma unroll
    for (int j = 0; j < KB; ++j) {
      wa[j][0] = ldg_stream(w0 + k + 128 * j);
      wa[j][1] = ldg_stream(w0 + k + 128 * j + 64);
      wb[j][0] = ldg_stream(w1 + k + 128 * j);
      wb[j][1] = ldg_stream(w1 + k + 128 * j + 64);
    }
#pragma unroll
    for (int j = 0; j < KB; ++j) {
      const uint32_t* ra = reinterpret_cast<const uint32_t*>(&wa[j][0]);  // 8 words: row g
      const uint32_t* rb = reinterpret_cast<const uint32_t*>(&wb[j][0]);  // 8 words: row g+8
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        uint4 x0 = make_uint4(0, 0, 0, 0), x1 = make_uint4(0, 0, 0, 0);
        if (ap[t] != nullptr) {
          x0 = ldg_keep(ap[t] + k + 128 * j);
          x1 = ldg_keep(ap[t] + k + 128 * j + 64);
        }
        const uint32_t xb[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int i = 0; i < 4; ++i)
          mma_e4m3(acc[t], ra[2 * i], rb[2 * i], ra[2 * i + 1], rb[2 * i + 1], xb[2 * i], xb[2 * i + 1]);
      }
    }
  }

  // acc[t]: c0,c1 -> (row g, tokens 8t+2q, 8t+2q+1); c2,c3 -> (row g+8, same tokens)
  if constexpr (KW == 1) {
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        int r = g + (e >> 1) * 8, m = 8 * t + 2 * q + (e & 1);
        if (m < M) {
          float v = sb[n0 + r] * acc[t][e];
          out[(size_t)m * N + n0 + r] = __float2bfloat16_rn(sa[m] * v);
        }
      }
  } else {
    __shared__ float red[KW][MT * 8][17];
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) red[warp][8 * t + 2 * q + (e & 1)][g + (e >> 1) * 8] = acc[t][e];
    __syncthreads();
    for (int idx = threadIdx.x; idx < MT * 8 * 16; idx += KW * 32) {
      int m = idx >> 4, r = idx & 15;
      if (m < M) {
        float s = 0.f;
#pragma unroll
        for (int ww = 0; ww < KW; ++ww) s += red[ww][m][r];
        float v = sb[n0 + r] * s;
        out[(size_t)m * N + n0 + r] = __float2bfloat16_rn(sa[m] * v);
      }
    }
  }
}

template <int MT, int KW, int KB>
void launch(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& w, const torch::Tensor& sa,
            const torch::Tensor& sb, int M, int N, int K, cudaStream_t st) {
  TORCH_CHECK(K % (KW * 128 * KB) == 0, "K must be a multiple of KW*128*KB");
  dim3 grid(N / 16), block(KW * 32);
  skinny_fp8_kernel<MT, KW, KB><<<grid, block, 0, st>>>(
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), reinterpret_cast<const uint8_t*>(a.data_ptr()),
      reinterpret_cast<const uint8_t*>(w.data_ptr()), sa.data_ptr<float>(), sb.data_ptr<float>(), M, N, K);
}

}  // namespace

// out [M,N] bf16, a [M,K] fp8 contiguous, w [N,K] fp8 contiguous (vLLM's B is w.t()), sa [M] / [M,1]
// fp32, sb [N] / [N,1] fp32. kw = warps splitting K per block, kb = k-blocks in flight per iteration.
void skinny_fp8_mm(torch::Tensor out, torch::Tensor a, torch::Tensor w, torch::Tensor sa, torch::Tensor sb,
                   int64_t kw, int64_t kb) {
  const int M = a.size(0), K = a.size(1), N = w.size(0);
  TORCH_CHECK(w.size(1) == K && N % 16 == 0 && M >= 1 && M <= 64);
  TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && sa.numel() == M && sb.numel() == N);
  TORCH_CHECK(sa.scalar_type() == at::kFloat && sb.scalar_type() == at::kFloat);
  auto st = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  const int mt = (M + 7) / 8;
#define KWKB(MT_, KW_, KB_)                                                      \
  if (kw == KW_ && kb == KB_) {                                                   \
    if constexpr (KW_ * MT_ * 8 * 17 * 4 <= 48 * 1024)                            \
      return launch<MT_, KW_, KB_>(out, a, w, sa, sb, M, N, K, st);               \
  }
#define DISPATCH_KW_KB(MT_)                                                       \
  KWKB(MT_, 1, 4) KWKB(MT_, 2, 4) KWKB(MT_, 4, 4) KWKB(MT_, 4, 2) KWKB(MT_, 8, 2) \
  KWKB(MT_, 8, 4) KWKB(MT_, 8, 1) KWKB(MT_, 16, 1) KWKB(MT_, 16, 2)
  switch (mt) {
    case 1: { DISPATCH_KW_KB(1) break; }
    case 2: { DISPATCH_KW_KB(2) break; }
    case 3: { DISPATCH_KW_KB(3) break; }
    case 4: { DISPATCH_KW_KB(4) break; }
    case 5: case 6: { DISPATCH_KW_KB(6) break; }
    case 7: case 8: { DISPATCH_KW_KB(8) break; }
  }
  TORCH_CHECK(false, "unsupported kw/kb combination");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("mm", &skinny_fp8_mm); }
