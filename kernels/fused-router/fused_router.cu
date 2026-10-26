// Fused MoE router for decode: router GEMM + scoring + top-k + weight normalisation (+ shared-expert gate)
// in ONE kernel for 1..32 tokens per token block, SM80+ tensor cores (written for SM120, no tcgen05/TMA).
//
// Design (borrowed from DeepGEMM bf16_mega_gate, which targets SM100 UMMA + clusters; see README):
//   * "swap-AB": the experts are the MMA M dimension (16 rows of the weight matrix per CTA group),
//     the tokens are the MMA N dimension (8 tokens per n-tile), so M=1..32 tokens cost 1..4 n-tiles.
//     mma.sync.m16n8k16 bf16 -> fp32, accumulating in fp32.
//   * split-K across CTAs: grid = (groups * SPLIT, token_blocks). Each CTA streams its 16 x (K/SPLIT)
//     weight slice exactly once, with 16-byte loads (the k order inside a 32-wide step is permuted
//     identically for A and B, which is legal for a dot product).
//   * the partial fp32 logits go to a global scratch (fixed slot per split => deterministic sum order);
//     the LAST CTA to arrive (threadfence + atomic ticket, the "threadFenceReduction" pattern) sums the
//     splits in order and does scoring + top-k + normalisation, one warp per token. No co-residency
//     assumption, so no deadlock when other streams (the shared-expert aux stream) occupy SMs.
//   * the ticket counter is reset by the last CTA, so the kernel can be replayed from a CUDA graph.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cfloat>

namespace cg = cooperative_groups;

namespace {

enum Scoring { SC_SOFTMAX = 0, SC_SIGMOID = 1, SC_SQRTSOFTPLUS = 2, SC_NONE = 3 };
enum Flags { F_RENORM = 1, F_ROUND_BF16 = 2, F_BIAS = 4, F_WRITE_LOGITS = 8, F_SH_FP32 = 16 };

struct Args {
  const __nv_bfloat16* x;      // [M, K], row stride ldx
  const __nv_bfloat16* w;      // [R, K] contiguous; rows E..R-1 are shared-expert gate rows
  const float* bias;           // [E] selection bias (or null)
  float* scratch;              // [token_blocks][split][TB][Rp]
  int* counters;               // [token_blocks], zero before the first launch, self-resetting
  float* topk_w;               // [M, topk]
  int* topk_i;                 // [M, topk]
  void* sh_out;                // [M, S] bf16 (or fp32 with F_SH_FP32): sigmoid(logit) of the gate rows
  __nv_bfloat16* logits_out;   // [M, E] (F_WRITE_LOGITS)
  const bool* pad;             // [M] optional: padded rows get topk_i = -1 (vLLM VLLM_MOE_SKIP_PADDING contract)
  int M, K, ldx, E, S, R, Rp, G, split, topk, scoring, flags, TB;
  int ablate;                  // benchmark-only: 0 full kernel, 1 stop after the partial stores, 2 + ticket, 3 + partial reads
  const __nv_bfloat16* pf_w;   // optional: next layer's router weight [R,K]; this kernel prefetches its slice of it into L2 (evict_last)
  int accmode;                 // 0: chain the tensor-core accumulator over K; 1: fresh zero accumulator per 32-wide step, added in fp32 (RN)
  int relacq;                  // 1: one release/acquire pair by thread 0 (CUTLASS semaphore style) instead of per-thread fences
  float scale;
};

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float bf16_round(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// order-preserving float -> uint32 map, so integer redux (sm_80+) can do a float max
__device__ __forceinline__ unsigned f2ord(float v) {
  unsigned u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float score_fn(float x, int scoring) {
  if (scoring == SC_SIGMOID) return 1.0f / (1.0f + expf(-x));
  if (scoring == SC_SQRTSOFTPLUS) return sqrtf(x > 20.f ? x : log1pf(expf(x)));
  return x;
}

// scoring + top-k + normalisation of ONE token by one warp. v[j] = logit of expert lane*NPL+j (bf16-rounded if requested).
template <int NPL>
__device__ __forceinline__ void route_token(const Args& a, int token, int lane, float (&v)[NPL]) {
  const bool do_bias = a.flags & F_BIAS, renorm = a.flags & F_RENORM, rnd = a.flags & F_ROUND_BF16;
  // fast path: softmax without bias on bf16-rounded logits. The ranking can then use the 16-bit bf16 logit order
  // packed with the inverted expert index in one uint32, so each pick is ONE redux.max and ties go to the lower
  // index by construction (the same as vLLM's topkGating, whose softmax is monotone on distinct bf16 logits).
  const bool fast = (a.scoring == SC_SOFTMAX) && !do_bias && rnd;
  float ssum = 0.f, mywt = 0.f;
  int myid = 0;
  if (fast) {
    unsigned key[NPL];
#pragma unroll
    for (int j = 0; j < NPL; ++j) {
      const int e = lane * NPL + j;
      unsigned bits = (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(v[j]));
      bits = (bits == 0x8000u) ? 0u : bits;   // -0 == +0 for ties
      const unsigned ord = (bits & 0x8000u) ? (~bits & 0xFFFFu) : (bits | 0x8000u);
      key[j] = (e < a.E) ? ((ord << 16) | (0xFFFFu - (unsigned)e)) : 0u;
    }
    auto lane_max = [&]() {
      unsigned mk = key[0];
#pragma unroll
      for (int j = 1; j < NPL; ++j) mk = max(mk, key[j]);
      return mk;
    };
    auto unord_logit = [](unsigned W) {   // key -> bf16 logit as float
      const unsigned ord = W >> 16;
      const unsigned bits = (ord & 0x8000u) ? (ord & 0x7FFFu) : (~ord & 0xFFFFu);
      return __uint_as_float(bits << 16);
    };
    const float mx = unord_logit(__reduce_max_sync(0xffffffffu, lane_max()));
    float ls = 0.f;
#pragma unroll
    for (int j = 0; j < NPL; ++j) ls += (lane * NPL + j < a.E) ? expf(v[j] - mx) : 0.f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ls += __shfl_xor_sync(0xffffffffu, ls, o);
    const float inv = 1.0f / ls;
    float mylogit = 0.f;
    for (int k = 0; k < a.topk; ++k) {
      const unsigned W = __reduce_max_sync(0xffffffffu, lane_max());
#pragma unroll
      for (int j = 0; j < NPL; ++j) key[j] = (key[j] == W) ? 0u : key[j];
      if (lane == k) { mylogit = unord_logit(W); myid = (int)(0xFFFFu - (W & 0xFFFFu)); }
    }
    mywt = expf(mylogit - mx) * inv;
    for (int k = 0; k < a.topk; ++k) ssum += __shfl_sync(0xffffffffu, mywt, k);
  } else {
    // general path: any scoring, optional selection bias, fp32 or bf16 logits
    float sc[NPL], sel[NPL];
    if (a.scoring == SC_SOFTMAX) {
      float mx = -FLT_MAX;
#pragma unroll
      for (int j = 0; j < NPL; ++j) mx = fmaxf(mx, v[j]);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
      float ls = 0.f;
#pragma unroll
      for (int j = 0; j < NPL; ++j) { sc[j] = (lane * NPL + j < a.E) ? expf(v[j] - mx) : 0.f; ls += sc[j]; }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) ls += __shfl_xor_sync(0xffffffffu, ls, o);
      const float inv = 1.0f / ls;
#pragma unroll
      for (int j = 0; j < NPL; ++j) sc[j] *= inv;
    } else {
#pragma unroll
      for (int j = 0; j < NPL; ++j) sc[j] = (lane * NPL + j < a.E) ? score_fn(v[j], a.scoring) : 0.f;
    }
#pragma unroll
    for (int j = 0; j < NPL; ++j) {
      const int e = lane * NPL + j;
      sel[j] = (e < a.E) ? (do_bias ? sc[j] + __ldg(a.bias + e) : sc[j]) : -INFINITY;
    }
    for (int k = 0; k < a.topk; ++k) {
      float bv = -INFINITY;
#pragma unroll
      for (int j = 0; j < NPL; ++j) bv = fmaxf(bv, sel[j]);
      const unsigned umax = __reduce_max_sync(0xffffffffu, f2ord(bv));
      const unsigned win = __ffs(__ballot_sync(0xffffffffu, f2ord(bv) == umax)) - 1;
      float uw = 0.f;
      int bi = 0;
      if (lane == win) {
        bool found = false;
#pragma unroll
        for (int j = 0; j < NPL; ++j) {
          if (!found && sel[j] == bv) { found = true; uw = sc[j]; bi = lane * NPL + j; sel[j] = -INFINITY; }
        }
      }
      uw = __shfl_sync(0xffffffffu, uw, win);
      bi = __shfl_sync(0xffffffffu, bi, win);
      ssum += uw;
      if (lane == k) { mywt = uw; myid = bi; }
    }
  }
  if (lane < a.topk) {
    float w = mywt;
    if (renorm) w = w / ssum;
    a.topk_w[(size_t)token * a.topk + lane] = w * a.scale;
    a.topk_i[(size_t)token * a.topk + lane] = (a.pad != nullptr && a.pad[token]) ? -1 : myid;
  }
}

template <int NT, int NW, int NPL>
__global__ void __launch_bounds__(NW * 32)
fused_router_kernel(const Args a) {
  constexpr int GW = NW < 8 ? NW : 8;   // warps that take part in the GEMM phase
  constexpr int UNR = 4;
  __shared__ float red[GW][NT * 128];
  __shared__ int s_last;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int r = lane >> 2, t = lane & 3;
  const int g = blockIdx.x % a.G, s = blockIdx.x / a.G, tb = blockIdx.y;
  const int row0 = g * 16;
  const int steps = a.K >> 5;
  const int s_begin = (int)(((long)steps * s) / a.split);
  const int s_end = (int)(((long)steps * (s + 1)) / a.split);

  if (warp < GW) {
    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;

    const int ra = row0 + r, rb = row0 + r + 8;
    const __nv_bfloat16* wa_p = a.w + (size_t)ra * a.K + 8 * t;
    const __nv_bfloat16* wb_p = a.w + (size_t)rb * a.K + 8 * t;
    const __nv_bfloat16* x_p[NT];
    bool x_ok[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const int tok = tb * a.TB + n * 8 + r;
      x_ok[n] = tok < a.M;
      x_p[n] = a.x + (size_t)(x_ok[n] ? tok : 0) * a.ldx + 8 * t;
    }
    const uint4 z4 = make_uint4(0, 0, 0, 0);
    for (int st0 = s_begin + warp; st0 < s_end; st0 += GW * UNR) {
      uint4 wa[UNR], wb[UNR], xv[UNR][NT];
#pragma unroll
      for (int u = 0; u < UNR; ++u) {
        const int st = st0 + u * GW;
        const bool ok = st < s_end;
        const int k0 = st << 5;
        wa[u] = (ok && ra < a.R) ? __ldg(reinterpret_cast<const uint4*>(wa_p + k0)) : z4;
        wb[u] = (ok && rb < a.R) ? __ldg(reinterpret_cast<const uint4*>(wb_p + k0)) : z4;
#pragma unroll
        for (int n = 0; n < NT; ++n)
          xv[u][n] = (ok && x_ok[n]) ? __ldg(reinterpret_cast<const uint4*>(x_p[n] + k0)) : z4;
      }
#pragma unroll
      for (int u = 0; u < UNR; ++u) {
#pragma unroll
        for (int n = 0; n < NT; ++n) {
          if (a.accmode == 0) {
            mma16816(acc[n], wa[u].x, wb[u].x, wa[u].y, wb[u].y, xv[u][n].x, xv[u][n].y);
            mma16816(acc[n], wa[u].z, wb[u].z, wa[u].w, wb[u].w, xv[u][n].z, xv[u][n].w);
          } else {
            float c[4] = {0.f, 0.f, 0.f, 0.f};
            mma16816(c, wa[u].x, wb[u].x, wa[u].y, wb[u].y, xv[u][n].x, xv[u][n].y);
            mma16816(c, wa[u].z, wb[u].z, wa[u].w, wb[u].w, xv[u][n].z, xv[u][n].w);
            acc[n][0] += c[0]; acc[n][1] += c[1]; acc[n][2] += c[2]; acc[n][3] += c[3];
          }
        }
      }
    }
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) red[warp][n * 128 + lane * 4 + i] = acc[n][i];
  }
  if (a.pf_w != nullptr) {   // overlap the next layer's cold weight fetch with this kernel's tail
    const int rows = min(16, a.R - row0), chunks = s_end - s_begin;   // 64-byte chunks per row slice
    for (int i = tid; i < rows * chunks; i += NW * 32) {
      const __nv_bfloat16* p = a.pf_w + (size_t)(row0 + i / chunks) * a.K + (size_t)(s_begin + i % chunks) * 32;
      asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(p));
    }
  }
  __syncthreads();

  // sum the GW warps in fixed order and write this CTA's partial logits [token][row]
  {
    const int ntok = min(a.TB, a.M - tb * a.TB);
    float* part = a.scratch + ((size_t)(tb * a.split + s) * a.TB) * a.Rp;
    for (int e = tid; e < NT * 128; e += NW * 32) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < GW; ++w) v += red[w][e];
      const int n = e >> 7, l = (e >> 2) & 31, i = e & 3;
      const int lr = l >> 2, lt = l & 3;
      const int row = row0 + lr + ((i & 2) ? 8 : 0);
      const int col = n * 8 + 2 * lt + (i & 1);
      if (col < ntok) part[(size_t)col * a.Rp + row] = v;
    }
  }
  if (a.ablate == 1) return;
  if (a.relacq) {
    __syncthreads();
    if (tid == 0) {
      int prev;
      asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], 1;" : "=r"(prev) : "l"(a.counters + tb) : "memory");
      s_last = (prev == (int)(a.G * a.split) - 1);
      if (s_last) a.counters[tb] = 0;   // re-arm for the next launch / graph replay
    }
    __syncthreads();
    if (!s_last) return;
  } else {
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      const int prev = atomicAdd(a.counters + tb, 1);
      s_last = (prev == (int)(a.G * a.split) - 1);
      if (s_last) a.counters[tb] = 0;   // re-arm for the next launch / graph replay
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
  }
  if (a.ablate == 2) return;

  // ---------------- last CTA of this token block: scoring + top-k, one warp per token ----------------
  // Lane layout: lane owns the NPL consecutive experts [lane*NPL, lane*NPL+NPL) so partials load as float4 and
  // "lowest lane, then lowest j" is "lowest expert index".
  const int ntok = min(a.TB, a.M - tb * a.TB);
  const bool rnd = a.flags & F_ROUND_BF16;
  for (int m = warp; m < ntok; m += NW) {
    const int token = tb * a.TB + m;
    float v[NPL];
    {
      const float* base = a.scratch + ((size_t)(tb * a.split) * a.TB + m) * a.Rp + lane * NPL;
      const size_t sstride = (size_t)a.TB * a.Rp;
      float acc[NPL];
#pragma unroll
      for (int i = 0; i < NPL; ++i) acc[i] = 0.f;
      for (int sp0 = 0; sp0 < a.split; sp0 += 4) {
        float4 t[4][NPL / 4];
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
          for (int c = 0; c < NPL / 4; ++c)
            t[u][c] = (sp0 + u < a.split && lane * NPL + 4 * c < a.Rp)
                          ? __ldcg(reinterpret_cast<const float4*>(base + (size_t)(sp0 + u) * sstride + 4 * c))
                          : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int u = 0; u < 4; ++u)
          if (sp0 + u < a.split) {
#pragma unroll
            for (int c = 0; c < NPL / 4; ++c) {
              acc[4 * c + 0] += t[u][c].x; acc[4 * c + 1] += t[u][c].y;
              acc[4 * c + 2] += t[u][c].z; acc[4 * c + 3] += t[u][c].w;
            }
          }
      }
#pragma unroll
      for (int j = 0; j < NPL; ++j) {
        const int e = lane * NPL + j;
        float sum = acc[j];
        if (e < a.E) {
          if (rnd) sum = bf16_round(sum);
          if (a.flags & F_WRITE_LOGITS) a.logits_out[(size_t)token * a.E + e] = __float2bfloat16_rn(sum);
          v[j] = sum;
        } else {
          v[j] = -FLT_MAX;
        }
      }
    }
    // shared-expert gate rows: sigmoid(bf16 logit) like F.sigmoid(expert_gate(x))
    if (lane < a.S) {
      float sum = 0.f;
#pragma unroll 4
      for (int sp = 0; sp < a.split; ++sp)
        sum += __ldcg(a.scratch + ((size_t)(tb * a.split + sp) * a.TB + m) * a.Rp + a.E + lane);
      sum = bf16_round(sum);
      const float sg = 1.0f / (1.0f + expf(-sum));
      if (a.flags & F_SH_FP32) reinterpret_cast<float*>(a.sh_out)[(size_t)token * a.S + lane] = sg;
      else reinterpret_cast<__nv_bfloat16*>(a.sh_out)[(size_t)token * a.S + lane] = __float2bfloat16_rn(sg);
    }
    if (a.ablate == 3) {   // benchmark-only: keep the loads alive, skip scoring/top-k
      float t0 = 0.f;
#pragma unroll
      for (int j = 0; j < NPL; ++j) t0 += v[j];
      if (lane < a.topk) { a.topk_w[(size_t)token * a.topk + lane] = t0; a.topk_i[(size_t)token * a.topk + lane] = lane; }
      continue;
    }

    route_token<NPL>(a, token, lane, v);
  }
}


// ------------------------------------------------------------------------------------------------------------
// Cluster variant: ONE cluster of CS CTAs per token block (thread block clusters + distributed shared memory, which
// SM120 supports up to 16 CTAs: results/cluster_probe.txt). CTA `rank` owns the row groups g = rank + slot*CS, streams
// them over the whole K (no split-K, no global scratch, no ticket), the warps split K and are summed in a fixed order
// in shared memory. After a cluster barrier every token is routed by one warp that gathers the logits of all 257 rows
// from the CTAs' shared memory (ld.shared::cluster), so the cross-CTA hand-off is one cluster barrier + one DSMEM read.
// ------------------------------------------------------------------------------------------------------------
template <int NT, int NW, int CS, int SLOTS>
__global__ void __cluster_dims__(CS, 1, 1) __launch_bounds__(NW * 32)
fused_router_cluster_kernel(const Args a) {
  constexpr int NPL = 8;
  constexpr int UNR = (NT <= 2) ? 4 : 2;
  __shared__ float red[NW][NT * 128];
  __shared__ __align__(16) float lgs[SLOTS][32][16];   // [slot][token][row within the 16-row group]
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = (int)cluster.block_rank();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int r = lane >> 2, t = lane & 3;
  const int tb = blockIdx.y;
  const int steps = a.K >> 5;
  const int s_begin = (steps * warp) / NW, s_end = (steps * (warp + 1)) / NW;   // this warp's K slice

  float acc[SLOTS][NT][4];
#pragma unroll
  for (int sl = 0; sl < SLOTS; ++sl)
#pragma unroll
    for (int n = 0; n < NT; ++n) acc[sl][n][0] = acc[sl][n][1] = acc[sl][n][2] = acc[sl][n][3] = 0.f;
  const __nv_bfloat16* x_p[NT];
  bool x_ok[NT];
#pragma unroll
  for (int n = 0; n < NT; ++n) {
    const int tok = tb * a.TB + n * 8 + r;
    x_ok[n] = tok < a.M;
    x_p[n] = a.x + (size_t)(x_ok[n] ? tok : 0) * a.ldx + 8 * t;
  }
  const uint4 z4 = make_uint4(0, 0, 0, 0);
  for (int st0 = s_begin; st0 < s_end; st0 += UNR) {
    uint4 wa[SLOTS][UNR], wb[SLOTS][UNR], xv[UNR][NT];
#pragma unroll
    for (int u = 0; u < UNR; ++u) {
      const int st = st0 + u;
      const bool ok = st < s_end;
      const int k0 = st << 5;
#pragma unroll
      for (int sl = 0; sl < SLOTS; ++sl) {
        const int row0 = (rank + sl * CS) * 16;
        const int ra = row0 + r, rb = row0 + r + 8;
        wa[sl][u] = (ok && ra < a.R) ? __ldg(reinterpret_cast<const uint4*>(a.w + (size_t)ra * a.K + 8 * t + k0)) : z4;
        wb[sl][u] = (ok && rb < a.R) ? __ldg(reinterpret_cast<const uint4*>(a.w + (size_t)rb * a.K + 8 * t + k0)) : z4;
      }
#pragma unroll
      for (int n = 0; n < NT; ++n)
        xv[u][n] = (ok && x_ok[n]) ? __ldg(reinterpret_cast<const uint4*>(x_p[n] + k0)) : z4;
    }
#pragma unroll
    for (int u = 0; u < UNR; ++u)
#pragma unroll
      for (int sl = 0; sl < SLOTS; ++sl)
#pragma unroll
        for (int n = 0; n < NT; ++n) {
          mma16816(acc[sl][n], wa[sl][u].x, wb[sl][u].x, wa[sl][u].y, wb[sl][u].y, xv[u][n].x, xv[u][n].y);
          mma16816(acc[sl][n], wa[sl][u].z, wb[sl][u].z, wa[sl][u].w, wb[sl][u].w, xv[u][n].z, xv[u][n].w);
        }
  }
  if (a.pf_w != nullptr) {   // prefetch this CTA's rows of the next layer's weight into L2
    for (int sl = 0; sl < SLOTS; ++sl) {
      const int row0 = (rank + sl * CS) * 16;
      const int rows = min(16, a.R - row0), chunks = s_end - s_begin;
      for (int i = lane; i < rows * chunks; i += 32) {
        const __nv_bfloat16* p = a.pf_w + (size_t)(row0 + i / chunks) * a.K + (size_t)(s_begin + i % chunks) * 32;
        asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(p));
      }
    }
  }
  const int ntok_blk = min(a.TB, a.M - tb * a.TB);
#pragma unroll
  for (int sl = 0; sl < SLOTS; ++sl) {
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) red[warp][n * 128 + lane * 4 + i] = acc[sl][n][i];
    __syncthreads();
    for (int e = tid; e < NT * 128; e += NW * 32) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < NW; ++w) v += red[w][e];
      const int n = e >> 7, l = (e >> 2) & 31, i = e & 3;
      const int col = n * 8 + 2 * (l & 3) + (i & 1);
      const int row = (l >> 2) + ((i & 2) ? 8 : 0);
      if (col < ntok_blk) lgs[sl][col][row] = v;
    }
    __syncthreads();
  }
  cluster.sync();   // all logits of this token block are now visible cluster-wide
  if (a.ablate != 5) {
    const int m = rank + CS * warp;   // token handled by this warp (ntok <= 32 <= CS * NW)
    if (m < ntok_blk) {
      const int token = tb * a.TB + m;
      const bool rnd = a.flags & F_ROUND_BF16;
      float v[NPL];
#pragma unroll
      for (int c = 0; c < NPL / 4; ++c) {
        const int e0 = lane * NPL + 4 * c;
        const int grp = e0 >> 4;
        float4 q = make_float4(0.f, 0.f, 0.f, 0.f);
        if (grp < a.G) {
          const float* src = cluster.map_shared_rank(&lgs[grp / CS][m][e0 & 15], grp % CS);
          q = *reinterpret_cast<const float4*>(src);
        }
        const float qq[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int e = e0 + j;
          float sum = qq[j];
          if (e < a.E) {
            if (rnd) sum = bf16_round(sum);
            if (a.flags & F_WRITE_LOGITS) a.logits_out[(size_t)token * a.E + e] = __float2bfloat16_rn(sum);
            v[4 * c + j] = sum;
          } else {
            v[4 * c + j] = -FLT_MAX;
          }
        }
      }
      if (lane < a.S) {   // shared-expert gate rows
        const int e = a.E + lane, grp = e >> 4;
        float sum = *cluster.map_shared_rank(&lgs[grp / CS][m][e & 15], grp % CS);
        sum = bf16_round(sum);
        const float sg = 1.0f / (1.0f + expf(-sum));
        if (a.flags & F_SH_FP32) reinterpret_cast<float*>(a.sh_out)[(size_t)token * a.S + lane] = sg;
        else reinterpret_cast<__nv_bfloat16*>(a.sh_out)[(size_t)token * a.S + lane] = __float2bfloat16_rn(sg);
      }
      route_token<NPL>(a, token, lane, v);
    }
  }
  cluster.sync();   // keep every CTA's shared memory alive until all readers are done
}

__global__ void noop_kernel(int* p) { if (p && threadIdx.x == 1023) *p = 0; }

template <int NT, int NW, int NPL>
void launch_one(const Args& a, int token_blocks, cudaStream_t st) {
  dim3 grid(a.G * a.split, token_blocks);
  fused_router_kernel<NT, NW, NPL><<<grid, NW * 32, 0, st>>>(a);
}

template <int NT, int NPL>
void launch_nw(const Args& a, int nw, int token_blocks, cudaStream_t st) {
  switch (nw) {
    case 4: launch_one<NT, 4, NPL>(a, token_blocks, st); break;
    case 8: launch_one<NT, 8, NPL>(a, token_blocks, st); break;
    case 12: launch_one<NT, 12, NPL>(a, token_blocks, st); break;
    case 16: launch_one<NT, 16, NPL>(a, token_blocks, st); break;
    case 24: launch_one<NT, 24, NPL>(a, token_blocks, st); break;
    case 32: launch_one<NT, 32, NPL>(a, token_blocks, st); break;
    default: TORCH_CHECK(false, "nw must be 4, 8, 12, 16, 24 or 32");
  }
}

template <int NPL>
void launch_nt(const Args& a, int nt, int nw, int token_blocks, cudaStream_t st) {
  switch (nt) {
    case 1: launch_nw<1, NPL>(a, nw, token_blocks, st); break;
    case 2: launch_nw<2, NPL>(a, nw, token_blocks, st); break;
    case 3: launch_nw<3, NPL>(a, nw, token_blocks, st); break;
    case 4: launch_nw<4, NPL>(a, nw, token_blocks, st); break;
    default: TORCH_CHECK(false, "nt must be 1..4");
  }
}


template <int NT, int NW, int CS, int SLOTS>
void launch_cluster_one(const Args& a, int token_blocks, cudaStream_t st) {
  static bool once = [] {
    cudaFuncSetAttribute(fused_router_cluster_kernel<NT, NW, CS, SLOTS>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    return true;
  }();
  (void)once;
  fused_router_cluster_kernel<NT, NW, CS, SLOTS><<<dim3(CS, token_blocks), NW * 32, 0, st>>>(a);
}

template <int NT, int NW>
void launch_cluster_cs(const Args& a, int cs, int token_blocks, cudaStream_t st) {
  const int slots = (a.G + cs - 1) / cs;
  if (cs == 16 && slots == 1) launch_cluster_one<NT, NW, 16, 1>(a, token_blocks, st);
  else if (cs == 16 && slots == 2) launch_cluster_one<NT, NW, 16, 2>(a, token_blocks, st);
  else if (cs == 8 && slots == 2) launch_cluster_one<NT, NW, 8, 2>(a, token_blocks, st);
  else if (cs == 8 && slots == 3) launch_cluster_one<NT, NW, 8, 3>(a, token_blocks, st);
  else TORCH_CHECK(false, "cluster variant supports cs=16 with <=2 row-group slots (E+S <= 512 rows -> E <= 256 here) or cs=8 with 2..3 slots; got cs=", cs, " groups=", a.G);
}

template <int NT>
void launch_cluster_nw(const Args& a, int nw, int cs, int token_blocks, cudaStream_t st) {
  if (nw == 8) launch_cluster_cs<NT, 8>(a, cs, token_blocks, st);
  else if (nw == 16) launch_cluster_cs<NT, 16>(a, cs, token_blocks, st);
  else TORCH_CHECK(false, "cluster variant: nw must be 8 or 16");
}

}  // namespace

// x [M,K] bf16 (last dim contiguous), w [E+S,K] bf16 contiguous, scratch fp32, counters int32
void fused_router(torch::Tensor x, torch::Tensor w, int64_t E, int64_t topk, int64_t scoring, int64_t flags,
                  double scale, c10::optional<torch::Tensor> bias, torch::Tensor scratch, torch::Tensor counters,
                  torch::Tensor topk_w, torch::Tensor topk_i, c10::optional<torch::Tensor> sh_out,
                  c10::optional<torch::Tensor> logits_out, c10::optional<torch::Tensor> pad, c10::optional<torch::Tensor> pf, int64_t nt, int64_t nw,
                  int64_t split, int64_t ablate, int64_t relacq, int64_t accmode) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.stride(1) == 1, "x: bf16 [M,K], K contiguous");
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kBFloat16 && w.is_contiguous(), "w: bf16 contiguous [R,K]");
  const int M = x.size(0), K = x.size(1), R = w.size(0);
  TORCH_CHECK(w.size(1) == K && K % 32 == 0, "K must be a multiple of 32");
  TORCH_CHECK(E <= R && R - E <= 32 && topk >= 1 && topk <= 32 && topk <= E, "bad E/S/topk");
  TORCH_CHECK(E <= 512, "E <= 512");
  if (M == 0) return;
  c10::cuda::CUDAGuard guard(x.device());
  Args a{};
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.w = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  a.bias = bias.has_value() ? bias->data_ptr<float>() : nullptr;
  a.scratch = scratch.data_ptr<float>();
  a.counters = counters.data_ptr<int>();
  a.topk_w = topk_w.data_ptr<float>();
  a.topk_i = topk_i.data_ptr<int>();
  a.sh_out = sh_out.has_value() ? sh_out->data_ptr() : nullptr;
  a.logits_out = logits_out.has_value() ? reinterpret_cast<__nv_bfloat16*>(logits_out->data_ptr()) : nullptr;
  a.pad = pad.has_value() ? reinterpret_cast<const bool*>(pad->data_ptr()) : nullptr;
  if (pad.has_value()) TORCH_CHECK(pad->scalar_type() == torch::kBool && pad->numel() >= M, "pad: bool [M]");
  a.pf_w = pf.has_value() ? reinterpret_cast<const __nv_bfloat16*>(pf->data_ptr()) : nullptr;
  if (pf.has_value()) TORCH_CHECK(pf->is_cuda() && pf->scalar_type() == torch::kBFloat16 && pf->sizes() == w.sizes(), "pf: same shape as w");
  a.M = M; a.K = K; a.ldx = x.stride(0); a.E = E; a.R = R; a.S = R - E;
  a.G = (R + 15) / 16; a.Rp = a.G * 16;
  a.split = split; a.topk = topk; a.scoring = scoring; a.flags = flags; a.scale = (float)scale;
  a.TB = nt * 8; a.ablate = ablate; a.relacq = relacq; a.accmode = accmode;
  if (a.S > 0) TORCH_CHECK(sh_out.has_value(), "sh_out required when the weight has shared-gate rows");
  if (flags & F_BIAS) TORCH_CHECK(bias.has_value(), "bias required");
  if (flags & F_WRITE_LOGITS) TORCH_CHECK(logits_out.has_value(), "logits_out required");
  TORCH_CHECK(split >= 1 && split <= K / 32, "bad split");
  const int token_blocks = (M + a.TB - 1) / a.TB;
  const size_t need = (size_t)token_blocks * split * a.TB * a.Rp;
  TORCH_CHECK((size_t)scratch.numel() >= need, "scratch too small: need ", need, " floats");
  TORCH_CHECK(counters.numel() >= token_blocks, "counters too small");
  auto st = c10::cuda::getCurrentCUDAStream();
  if (E <= 128) launch_nt<4>(a, nt, nw, token_blocks, st);
  else if (E <= 256) launch_nt<8>(a, nt, nw, token_blocks, st);
  else launch_nt<16>(a, nt, nw, token_blocks, st);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void fused_router_cluster(torch::Tensor x, torch::Tensor w, int64_t E, int64_t topk, int64_t scoring, int64_t flags,
                          double scale, c10::optional<torch::Tensor> bias, torch::Tensor topk_w, torch::Tensor topk_i,
                          c10::optional<torch::Tensor> sh_out, c10::optional<torch::Tensor> logits_out,
                          c10::optional<torch::Tensor> pad, c10::optional<torch::Tensor> pf, int64_t nt, int64_t nw,
                          int64_t cs, int64_t ablate) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.stride(1) == 1, "x: bf16 [M,K], K contiguous");
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kBFloat16 && w.is_contiguous(), "w: bf16 contiguous [R,K]");
  const int M = x.size(0), K = x.size(1), R = w.size(0);
  TORCH_CHECK(w.size(1) == K && K % 32 == 0, "K must be a multiple of 32");
  TORCH_CHECK(E <= 256 && E <= R && R - E <= 32 && topk >= 1 && topk <= 32 && topk <= E, "cluster variant: E <= 256, S <= 32");
  TORCH_CHECK(nt >= 1 && nt <= 4, "nt in 1..4");
  if (M == 0) return;
  c10::cuda::CUDAGuard guard(x.device());
  Args a{};
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.w = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  a.bias = bias.has_value() ? bias->data_ptr<float>() : nullptr;
  a.topk_w = topk_w.data_ptr<float>();
  a.topk_i = topk_i.data_ptr<int>();
  a.sh_out = sh_out.has_value() ? sh_out->data_ptr() : nullptr;
  a.logits_out = logits_out.has_value() ? reinterpret_cast<__nv_bfloat16*>(logits_out->data_ptr()) : nullptr;
  a.pad = pad.has_value() ? reinterpret_cast<const bool*>(pad->data_ptr()) : nullptr;
  a.pf_w = pf.has_value() ? reinterpret_cast<const __nv_bfloat16*>(pf->data_ptr()) : nullptr;
  a.M = M; a.K = K; a.ldx = x.stride(0); a.E = E; a.R = R; a.S = R - E;
  a.G = (R + 15) / 16; a.Rp = a.G * 16; a.split = 1; a.topk = topk; a.scoring = scoring; a.flags = flags;
  a.scale = (float)scale; a.TB = nt * 8; a.ablate = ablate;
  if (a.S > 0) TORCH_CHECK(sh_out.has_value(), "sh_out required when the weight has shared-gate rows");
  if (flags & F_BIAS) TORCH_CHECK(bias.has_value(), "bias required");
  if (flags & F_WRITE_LOGITS) TORCH_CHECK(logits_out.has_value(), "logits_out required");
  const int token_blocks = (M + a.TB - 1) / a.TB;
  auto st = c10::cuda::getCurrentCUDAStream();
  switch (nt) {
    case 1: launch_cluster_nw<1>(a, nw, cs, token_blocks, st); break;
    case 2: launch_cluster_nw<2>(a, nw, cs, token_blocks, st); break;
    case 3: launch_cluster_nw<3>(a, nw, cs, token_blocks, st); break;
    default: launch_cluster_nw<4>(a, nw, cs, token_blocks, st); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fr_noop(int64_t ncta, int64_t nthreads) {
  noop_kernel<<<ncta, nthreads, 0, c10::cuda::getCurrentCUDAStream()>>>(nullptr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_router_cluster", &fused_router_cluster, "cluster variant (DSMEM hand-off)");
  m.def("noop", &fr_noop, "empty kernel (launch-gap calibration)");
  m.def("fused_router", &fused_router, "fused router GEMM + scoring + top-k");
}
