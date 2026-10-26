// PDL microbenchmark: a chain of decode-like kernels in a CUDA graph, launched with and without
// programmatic dependent launch (cudaLaunchAttributeProgrammaticStreamSerialization +
// griddepcontrol.wait / launch_dependents).
//
// Per "layer" (M tokens, default 4 = c=1 MTP-2 padded):
//   glueA  : amax-quantize previous bf16 output to fp8 (stand-in for the Triton rms+quant kernels)
//   gemm1  : skinny FP8 GEMM, N=12288 K=2048 (GDN in_proj_qkvz, kernels/dense-fp8-gemm/skinny)
//   glueB  : quantize first 4096 columns of gemm1 output
//   tiny x E: 1-2 us dependent elementwise kernels (stand-ins for router, norms, topk, ...)
//   gemm2  : skinny FP8 GEMM, N=2048 K=4096 (GDN out_proj)
// Weights are distinct per layer (L*33.6 MB > 128 MB L2), so each GEMM reads cold DRAM like a real step.
//
// build: nvcc -O3 -std=c++17 -gencode=arch=compute_120a,code=sm_120a pdl_chain.cu -o pdl_chain
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void l2_bulk_prefetch(const void* p, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(bytes) : "memory");
}

__device__ __forceinline__ uint4 ldg_stream(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint4 ldg_keep(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ void mma_e4m3(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// MODE 0: no PDL instructions. 1: wait at top. 2: trigger + wait at top.
// 3: trigger, first-iteration weight loads issued, then wait, then activations.
// 4: trigger, bulk L2 prefetch of this CTA's whole weight slice, then wait.
// Same arithmetic and reduction order in every mode.
template <int MT, int KW, int KB, int MODE>
__global__ void __launch_bounds__(KW * 32) skinny_fp8(
    __nv_bfloat16* __restrict__ out, const uint8_t* __restrict__ a, const uint8_t* __restrict__ w,
    const float* __restrict__ sa, const float* __restrict__ sb, int M, int N, int K) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, q = lane & 3;
  const int n0 = blockIdx.x * 16;
  const int kper = K / KW;
  const int kbeg = warp * kper;
  if constexpr (MODE >= 2) pdl_trigger();
  if constexpr (MODE == 4) {
    if (warp == 0 && lane < 16) l2_bulk_prefetch(w + (size_t)(n0 + lane) * K, K);
  }
  const uint8_t* w0 = w + (size_t)(n0 + g) * K + kbeg + 16 * q;
  const uint8_t* w1 = w0 + (size_t)8 * K;
  const uint8_t* ap[MT];
#pragma unroll
  for (int t = 0; t < MT; ++t) { int m = 8 * t + g; ap[t] = (m < M) ? a + (size_t)m * K + kbeg + 16 * q : nullptr; }
  float acc[MT][4];
#pragma unroll
  for (int t = 0; t < MT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
  if constexpr (MODE == 1 || MODE == 2 || MODE == 4) pdl_wait();

  for (int k = 0; k < kper; k += 128 * KB) {
    uint4 wa[KB][2], wb[KB][2];
#pragma unroll
    for (int j = 0; j < KB; ++j) {
      wa[j][0] = ldg_stream(w0 + k + 128 * j);
      wa[j][1] = ldg_stream(w0 + k + 128 * j + 64);
      wb[j][0] = ldg_stream(w1 + k + 128 * j);
      wb[j][1] = ldg_stream(w1 + k + 128 * j + 64);
    }
    if constexpr (MODE == 3) { if (k == 0) pdl_wait(); }
#pragma unroll
    for (int j = 0; j < KB; ++j) {
      const uint32_t* ra = reinterpret_cast<const uint32_t*>(&wa[j][0]);
      const uint32_t* rb = reinterpret_cast<const uint32_t*>(&wb[j][0]);
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        uint4 x0 = make_uint4(0, 0, 0, 0), x1 = make_uint4(0, 0, 0, 0);
        if (ap[t] != nullptr) { x0 = ldg_keep(ap[t] + k + 128 * j); x1 = ldg_keep(ap[t] + k + 128 * j + 64); }
        const uint32_t xb[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int i = 0; i < 4; ++i)
          mma_e4m3(acc[t], ra[2 * i], rb[2 * i], ra[2 * i + 1], rb[2 * i + 1], xb[2 * i], xb[2 * i + 1]);
      }
    }
  }
  if constexpr (KW == 1) {
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        int r = g + (e >> 1) * 8, m = 8 * t + 2 * q + (e & 1);
        if (m < M) { float v = sb[n0 + r] * acc[t][e]; out[(size_t)m * N + n0 + r] = __float2bfloat16_rn(sa[m] * v); }
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

// glue: per-token amax + fp8 quantize. One block (256 thr) per token. PDLK: trigger+wait at top.
template <int PDLK>
__global__ void glue_quant(const __nv_bfloat16* __restrict__ in, int ld_in, uint8_t* __restrict__ a, float* __restrict__ sa, int K) {
  if constexpr (PDLK) { pdl_trigger(); pdl_wait(); }
  const int m = blockIdx.x;
  const __nv_bfloat16* x = in + (size_t)m * ld_in;
  float amax = 0.f;
  for (int k = threadIdx.x; k < K; k += 256) amax = fmaxf(amax, fabsf(__bfloat162float(x[k])));
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  __shared__ float sm[8];
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = amax;
  __syncthreads();
  amax = sm[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) amax = fmaxf(amax, sm[i]);
  float scale = amax > 0.f ? amax / 448.f : 1.f;
  float inv = 1.f / scale;
  if (threadIdx.x == 0) sa[m] = scale;
  for (int k = threadIdx.x; k < K; k += 256)
    a[(size_t)m * K + k] = (uint8_t)__nv_cvt_float_to_fp8(__bfloat162float(x[k]) * inv, __NV_SATFINITE, __NV_E4M3);
}

// tiny dependent elementwise kernel on a small fp32 buffer (router/norm/topk stand-in)
template <int PDLK>
__global__ void tiny_ew(float* __restrict__ buf, int n) {
  if constexpr (PDLK) { pdl_trigger(); pdl_wait(); }
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] = buf[i] * 0.999f + 0.001f;
}

struct Cfg { const char* name; int gemm_mode; bool gemm_attr; bool other_aware; };

template <typename... Args>
static void launch(void (*kern)(Args...), dim3 grid, dim3 block, cudaStream_t st, bool attr, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = 0; cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = attr ? 1 : 0;
  CK(cudaLaunchKernelEx(&cfg, kern, args...));
}

struct Bufs {
  int L, M, E;
  std::vector<uint8_t*> w1, w2;
  float *sb1, *sb2;
  __nv_bfloat16 *x0, *o1, *o2;
  uint8_t *a1, *a2;
  float *sa1, *sa2, *scratch;
};

template <int MODE>
static void gemm_launch(cudaStream_t st, bool attr, __nv_bfloat16* out, const uint8_t* a, const uint8_t* w,
                        const float* sa, const float* sb, int M, int N, int K, bool is1) {
  // M<=8 -> MT=1.  gemm1: (KW=1,KB=4); gemm2: (KW=8,KB=1)  as in dense-fp8-gemm/skinny/dispatch.py
  if (is1) launch(skinny_fp8<1, 1, 4, MODE>, dim3(N / 16), dim3(32), st, attr, out, a, w, sa, sb, M, N, K);
  else     launch(skinny_fp8<1, 8, 1, MODE>, dim3(N / 16), dim3(256), st, attr, out, a, w, sa, sb, M, N, K);
}
static void gemm_dispatch(int mode, cudaStream_t st, bool attr, __nv_bfloat16* out, const uint8_t* a, const uint8_t* w,
                          const float* sa, const float* sb, int M, int N, int K, bool is1) {
  switch (mode) {
    case 0: gemm_launch<0>(st, attr, out, a, w, sa, sb, M, N, K, is1); break;
    case 2: gemm_launch<2>(st, attr, out, a, w, sa, sb, M, N, K, is1); break;
    case 3: gemm_launch<3>(st, attr, out, a, w, sa, sb, M, N, K, is1); break;
    case 4: gemm_launch<4>(st, attr, out, a, w, sa, sb, M, N, K, is1); break;
  }
}

static void glue_dispatch(bool aware, cudaStream_t st, const __nv_bfloat16* in, int ld, uint8_t* a, float* sa, int K, int M) {
  if (aware) launch(glue_quant<1>, dim3(M), dim3(256), st, true, in, ld, a, sa, K);
  else       launch(glue_quant<0>, dim3(M), dim3(256), st, false, in, ld, a, sa, K);
}
static void tiny_dispatch(bool aware, cudaStream_t st, float* buf, int n) {
  if (aware) launch(tiny_ew<1>, dim3((n + 255) / 256), dim3(256), st, true, buf, n);
  else       launch(tiny_ew<0>, dim3((n + 255) / 256), dim3(256), st, false, buf, n);
}

static int enqueue_chain(const Bufs& B, const Cfg& c, cudaStream_t st) {
  int nk = 0;
  const bool oa = c.other_aware;
  for (int l = 0; l < B.L; ++l) {
    const __nv_bfloat16* prev = (l == 0) ? B.x0 : B.o2;
    glue_dispatch(oa, st, prev, 2048, B.a1, B.sa1, 2048, B.M); ++nk;
    gemm_dispatch(c.gemm_mode, st, c.gemm_attr, B.o1, B.a1, B.w1[l], B.sa1, B.sb1, B.M, 12288, 2048, true); ++nk;
    glue_dispatch(oa, st, B.o1, 12288, B.a2, B.sa2, 4096, B.M); ++nk;
    for (int e = 0; e < B.E; ++e) { tiny_dispatch(oa, st, B.scratch, 1024); ++nk; }
    gemm_dispatch(c.gemm_mode, st, c.gemm_attr, B.o2, B.a2, B.w2[l], B.sa2, B.sb2, B.M, 2048, 4096, false); ++nk;
  }
  return nk;
}

static float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
  int L = 16, M = 4, E = 6, reps = 100, samples = 9, rounds = 4, floor_n = 600, only = -1;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!strcmp(argv[i], "--L")) L = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--M")) M = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--E")) E = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--reps")) reps = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--samples")) samples = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--rounds")) rounds = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--floor")) floor_n = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--cfg")) only = atoi(argv[i + 1]);
  }
  cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  Bufs B; B.L = L; B.M = M; B.E = E;
  srand(1);
  auto rand_bytes = [&](size_t n) {  // finite e4m3 values in [0, 1.875]
    std::vector<uint8_t> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = (uint8_t)(rand() % 0x40);
    return h;
  };
  for (int l = 0; l < L; ++l) {
    uint8_t *p1, *p2;
    CK(cudaMalloc(&p1, (size_t)12288 * 2048)); CK(cudaMalloc(&p2, (size_t)2048 * 4096));
    auto h1 = rand_bytes((size_t)12288 * 2048), h2 = rand_bytes((size_t)2048 * 4096);
    CK(cudaMemcpy(p1, h1.data(), h1.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(p2, h2.data(), h2.size(), cudaMemcpyHostToDevice));
    B.w1.push_back(p1); B.w2.push_back(p2);
  }
  std::vector<float> ones(12288, 0.01f);
  CK(cudaMalloc(&B.sb1, 12288 * 4)); CK(cudaMalloc(&B.sb2, 2048 * 4));
  CK(cudaMemcpy(B.sb1, ones.data(), 12288 * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(B.sb2, ones.data(), 2048 * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&B.x0, M * 2048 * 2)); CK(cudaMalloc(&B.o1, (size_t)M * 12288 * 2)); CK(cudaMalloc(&B.o2, M * 2048 * 2));
  CK(cudaMalloc(&B.a1, M * 2048)); CK(cudaMalloc(&B.a2, M * 4096));
  CK(cudaMalloc(&B.sa1, 64 * 4)); CK(cudaMalloc(&B.sa2, 64 * 4)); CK(cudaMalloc(&B.scratch, 1024 * 4));
  {
    std::vector<__nv_bfloat16> hx(M * 2048);
    for (auto& v : hx) v = __float2bfloat16((rand() % 2001 - 1000) / 500.f);
    CK(cudaMemcpy(B.x0, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    std::vector<float> hs(1024, 0.5f); CK(cudaMemcpy(B.scratch, hs.data(), 4096, cudaMemcpyHostToDevice));
  }
  CK(cudaMemset(B.sa1, 0, 256)); CK(cudaMemset(B.sa2, 0, 256));

  Cfg cfgs[] = {
      {"plain", 0, false, false},
      {"gemm-only PDL, wait+trigger", 2, true, false},
      {"gemm-only PDL, +weight regs before wait", 3, true, false},
      {"gemm-only PDL, +L2 bulk prefetch before wait", 4, true, false},
      {"all PDL, wait+trigger", 2, true, true},
      {"all PDL, +weight regs before wait", 3, true, true},
      {"all PDL, +L2 bulk prefetch before wait", 4, true, true},
  };
  const int NC = sizeof(cfgs) / sizeof(cfgs[0]);
  auto skip = [&](int i) { return only >= 0 && i != only; };
  std::vector<cudaGraphExec_t> ge(NC);
  std::vector<int> nks(NC);
  std::vector<uint32_t> sums(NC);
  for (int i = 0; i < NC; ++i) {
    if (skip(i)) continue;
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    nks[i] = enqueue_chain(B, cfgs[i], st);
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge[i], g, 0));
    CK(cudaGraphDestroy(g));
    // checksum of final output after one run
    CK(cudaGraphLaunch(ge[i], st)); CK(cudaStreamSynchronize(st));
    std::vector<uint16_t> ho(M * 2048);
    CK(cudaMemcpy(ho.data(), B.o2, ho.size() * 2, cudaMemcpyDeviceToHost));
    uint32_t h = 2166136261u; for (auto v : ho) { h ^= v; h *= 16777619u; }
    sums[i] = h;
  }
  printf("# L=%d layers, M=%d, E=%d tiny kernels/layer, %d kernels per graph, weights %.0f MB, reps=%d x samples=%d x rounds=%d\n",
         L, M, E, nks[0], L * 33.6, reps, samples, rounds);
  for (int i = 0; i < NC; ++i) if (!skip(i)) printf("# output checksum %-48s %08x %s\n", cfgs[i].name, sums[i], sums[i] == sums[only >= 0 ? only : 0] ? "(== plain)" : "(DIFFERS)");

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  std::vector<std::vector<float>> times(NC);
  for (int r = 0; r < rounds; ++r)
    for (int i = 0; i < NC; ++i) {
      if (skip(i)) continue;
      for (int w = 0; w < 5; ++w) CK(cudaGraphLaunch(ge[i], st));
      for (int s = 0; s < samples; ++s) {
        CK(cudaEventRecord(e0, st));
        for (int k = 0; k < reps; ++k) CK(cudaGraphLaunch(ge[i], st));
        CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        times[i].push_back(ms * 1000.f / reps);
      }
    }
  printf("\n%-50s %10s %10s %10s %9s\n", "config (chain of real-shaped skinny kernels)", "median us", "min us", "us/kernel", "vs plain");
  float base = times[0].empty() ? 0.f : median(times[0]);
  for (int i = 0; i < NC; ++i) {
    if (skip(i)) continue;
    float md = median(times[i]), mn = *std::min_element(times[i].begin(), times[i].end());
    printf("%-50s %10.1f %10.1f %10.3f %+8.1f%%  (%+.1f us)\n", cfgs[i].name, md, mn, md / nks[i], 100.f * (md - base) / base, md - base);
  }

  if (only >= 0) return 0;
  // floor: chain of tiny dependent kernels, plain vs PDL
  {
    cudaGraphExec_t gp[2];
    for (int v = 0; v < 2; ++v) {
      cudaGraph_t g; CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
      for (int k = 0; k < floor_n; ++k) tiny_dispatch(v == 1, st, B.scratch, 1024);
      CK(cudaStreamEndCapture(st, &g)); CK(cudaGraphInstantiate(&gp[v], g, 0)); CK(cudaGraphDestroy(g));
    }
    std::vector<float> t[2];
    for (int r = 0; r < rounds; ++r)
      for (int v = 0; v < 2; ++v) {
        for (int w = 0; w < 5; ++w) CK(cudaGraphLaunch(gp[v], st));
        for (int s = 0; s < samples; ++s) {
          CK(cudaEventRecord(e0, st));
          for (int k = 0; k < reps; ++k) CK(cudaGraphLaunch(gp[v], st));
          CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
          float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
          t[v].push_back(ms * 1000.f / reps);
        }
      }
    printf("\nfloor: chain of %d dependent tiny elementwise kernels (4 CTAs each, ~no work)\n", floor_n);
    printf("  plain      %8.1f us  = %.3f us/kernel\n", median(t[0]), median(t[0]) / floor_n);
    printf("  all PDL    %8.1f us  = %.3f us/kernel\n", median(t[1]), median(t[1]) / floor_n);
  }
  return 0;
}
