// cuBLASLt GEMM sweep: FP16 / BF16 / TF32 / FP8 / NVFP4 (all fp32 accumulate)
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)
#define CB(x) do { cublasStatus_t st_ = (x); if (st_ != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS %d @%d\n", (int)st_, __LINE__); exit(1);} } while (0)

__global__ void fill_rand(unsigned char* p, size_t n, unsigned seed) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    unsigned x = (unsigned)i * 2654435761u + seed;
    x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
    p[i] = (unsigned char)x;
  }
}
// fp16/bf16/fp8: random bytes would give NaN/Inf, so build safe values per type
__global__ void fill_half(__half_raw* p, size_t n, unsigned seed, bool bf16) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    unsigned x = (unsigned)i * 2654435761u + seed;
    x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
    unsigned mant = x & 0x3ff;           // random mantissa
    unsigned sign = (x >> 10) & 1;
    unsigned short v;
    if (bf16) v = (sign << 15) | (0x3f00 | (x & 0x7f));        // ~0.5..1
    else      v = (sign << 15) | (0x3800 | (mant & 0x3ff));    // ~0.5..1
    p[i].x = v;
  }
}
__global__ void fill_fp8(unsigned char* p, size_t n, unsigned seed) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    unsigned x = (unsigned)i * 2654435761u + seed;
    x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
    p[i] = (unsigned char)(((x & 1) << 7) | (0x30 | (x & 0x0f)));  // e4m3 ~0.5..2
  }
}
__global__ void fill_f32(float* p, size_t n, unsigned seed) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    unsigned x = (unsigned)i * 2654435761u + seed;
    x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
    p[i] = ((x & 0xffff) / 65535.f) - 0.5f;
  }
}

enum Kind { K_FP16, K_BF16, K_TF32, K_FP8, K_FP4 };

static size_t round_up(size_t a, size_t b) { return (a + b - 1) / b * b; }

void run(Kind kind, const char* name, int M, int N, int K, double sustain_sec) {
  cublasLtHandle_t lt; CB(cublasLtCreate(&lt));
  cudaDataType ta, tc; cublasComputeType_t comp = CUBLAS_COMPUTE_32F;
  double bytes_per_elem;
  switch (kind) {
    case K_FP16: ta = CUDA_R_16F; tc = CUDA_R_16F; bytes_per_elem = 2; break;
    case K_BF16: ta = CUDA_R_16BF; tc = CUDA_R_16BF; bytes_per_elem = 2; break;
    case K_TF32: ta = CUDA_R_32F; tc = CUDA_R_32F; comp = CUBLAS_COMPUTE_32F_FAST_TF32; bytes_per_elem = 4; break;
    case K_FP8:  ta = CUDA_R_8F_E4M3; tc = CUDA_R_16BF; bytes_per_elem = 1; break;
    case K_FP4:  ta = CUDA_R_4F_E2M1; tc = CUDA_R_16BF; bytes_per_elem = 0.5; break;
  }
  size_t szA = (size_t)M * K * bytes_per_elem, szB = (size_t)N * K * bytes_per_elem;
  size_t elA = (size_t)M * K, elB = (size_t)N * K;
  size_t szC = (size_t)M * N * (tc == CUDA_R_32F ? 4 : 2);
  int rot = (int)((600ull << 20) / (szA ? szA : 1)); if (rot < 1) rot = 1; if (rot > 64) rot = 64;  // defeat L2 reuse of weights
  void *A, *B, *C, *D, *ws; size_t wsz = 512ull << 20;
  CK(cudaMalloc(&A, szA * rot)); CK(cudaMalloc(&B, szB)); CK(cudaMalloc(&C, szC)); CK(cudaMalloc(&D, szC)); CK(cudaMalloc(&ws, wsz));
  CK(cudaMemset(C, 0, szC));
  if (kind == K_FP16 || kind == K_BF16) {
    fill_half<<<1024, 256>>>((__half_raw*)A, elA * rot, 1, kind == K_BF16);
    fill_half<<<1024, 256>>>((__half_raw*)B, elB, 2, kind == K_BF16);
  } else if (kind == K_TF32) {
    fill_f32<<<1024, 256>>>((float*)A, elA * rot, 1); fill_f32<<<1024, 256>>>((float*)B, elB, 2);
  } else if (kind == K_FP8) {
    fill_fp8<<<1024, 256>>>((unsigned char*)A, szA * rot, 1); fill_fp8<<<1024, 256>>>((unsigned char*)B, szB, 2);
  } else {
    fill_rand<<<1024, 256>>>((unsigned char*)A, szA * rot, 1); fill_rand<<<1024, 256>>>((unsigned char*)B, szB, 2);
  }
  // block scales for NVFP4 (UE4M3, 16-wide). 0x38 = 1.0 in e4m3.
  void *sA = nullptr, *sB = nullptr; float one = 1.f; float *dOne = nullptr;
  if (kind == K_FP4) {
    size_t scA = round_up(M, 128) * round_up((K + 15) / 16, 4), scB = round_up(N, 128) * round_up((K + 15) / 16, 4);
    CK(cudaMalloc(&sA, scA)); CK(cudaMalloc(&sB, scB));
    CK(cudaMemset(sA, 0x38, scA)); CK(cudaMemset(sB, 0x38, scB));
  }
  if (kind == K_FP8) { CK(cudaMalloc(&dOne, 4)); CK(cudaMemcpy(dOne, &one, 4, cudaMemcpyHostToDevice)); }
  CK(cudaDeviceSynchronize());

  cublasLtMatmulDesc_t op; CB(cublasLtMatmulDescCreate(&op, comp, CUDA_R_32F));
  cublasOperation_t T = CUBLAS_OP_T, Nn = CUBLAS_OP_N;   // TN layout, required for fp8/fp4
  CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof(T)));
  CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &Nn, sizeof(Nn)));
  if (kind == K_FP8) {
    CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dOne, sizeof(dOne)));
    CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dOne, sizeof(dOne)));
  }
  if (kind == K_FP4) {
    cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)));
    CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)));
    CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sA, sizeof(sA)));
    CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sB, sizeof(sB)));
  }
  cublasLtMatrixLayout_t lA, lB, lC;
  CB(cublasLtMatrixLayoutCreate(&lA, ta, K, M, K));   // A^T : K x M storage, ld=K
  CB(cublasLtMatrixLayoutCreate(&lB, ta, K, N, K));
  CB(cublasLtMatrixLayoutCreate(&lC, tc, M, N, M));
  cublasLtMatmulPreference_t pref; CB(cublasLtMatmulPreferenceCreate(&pref));
  CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));
  const int NA = 24; cublasLtMatmulHeuristicResult_t res[NA]; int nres = 0;
  cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, op, lA, lB, lC, lC, pref, NA, res, &nres);
  if (st != CUBLAS_STATUS_SUCCESS || nres == 0) { printf("%-6s %5d: no algo (status %d)\n", name, M, (int)st); return; }

  float alpha = 1.f, beta = 0.f;
  cudaStream_t s; CK(cudaStreamCreate(&s));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  double flop = 2.0 * M * N * K;
  auto timeit = [&](int a, int iters) {
    CK(cudaEventRecord(e0, s));
    for (int i = 0; i < iters; i++)
      CB(cublasLtMatmul(lt, op, &alpha, (char*)A + (size_t)(i % rot) * szA, lA, B, lB, &beta, C, lC, D, lC, &res[a].algo, ws, wsz, s));
    CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return ms / iters;
  };
  double bestS = 0, bestMs = 0; int bestA = -1; double heurS = 0;
  for (int a = 0; a < nres; a++) {
    if (cublasLtMatmul(lt, op, &alpha, A, lA, B, lB, &beta, C, lC, D, lC, &res[a].algo, ws, wsz, s) != CUBLAS_STATUS_SUCCESS) continue;
    timeit(a, 3);
    double t = timeit(a, 10);
    int iters = (int)(sustain_sec * 1000.0 / t); if (iters < 20) iters = 20;
    double ts = timeit(a, iters);                       // sustained: lets the power cap settle
    double tf = flop / (ts * 1e-3) / 1e12;
    if (a == 0) heurS = tf;
    if (tf > bestS) { bestS = tf; bestA = a; bestMs = ts; }
    if (getenv("VERBOSE")) printf("      %-6s algo#%d  burst %7.1f  sustained %7.1f TFLOPS\n", name, a, flop / (t * 1e-3) / 1e12, tf);
  }
  printf("   -> %-6s M=%d N=%d K=%d  heur#0 %7.1f  best %7.1f TFLOPS (algo#%d)  %8.1f us  weights %7.0f GB/s\n",
         name, M, N, K, heurS, bestS, bestA, bestMs * 1e3, (double)M * K * bytes_per_elem / (bestMs * 1e-3) / 1e9);
  fflush(stdout);
  cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(D); cudaFree(ws);
  if (sA) { cudaFree(sA); cudaFree(sB); } if (dOne) cudaFree(dOne);
  cublasLtDestroy(lt);
}

int main(int argc, char** argv) {
  int n = argc > 1 ? atoi(argv[1]) : 8192;
  double sustain = argc > 2 ? atof(argv[2]) : 3.0;
  std::string only = argc > 3 ? argv[3] : "";
  struct { Kind k; const char* n; } list[] = {{K_TF32, "TF32"}, {K_FP16, "FP16"}, {K_BF16, "BF16"}, {K_FP8, "FP8"}, {K_FP4, "NVFP4"}};
  int M = n, N = n, K = n;
  if (argc > 6) { M = atoi(argv[4]); N = atoi(argv[5]); K = atoi(argv[6]); }
  for (auto& e : list) if (only.empty() || only == e.n) run(e.k, e.n, M, N, K, sustain);
  return 0;
}
