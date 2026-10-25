// Register-resident tensor-core issue-rate probe for SM120 (mma.sync). Reports FLOP/clk/SM from clock64 (independent of power-cap clocks).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(err_), __LINE__); exit(1); } } while (0)

enum Mode { F16_F32, F16_F16, BF16_F32, TF32_F32, E4M3_F32, E5M2_F32, S8_S32, FP4_BLOCK };

template <int M> __device__ __forceinline__ void mma_op(uint32_t *a, uint32_t *b, uint32_t *c, int nc, uint32_t sf) {
  if constexpr (M == F16_F32) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else if constexpr (M == F16_F16) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else if constexpr (M == BF16_F32) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else if constexpr (M == TF32_F32) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else if constexpr (M == E4M3_F32) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else if constexpr (M == E5M2_F32) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else if constexpr (M == S8_S32) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else if constexpr (M == FP4_BLOCK) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_SPECIFIC__) || 1
    uint16_t bid = 0, tid = 0;
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, %13}, %11, {%12, %13};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sf), "r"(sf), "h"(bid), "h"(tid));
#endif
  }
}

template <int M, int NC>
__global__ void probe(long long iters, unsigned long long *cycles, uint32_t *sink) {
  uint32_t a[4], b[2], c[8][4];
  uint32_t seed = threadIdx.x * 2654435761u + blockIdx.x;
  for (int i = 0; i < 4; i++) a[i] = 0x3c003c00u ^ (seed >> (i & 3));
  for (int i = 0; i < 2; i++) b[i] = 0x3c003c00u;
  for (int j = 0; j < 8; j++) for (int i = 0; i < 4; i++) c[j][i] = 0;
  uint32_t sf = 0x38383838u;
  __syncthreads();
  long long t0 = clock64();
  for (long long it = 0; it < iters; it++) {
#pragma unroll
    for (int j = 0; j < 8; j++) mma_op<M>(a, b, c[j], NC, sf);
  }
  long long t1 = clock64();
  __syncthreads();
  uint32_t acc = 0;
  for (int j = 0; j < 8; j++) for (int i = 0; i < 4; i++) acc ^= c[j][i];
  if (threadIdx.x % 32 == 0) {
    atomicMax(&cycles[blockIdx.x], (unsigned long long)(t1 - t0));
  }
  if (acc == 0x12345678u) sink[0] = acc;  // keep results alive
}

struct Result { double flop_clk_sm, tflops, clk_ghz; };

template <int M, int NC>
Result run(const char *name, double flops_per_mma, long long iters, int warps_per_block) {
  int dev = 0; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
  int sms = p.multiProcessorCount;
  unsigned long long *cyc; uint32_t *sink;
  CK(cudaMalloc(&cyc, sizeof(unsigned long long) * sms)); CK(cudaMalloc(&sink, 4));
  CK(cudaMemset(cyc, 0, sizeof(unsigned long long) * sms));
  probe<M, NC><<<sms, warps_per_block * 32>>>(1000, cyc, sink);  // warm up
  CK(cudaDeviceSynchronize()); CK(cudaMemset(cyc, 0, sizeof(unsigned long long) * sms));
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  probe<M, NC><<<sms, warps_per_block * 32>>>(iters, cyc, sink);
  cudaEventRecord(e); CK(cudaEventSynchronize(e)); CK(cudaGetLastError());
  float ms; cudaEventElapsedTime(&ms, s, e);
  unsigned long long *h = (unsigned long long *)malloc(sizeof(unsigned long long) * sms);
  CK(cudaMemcpy(h, cyc, sizeof(unsigned long long) * sms, cudaMemcpyDeviceToHost));
  double cmax = 0, csum = 0; for (int i = 0; i < sms; i++) { if (h[i] > cmax) cmax = h[i]; csum += h[i]; }
  double total_flop_per_sm = (double)warps_per_block * iters * 8 * flops_per_mma;
  Result r; r.flop_clk_sm = total_flop_per_sm / (csum / sms);
  r.clk_ghz = (csum / sms) / (ms * 1e-3) / 1e9;
  r.tflops = total_flop_per_sm * sms / (ms * 1e-3) / 1e12;
  printf("%-22s %8.1f FLOP/clk/SM   avg SM clock %.2f GHz   %7.1f TFLOPS (register-resident)   %.0f ms\n", name, r.flop_clk_sm, r.clk_ghz, r.tflops, ms);
  fflush(stdout);
  cudaFree(cyc); cudaFree(sink); free(h);
  return r;
}

int main(int argc, char **argv) {
  long long iters = argc > 1 ? atoll(argv[1]) : 40000;
  int warps = argc > 2 ? atoi(argv[2]) : 8;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("%s  SMs=%d  cc=%d.%d  iters=%lld warps/block=%d\n", p.name, p.multiProcessorCount, p.major, p.minor, iters, warps);
  run<F16_F32, 4>("FP16 -> FP32 acc", 2.0 * 16 * 8 * 16, iters, warps);
  run<F16_F16, 2>("FP16 -> FP16 acc", 2.0 * 16 * 8 * 16, iters, warps);
  run<BF16_F32, 4>("BF16 -> FP32 acc", 2.0 * 16 * 8 * 16, iters, warps);
  run<TF32_F32, 4>("TF32 -> FP32 acc", 2.0 * 16 * 8 * 8, iters, warps);
  run<E4M3_F32, 4>("FP8 e4m3 -> FP32 acc", 2.0 * 16 * 8 * 32, iters, warps);
  run<E5M2_F32, 4>("FP8 e5m2 -> FP32 acc", 2.0 * 16 * 8 * 32, iters, warps);
  run<S8_S32, 4>("INT8 -> INT32", 2.0 * 16 * 8 * 32, iters, warps);
#ifdef WITH_FP4
  run<FP4_BLOCK, 4>("NVFP4 block-scaled", 2.0 * 16 * 8 * 64, iters, warps);
#endif
  return 0;
}
