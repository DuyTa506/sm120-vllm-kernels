// Can SM120 launch thread block clusters and use distributed shared memory? And what does a cluster barrier cost?
// (Only a feasibility probe for a cluster-based variant of the fused router; not used by the kernel.)
#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("  %s -> %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

template <int CS>
__global__ void __cluster_dims__(CS, 1, 1) cluster_exchange(float* out) {
  __shared__ float buf[64];
  cg::cluster_group cluster = cg::this_cluster();
  const unsigned rank = cluster.block_rank();
  buf[threadIdx.x % 64] = rank * 100.f + threadIdx.x % 64;
  cluster.sync();
  // every CTA reads the value of the next rank through distributed shared memory
  float* remote = cluster.map_shared_rank(buf, (rank + 1) % CS);
  float v = remote[threadIdx.x % 64];
  cluster.sync();   // nobody exits while a neighbour still reads its shared memory
  if (threadIdx.x == 0) out[blockIdx.x] = v;
}
__global__ void empty_kernel() {}

template <int CS>
int run(const char* name) {
  float* d; CK(cudaMalloc(&d, 4096 * sizeof(float)));
  cudaError_t e = cudaFuncSetAttribute(cluster_exchange<CS>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  printf("%s: NonPortableClusterSizeAllowed -> %s\n", name, cudaGetErrorString(e));
  cudaGetLastError();
  cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(CS * 4); cfg.blockDim = dim3(128);
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim = {CS, 1, 1};
  cfg.attrs = at; cfg.numAttrs = 1;
  int ncl = 0;
  e = cudaOccupancyMaxActiveClusters(&ncl, (void*)cluster_exchange<CS>, &cfg);
  printf("%s: max active clusters -> %s %d\n", name, cudaGetErrorString(e), ncl);
  cudaGetLastError();
  cluster_exchange<CS><<<CS * 4, 128>>>(d);
  e = cudaDeviceSynchronize();
  printf("%s: launch -> %s\n", name, cudaGetErrorString(e));
  if (e != cudaSuccess) return 0;
  float h[4096]; CK(cudaMemcpy(h, d, 4096 * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int b = 0; b < CS * 4; ++b) ok &= (h[b] == (float)(((b % CS) + 1) % CS * 100));
  printf("%s: DSMEM neighbour read %s\n", name, ok ? "correct" : "WRONG");
  // time a captured chain of 40 launches (per kernel us), against an empty-kernel chain
  cudaStream_t s; CK(cudaStreamCreate(&s));
  for (int variant = 0; variant < 2; ++variant) {
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < 40; ++i) {
      if (variant == 0) empty_kernel<<<CS * 4, 128, 0, s>>>();
      else cluster_exchange<CS><<<CS * 4, 128, 0, s>>>(d);
    }
    CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    float best = 1e9;
    for (int r = 0; r < 30; ++r) {
      cudaEventRecord(a, s); cudaGraphLaunch(ge, s); cudaEventRecord(b, s); cudaEventSynchronize(b);
      float ms; cudaEventElapsedTime(&ms, a, b); if (ms < best) best = ms;
    }
    printf("%s: %s chain: %.2f us per kernel (min of 30 replays)\n", name, variant ? "cluster-exchange (2 cluster barriers + DSMEM read)" : "empty-kernel", best * 1e3 / 40);
  }
  return 0;
}

int main() {
  int dev = 0; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
  int cl = -1; cudaDeviceGetAttribute(&cl, cudaDevAttrClusterLaunch, dev);
  printf("%s sm_%d%d, SMs %d, clusterLaunch attr %d, smem/block optin %zu KB, L2 %d MB\n", p.name, p.major, p.minor, p.multiProcessorCount, cl,
         p.sharedMemPerBlockOptin / 1024, p.l2CacheSize >> 20);
  run<2>("cluster 2"); run<4>("cluster 4"); run<8>("cluster 8"); run<16>("cluster 16");
  return 0;
}
