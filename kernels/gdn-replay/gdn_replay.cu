// GDN speculative-decode state replay kernels (standalone CUDA, C ABI).
//
// Derived from vLLM v0.32.0 csrc/gdn/fused_gdn_decode_kernel.cu (Apache-2.0,
// Copyright contributors to the vLLM project). Restricted to fp32 state,
// head_k_dim = head_v_dim = 128, bf16 activations.
//
//  base_decode   verbatim copy of the production kernel (reference, 1R + T writes)
//  verify        same math and output, but writes NO recurrent state. It saves what a
//                replay needs: per token delta (fp32 [HV,128]), normalized k (fp32
//                [H,128]) and decay (fp32 [HV]).
//  commit        one launch for all layers. For each request it reads the committed state,
//                folds the first n accepted tokens (h = h*decay; h = fma(k, delta, h), the
//                exact two roundings of the production loop) and writes the one state the
//                production kernel would have left in window slot n-1 (plus, optionally,
//                the block-boundary state in window slot m-1).
//  deferred      verify with a fold prologue: reads the lagging state once, folds the
//                previous step's n tokens from the saved inputs, writes the folded base
//                state back (1R + 1W per step) and saves this step's inputs.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ void cp_async_16b(float* smem_ptr, const float* gmem_ptr) {
  const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(smem_addr), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }

template <int ChunkV, int DimK, int Stages>
__device__ __forceinline__ void copy_state_chunk(float* shared_state, const float* state, int chunk,
                                                 int thread, int threads) {
  constexpr int kElementsPerCopy = 4;
  constexpr int kCopiesPerChunk = ChunkV * DimK / kElementsPerCopy;
  const int stage = chunk % Stages;
  for (int copy = thread; copy < kCopiesPerChunk; copy += threads) {
    const int element = copy * kElementsPerCopy;
    cp_async_16b(shared_state + stage * ChunkV * DimK + element, state + chunk * ChunkV * DimK + element);
  }
  cp_async_commit();
}

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunkV = 32;
constexpr int kNumChunks = kDimV / kChunkV;
constexpr int kRowsPerWarp = kChunkV / kWarps;  // 4
constexpr int kMaxMtpTokens = 8;
constexpr int kFoldMax = 4;
constexpr int kDtBiasFloat32 = 0;
constexpr int kDtBiasBFloat16 = 1;
constexpr int kDtBiasFloat16 = 2;

struct GdnDecodeStrides {
  int64_t mixed_row;
  int64_t a_row;
  int64_t b_row;
  int64_t gate_row;
  int64_t state_slot;
};

__device__ __forceinline__ float sigmoid_fast(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float silu_fast(float x) { return x * sigmoid_fast(x); }
__device__ __forceinline__ float softplus_fast(float x) { return x > 20.0f ? x : log1pf(__expf(x)); }

__device__ __forceinline__ float load_dt_bias(const void* dt_bias, int head, int dt_bias_type) {
  if (dt_bias_type == kDtBiasBFloat16) return __bfloat162float(static_cast<const bf16*>(dt_bias)[head]);
  if (dt_bias_type == kDtBiasFloat16) return __half2float(static_cast<const __half*>(dt_bias)[head]);
  return static_cast<const float*>(dt_bias)[head];
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
  return value;
}
struct Sum2 { float x; float y; };
__device__ __forceinline__ Sum2 warp_reduce_sum_pair(float x, float y) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, offset);
    y += __shfl_xor_sync(0xffffffffu, y, offset);
  }
  return {x, y};
}

// ---------------------------------------------------------------------------------------------
// verify / base kernel. WriteStates=true, SaveReplay=false is the production kernel.
// Fold=true adds the deferred prologue (see file header).
// ---------------------------------------------------------------------------------------------
template <int VPK, bool SigmoidGate, bool WriteStates, bool SaveReplay, bool Fold>
__global__ __launch_bounds__(kThreads, 2) void gdn_verify_kernel(
    const bf16* __restrict__ mixed_qkv, const bf16* __restrict__ a, const bf16* __restrict__ b,
    const float* __restrict__ a_log, const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices, const int* __restrict__ cu_seqlens,
    const int* __restrict__ num_accepted_tokens, float* __restrict__ state,
    const bf16* __restrict__ output_gate, const void* __restrict__ norm_weight, bf16* __restrict__ out,
    float* rep_delta, float* rep_k, float* rep_decay, const int* __restrict__ fold_n,
    int* marker, const int* epoch, int k_heads,
    int H, int HV, int W, int dt_bias_type, bool norm_weight_is_bf16, float scale, float norm_eps,
    GdnDecodeStrides strides) {
  const int request = blockIdx.x;
  const int value_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bos = cu_seqlens[request];
  const int eos = cu_seqlens[request + 1];
  const int num_tokens = eos - bos;
  if (num_tokens <= 0) return;

  const int accepted = num_accepted_tokens[request];
  const int source_slot =
      accepted > 0 && accepted <= W ? state_indices[request * W + accepted - 1] : 0;
  if (source_slot <= 0 || num_tokens > kMaxMtpTokens) {
    for (int linear = tid; linear < num_tokens * kDimV; linear += kThreads) {
      const int token = bos + linear / kDimV;
      const int value = linear % kDimV;
      const int64_t out_offset = (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      out[out_offset] = __float2bfloat16(0.0f);
    }
    return;
  }

  const int key_head = value_head / VPK;
  if constexpr (SaveReplay) {
    if (marker != nullptr && value_head == 0 && tid == 0) marker[request] = *epoch;
  }
  __shared__ float shared_state[2][kChunkV][kDimK];
  __shared__ float shared_q[kMaxMtpTokens][kDimK];
  __shared__ float shared_k[kMaxMtpTokens][kDimK];
  __shared__ bf16 shared_v[kMaxMtpTokens][kDimV];
  __shared__ bf16 shared_out[kMaxMtpTokens][kDimV];
  __shared__ float shared_decay[kMaxMtpTokens];
  __shared__ float shared_beta[kMaxMtpTokens];
  __shared__ float prev_k[Fold ? kFoldMax : 1][Fold ? kDimK : 1];
  __shared__ float prev_decay[Fold ? kFoldMax : 1];

  float* source_state = state + static_cast<int64_t>(source_slot) * strides.state_slot + value_head * kDimV * kDimK;
  copy_state_chunk<kChunkV, kDimK, 2>(&shared_state[0][0][0], source_state, 0, tid, kThreads);

  int nfold = 0;
  if constexpr (Fold) {
    nfold = fold_n[request];
    nfold = nfold < 0 ? 0 : (nfold > kFoldMax ? kFoldMax : nfold);
    // previous step's k and decay (read before this step overwrites them)
    for (int idx = tid; idx < nfold * kDimK; idx += kThreads) {
      const int t = idx / kDimK, d = idx % kDimK;
      prev_k[t][d] = rep_k[((static_cast<int64_t>(request) * W + t) * k_heads + value_head) * kDimK + d];
    }
    if (tid < nfold) prev_decay[tid] = rep_decay[(static_cast<int64_t>(request) * W + tid) * HV + value_head];
  }
  if constexpr (Fold) __syncthreads();  // everyone has read the previous rep_k/rep_decay

  if (warp < num_tokens) {
    const int t = warp;
    const int token = bos + t;
    const int64_t mixed_base = static_cast<int64_t>(token) * strides.mixed_row;
    float q_values[4];
    float k_values[4];
    float q_square = 0.0f;
    float k_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      q_values[i] = __bfloat162float(mixed_qkv[mixed_base + key_head * kDimK + dim]);
      k_values[i] = __bfloat162float(mixed_qkv[mixed_base + H * kDimK + key_head * kDimK + dim]);
      shared_v[t][dim] = mixed_qkv[mixed_base + 2 * H * kDimK + value_head * kDimV + dim];
      q_square += q_values[i] * q_values[i];
      k_square += k_values[i] * k_values[i];
    }
    const Sum2 qk_sums = warp_reduce_sum_pair(q_square, k_square);
    const float q_scale = __shfl_sync(0xffffffffu, lane == 0 ? rsqrtf(qk_sums.x + 1.0e-6f) * scale : 0.0f, 0);
    const float k_scale = __shfl_sync(0xffffffffu, lane == 0 ? rsqrtf(qk_sums.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      shared_q[t][dim] = q_values[i] * q_scale;
      shared_k[t][dim] = k_values[i] * k_scale;
      if constexpr (SaveReplay) {
        if (k_heads == HV || (value_head % VPK) == 0)
          rep_k[((static_cast<int64_t>(request) * W + t) * k_heads + (k_heads == HV ? value_head : key_head)) * kDimK + dim] = shared_k[t][dim];
      }
    }
    if (lane == 0) {
      const float a_value = __bfloat162float(a[static_cast<int64_t>(token) * strides.a_row + value_head]);
      const float b_value = __bfloat162float(b[static_cast<int64_t>(token) * strides.b_row + value_head]);
      const float g = -__expf(a_log[value_head]) * softplus_fast(a_value + load_dt_bias(dt_bias, value_head, dt_bias_type));
      shared_decay[t] = __expf(g);
      shared_beta[t] = sigmoid_fast(b_value);
      if constexpr (SaveReplay)
        rep_decay[(static_cast<int64_t>(request) * W + t) * HV + value_head] = shared_decay[t];
    }
  }
  __syncthreads();

  const int k_base = lane * 4;
  int rows[kRowsPerWarp];
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) rows[row] = warp + row * kWarps;

#pragma unroll
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    cp_async_wait_all();
    __syncthreads();
    if (chunk + 1 < kNumChunks)
      copy_state_chunk<kChunkV, kDimK, 2>(&shared_state[0][0][0], source_state, chunk + 1, tid, kThreads);

    float h[kRowsPerWarp][4];
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
      const float4 sv = *reinterpret_cast<const float4*>(&shared_state[chunk & 1][rows[row]][k_base]);
      h[row][0] = sv.x; h[row][1] = sv.y; h[row][2] = sv.z; h[row][3] = sv.w;
    }

    if constexpr (Fold) {
      // fold the previous step's accepted tokens: h = h*decay; h = fma(k, delta, h)
      float pd[kFoldMax][kRowsPerWarp];
#pragma unroll
      for (int t = 0; t < kFoldMax; ++t) {
        if (t < nfold) {
#pragma unroll
          for (int row = 0; row < kRowsPerWarp; ++row)
            pd[t][row] = rep_delta[((static_cast<int64_t>(request) * W + t) * HV + value_head) * kDimV + chunk * kChunkV + rows[row]];
        }
      }
#pragma unroll
      for (int t = 0; t < kFoldMax; ++t) {
        if (t >= nfold) break;
        const float4 k4 = *reinterpret_cast<const float4*>(&prev_k[t][k_base]);
        const float kv[4] = {k4.x, k4.y, k4.z, k4.w};
#pragma unroll
        for (int row = 0; row < kRowsPerWarp; ++row) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            h[row][i] = __fmul_rn(h[row][i], prev_decay[t]);
            h[row][i] = __fmaf_rn(kv[i], pd[t][row], h[row][i]);
          }
        }
      }
      // write the folded base state back (in place)
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int value = chunk * kChunkV + rows[row];
        *reinterpret_cast<float4*>(source_state + value * kDimK + k_base) = make_float4(h[row][0], h[row][1], h[row][2], h[row][3]);
      }
    }

    for (int t = 0; t < num_tokens; ++t) {
      const float4 q4 = *reinterpret_cast<const float4*>(&shared_q[t][k_base]);
      const float4 k4 = *reinterpret_cast<const float4*>(&shared_k[t][k_base]);
      const float q_values[4] = {q4.x, q4.y, q4.z, q4.w};
      const float k_values[4] = {k4.x, k4.y, k4.z, k4.w};

      float dot_hk[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] *= shared_decay[t];
          dot_hk[row] += h[row][i] * k_values[i];
        }
      }
      const Sum2 dot_hk_01 = warp_reduce_sum_pair(dot_hk[0], dot_hk[1]);
      const Sum2 dot_hk_23 = warp_reduce_sum_pair(dot_hk[2], dot_hk[3]);
      const float reduced_hk[kRowsPerWarp] = {dot_hk_01.x, dot_hk_01.y, dot_hk_23.x, dot_hk_23.y};

      float dot_hq[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int value = chunk * kChunkV + rows[row];
        const float delta = (__bfloat162float(shared_v[t][value]) - reduced_hk[row]) * shared_beta[t];
        if constexpr (SaveReplay) {
          if (lane == 0)
            rep_delta[((static_cast<int64_t>(request) * W + t) * HV + value_head) * kDimV + value] = delta;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] += k_values[i] * delta;
          dot_hq[row] += h[row][i] * q_values[i];
        }
      }
      const Sum2 dot_hq_01 = warp_reduce_sum_pair(dot_hq[0], dot_hq[1]);
      const Sum2 dot_hq_23 = warp_reduce_sum_pair(dot_hq[2], dot_hq[3]);
      if (lane == 0) {
        shared_out[t][chunk * kChunkV + rows[0]] = __float2bfloat16(dot_hq_01.x);
        shared_out[t][chunk * kChunkV + rows[1]] = __float2bfloat16(dot_hq_01.y);
        shared_out[t][chunk * kChunkV + rows[2]] = __float2bfloat16(dot_hq_23.x);
        shared_out[t][chunk * kChunkV + rows[3]] = __float2bfloat16(dot_hq_23.y);
      }

      if constexpr (WriteStates) {
        const int destination_slot = state_indices[request * W + t];
        if (destination_slot > 0) {
          float* destination_state = state + static_cast<int64_t>(destination_slot) * strides.state_slot + value_head * kDimV * kDimK;
#pragma unroll
          for (int row = 0; row < kRowsPerWarp; ++row) {
            const int value = chunk * kChunkV + rows[row];
            *reinterpret_cast<float4*>(destination_state + value * kDimK + k_base) =
                make_float4(h[row][0], h[row][1], h[row][2], h[row][3]);
          }
        }
      }
    }
  }
  __syncthreads();

  if (warp < num_tokens) {
    const int t = warp;
    float output_values[4];
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      output_values[i] = __bfloat162float(shared_out[t][value]);
      sum_square += output_values[i] * output_values[i];
    }
    sum_square = warp_reduce_sum(sum_square);
    const float rstd = rsqrtf(sum_square / static_cast<float>(kDimV) + norm_eps);
    const int token = bos + t;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      const float gate_input = __bfloat162float(output_gate[static_cast<int64_t>(token) * strides.gate_row + value_head * kDimV + value]);
      const float gate = SigmoidGate ? sigmoid_fast(gate_input) : silu_fast(gate_input);
      const float weight = norm_weight_is_bf16 ? __bfloat162float(static_cast<const bf16*>(norm_weight)[value])
                                               : static_cast<const float*>(norm_weight)[value];
      const int64_t out_offset = (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      out[out_offset] = __float2bfloat16(output_values[i] * rstd * weight * gate);
    }
  }
}

// ---------------------------------------------------------------------------------------------
// commit kernel: grid (requests, HV, layers), 256 threads. Warp w owns rows w + 8*j (j = 0..15),
// lane l owns k = 4l..4l+3 (one float4, a fully coalesced 512 B row per warp).
// Optional inputs: marker/epoch (skip rows whose verify did not run in this step), bound_m or
// (idx_mapping, newc, mamba_block) to derive the block-boundary token count m.
// ---------------------------------------------------------------------------------------------
__global__ __launch_bounds__(kThreads) void gdn_commit_kernel(
    const int64_t* __restrict__ state_ptrs, const int64_t* __restrict__ delta_ptrs,
    const int64_t* __restrict__ k_ptrs, const int64_t* __restrict__ decay_ptrs,
    const int64_t* __restrict__ marker_ptrs, const int64_t* __restrict__ state_strides,
    const int* __restrict__ epoch,
    const int* __restrict__ state_indices, const int* __restrict__ cu_seqlens,
    const int* __restrict__ acc_prev, const int* __restrict__ n_new, const int* __restrict__ bound_m,
    const int* __restrict__ idx_mapping, const int* __restrict__ newc, const int64_t* __restrict__ dst_ptrs,
    int mamba_block, int k_heads, int HV, int VPK, int W) {
  const int request = blockIdx.x;
  const int value_head = blockIdx.y;
  const int layer = blockIdx.z;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int num_tokens = cu_seqlens[request + 1] - cu_seqlens[request];
  if (num_tokens <= 0 || num_tokens > kMaxMtpTokens) return;
  if (marker_ptrs != nullptr) {
    const int* marker = reinterpret_cast<const int*>(marker_ptrs[layer]);
    if (marker[request] != *epoch) return;
  }
  const int accepted = acc_prev[request];
  const int source_slot = accepted > 0 && accepted <= W ? state_indices[request * W + accepted - 1] : 0;
  int n = n_new[request];
  n = n > num_tokens ? num_tokens : n;
  if (source_slot <= 0 || n <= 0) return;
  const int final_slot = state_indices[request * W + n - 1];
  int m = bound_m ? bound_m[request] : 0;
  if (newc != nullptr && mamba_block > 0) {
    const int slot = idx_mapping ? idx_mapping[request] : request;
    const int nc = newc[slot];
    const int c = nc - n;
    const int b = (nc / mamba_block) * mamba_block;
    m = (b >= c + 1) ? (b - c) : 0;
  }
  m = (m > 0 && m < n) ? m : 0;
  const int bound_slot = m > 0 ? state_indices[request * W + m - 1] : 0;
  if (final_slot <= 0 && bound_slot <= 0) return;

  const int64_t stride = state_strides[layer];
  float* state = reinterpret_cast<float*>(state_ptrs[layer]);
  const float* delta = reinterpret_cast<const float*>(delta_ptrs[layer]);
  const float* kk = reinterpret_cast<const float*>(k_ptrs[layer]);
  const float* decay = reinterpret_cast<const float*>(decay_ptrs[layer]);
  const int key_head = value_head / VPK;
  const int kh = k_heads == HV ? value_head : key_head;

  const int64_t head_off = static_cast<int64_t>(value_head) * kDimV * kDimK;
  const float* src = state + static_cast<int64_t>(source_slot) * stride + head_off;
  float4 h[16];
#pragma unroll
  for (int j = 0; j < 16; ++j)
    h[j] = *reinterpret_cast<const float4*>(src + (warp + 8 * j) * kDimK + lane * 4);

  float* dstate = dst_ptrs ? reinterpret_cast<float*>(dst_ptrs[layer]) : state;  // check mode writes elsewhere
  float* dst_final = final_slot > 0 ? dstate + static_cast<int64_t>(final_slot) * stride + head_off : nullptr;
  float* dst_bound = bound_slot > 0 ? dstate + static_cast<int64_t>(bound_slot) * stride + head_off : nullptr;

  for (int t = 0; t < n; ++t) {
    if (t == m && dst_bound) {
#pragma unroll
      for (int j = 0; j < 16; ++j) *reinterpret_cast<float4*>(dst_bound + (warp + 8 * j) * kDimK + lane * 4) = h[j];
    }
    const int64_t tok = static_cast<int64_t>(request) * W + t;
    const float dec = decay[tok * HV + value_head];
    const float4 k4 = *reinterpret_cast<const float4*>(kk + (tok * k_heads + kh) * kDimK + lane * 4);
    const float* dl = delta + (tok * HV + value_head) * kDimV;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float d = dl[warp + 8 * j];
      float4 x = h[j];
      x.x = __fmul_rn(x.x, dec); x.y = __fmul_rn(x.y, dec); x.z = __fmul_rn(x.z, dec); x.w = __fmul_rn(x.w, dec);
      x.x = __fmaf_rn(k4.x, d, x.x); x.y = __fmaf_rn(k4.y, d, x.y);
      x.z = __fmaf_rn(k4.z, d, x.z); x.w = __fmaf_rn(k4.w, d, x.w);
      h[j] = x;
    }
  }
  if (dst_final) {
#pragma unroll
    for (int j = 0; j < 16; ++j) *reinterpret_cast<float4*>(dst_final + (warp + 8 * j) * kDimK + lane * 4) = h[j];
  }
}

// ------------------------------------------------------------------------------------------------
// host side: plain C ABI (loaded with ctypes), no torch headers needed.
// ------------------------------------------------------------------------------------------------
}  // namespace

struct VerifyArgs {
  const void *mixed_qkv, *a, *b, *a_log, *dt_bias, *state_indices, *cu_seqlens, *num_accepted;
  void* state;
  const void *output_gate, *norm_weight;
  void* out;
  void *rep_delta, *rep_k, *rep_decay;
  const void* fold_n;
  void* marker;
  const void* epoch;
  int N, W, H, HV, k_heads;
  int write_states;   // check mode: stock kernel (writes states) that also saves the replay inputs
  int dt_type;        // 0 fp32, 1 bf16, 2 fp16
  int norm_w_bf16;
  int sigmoid_gate;
  float scale, eps;
  int64_t mixed_row, a_row, b_row, gate_row, state_slot;
};

template <bool WriteStates, bool SaveReplay, bool Fold>
int launch_verify(const VerifyArgs& p, cudaStream_t stream) {
  const int vpk = p.HV / p.H;
  dim3 grid(p.N, p.HV);
  GdnDecodeStrides st{p.mixed_row, p.a_row, p.b_row, p.gate_row, p.state_slot};
#define LAUNCH(VPK_, SIG_)                                                                                      \
  gdn_verify_kernel<VPK_, SIG_, WriteStates, SaveReplay, Fold><<<grid, kThreads, 0, stream>>>(                   \
      static_cast<const bf16*>(p.mixed_qkv), static_cast<const bf16*>(p.a), static_cast<const bf16*>(p.b),      \
      static_cast<const float*>(p.a_log), p.dt_bias, static_cast<const int*>(p.state_indices),                  \
      static_cast<const int*>(p.cu_seqlens), static_cast<const int*>(p.num_accepted),                           \
      static_cast<float*>(p.state), static_cast<const bf16*>(p.output_gate), p.norm_weight,                     \
      static_cast<bf16*>(p.out), static_cast<float*>(p.rep_delta), static_cast<float*>(p.rep_k),                \
      static_cast<float*>(p.rep_decay), static_cast<const int*>(p.fold_n), static_cast<int*>(p.marker),       \
      static_cast<const int*>(p.epoch), p.k_heads, p.H, p.HV, p.W, p.dt_type,                                 \
      p.norm_w_bf16 != 0, p.scale, p.eps, st)
  if (vpk == 2) { if (p.sigmoid_gate) LAUNCH(2, true); else LAUNCH(2, false); }
  else if (vpk == 1) { if (p.sigmoid_gate) LAUNCH(1, true); else LAUNCH(1, false); }
  else if (vpk == 4) { if (p.sigmoid_gate) LAUNCH(4, true); else LAUNCH(4, false); }
  else return -1;
#undef LAUNCH
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

extern "C" {
int gdn_base_decode(const VerifyArgs* p, void* stream) { return launch_verify<true, false, false>(*p, (cudaStream_t)stream); }
int gdn_verify(const VerifyArgs* p, void* stream) {
  return p->write_states ? launch_verify<true, true, false>(*p, (cudaStream_t)stream)
                         : launch_verify<false, true, false>(*p, (cudaStream_t)stream);
}
int gdn_deferred(const VerifyArgs* p, void* stream) { return launch_verify<false, true, true>(*p, (cudaStream_t)stream); }

// *_ptrs: int64 device arrays [L] holding each layer's data pointer.
int gdn_commit(const void* state_ptrs, const void* delta_ptrs, const void* k_ptrs, const void* decay_ptrs,
               const void* marker_ptrs, const void* state_strides, const void* epoch,
               const void* state_indices, const void* cu_seqlens, const void* acc_prev, const void* n_new,
               const void* bound_m, const void* idx_mapping, const void* newc, const void* dst_ptrs, int mamba_block,
               int N, int W, int L, int k_heads, int HV, int H, void* stream) {
  dim3 grid(N, HV, L);
  gdn_commit_kernel<<<grid, kThreads, 0, (cudaStream_t)stream>>>(
      static_cast<const int64_t*>(state_ptrs), static_cast<const int64_t*>(delta_ptrs),
      static_cast<const int64_t*>(k_ptrs), static_cast<const int64_t*>(decay_ptrs),
      static_cast<const int64_t*>(marker_ptrs), static_cast<const int64_t*>(state_strides),
      static_cast<const int*>(epoch), static_cast<const int*>(state_indices),
      static_cast<const int*>(cu_seqlens), static_cast<const int*>(acc_prev), static_cast<const int*>(n_new),
      static_cast<const int*>(bound_m), static_cast<const int*>(idx_mapping), static_cast<const int*>(newc),
      static_cast<const int64_t*>(dst_ptrs), mamba_block, k_heads, HV, HV / H, W);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}
}
