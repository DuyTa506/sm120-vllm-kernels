// SM120 FP8 (e4m3) TN GEMM with vLLM's ScaledEpilogue math:
//   D[m,n] = bf16_rn( scale_a[m] * ( scale_b[n] * acc[m,n] ) )
// Same CollectiveBuilder path as vLLM's cutlass_3x_gemm_sm120 (csrc/quantization/w8a8/cutlass/c3x),
// but with the tile shape / schedule / stage count as template parameters, so other configs can be
// timed against vLLM's built-in ones and checked for bitwise equality.
#pragma once
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

template <class Kernel>
struct enable_sm120_family_ : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};

template <class TileShape, class KernelSchedule, int Stages /*0 = auto carveout*/,
          class Scheduler = void, class EpiTile = cutlass::epilogue::collective::EpilogueTileAuto>
struct Sm120Fp8Gemm {
  using ElementAB = cutlass::float_e4m3_t;
  using ElementD = cutlass::bfloat16_t;
  using ElementAcc = float;
  using ClusterShape = Shape<_1, _1, _1>;

  using ScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, float, float,
                                                             Stride<_1, _0, int64_t>, 4>;
  using ScaleB = cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, float, float,
                                                             Stride<_0, _1, int64_t>, 4>;
  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, float, float,
                                                          cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, cutlass::epilogue::fusion::Sm90AccFetch>;
  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementD, float,
                                                          cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT = cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVT0>;

  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape, EpiTile,
      ElementAcc, float, void, LayoutD, AlignD, ElementD, LayoutD, AlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto, EVT>::CollectiveOp;

  using StageArg = std::conditional_t<
      Stages == 0,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::StageCount<Stages>>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementAB, cutlass::layout::RowMajor, 16,
      ElementAB, cutlass::layout::ColumnMajor, 16, ElementAcc, TileShape, ClusterShape, StageArg,
      KernelSchedule>::CollectiveOp;

  using GemmKernel = enable_sm120_family_<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, Scheduler>>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // out [M,N] bf16 row-major, a [M,K] e4m3 row-major, b [K,N] e4m3 column-major (= weight [N,K]
  // row-major, transposed view), sa [M] fp32, sb [N] fp32.
  static void run(torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b,
                  torch::Tensor const& sa, torch::Tensor const& sb, int raster, int swizzle,
                  int splits = 1, int decomp = 0) {
    int M = a.size(0), K = a.size(1), N = b.size(1);
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    typename Gemm::Arguments args;
    args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    args.problem_shape = {M, N, K, 1};
    args.mainloop = {reinterpret_cast<ElementAB const*>(a.data_ptr()), sA,
                     reinterpret_cast<ElementAB const*>(b.data_ptr()), sB};
    args.epilogue.ptr_C = nullptr;
    args.epilogue.dC = sD;
    args.epilogue.ptr_D = reinterpret_cast<ElementD*>(out.data_ptr());
    args.epilogue.dD = sD;
    // EVT args: {ScaleA, {ScaleB, Acc, Compute0}, Compute1}
    args.epilogue.thread = {
        {sa.data_ptr<float>(), 0.f, {_1{}, _0{}, int64_t(0)}},
        {{sb.data_ptr<float>(), 0.f, {_0{}, _1{}, int64_t(0)}}, {}, {}},
        {}};
    // swizzle < 0: leave hw_info.sm_count = 0 like vLLM's caller (CLC scheduler then launches the full
    // tile grid); swizzle >= 0: persistent grid of sm_count CTAs.
    args.hw_info.device_id = a.get_device();
    if (swizzle >= 0)
      args.hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    swizzle = swizzle < 0 ? -swizzle : swizzle;
    if constexpr (requires { args.scheduler.raster_order; }) {
      using RO = decltype(args.scheduler.raster_order);
      args.scheduler.raster_order = raster == 1 ? RO::AlongM : (raster == 2 ? RO::AlongN : RO::Heuristic);
      args.scheduler.max_swizzle_size = swizzle;
    }
    if constexpr (requires { args.scheduler.splits; }) {
      // decomp: 0 heuristic, 1 data-parallel, 2 split-K, 3 stream-K; deterministic reduction
      using DM = decltype(args.scheduler.decomposition_mode);
      args.scheduler.splits = splits;
      args.scheduler.decomposition_mode =
          decomp == 1 ? DM::DataParallel : (decomp == 2 ? DM::SplitK : (decomp == 3 ? DM::StreamK : DM::Heuristic));
      using RM = decltype(args.scheduler.reduction_mode);
      args.scheduler.reduction_mode = RM::Deterministic;
    }

    Gemm gemm;
    TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "cannot implement");
    size_t ws = Gemm::get_workspace_size(args);
    auto workspace = torch::empty({(int64_t)ws}, a.options().dtype(torch::kUInt8));
    auto stream = c10::cuda::getCurrentCUDAStream(a.get_device());
    TORCH_CHECK(gemm.initialize(args, workspace.data_ptr(), stream) == cutlass::Status::kSuccess, "init");
    TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "run");
  }
};
