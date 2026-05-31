
// Minimal test: GEMM + XeColReduction(atomic_add) on AccFetch only
// Tests whether XeColReduction compiles at all on BMG/IGC

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include "sycl_common.hpp"
#include "helper.h"

using namespace cute;

using ElementAccumulator = float;
using ElementCompute = float;
using ElementInputA = bfloat16_t;
using ElementInputB = bfloat16_t;
using ElementOutput = bfloat16_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;

using TileShape = Shape<_256, _256, _32>;

using StrideC = cute::Stride<int64_t, cute::Int<1>, int64_t>;
using StrideD = cute::Stride<int64_t, cute::Int<1>, int64_t>;

// Simplest possible: AccFetch → ColReduction → Identity
using Accum = cutlass::epilogue::fusion::XeAccFetch;

using ColReduction = cutlass::epilogue::fusion::XeColReduction<
    cutlass::plus,
    cutlass::plus,
    cutlass::atomic_add,
    0,
    TileShape,
    ElementCompute,
    ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<cute::Int<1>, cute::Int<0>, int64_t>,
    128 / cutlass::sizeof_bits_v<ElementCompute>
>;

using EVTReduce = cutlass::epilogue::fusion::XeEVT<ColReduction, Accum>;

using IdentityCompute = cutlass::epilogue::fusion::XeCompute<
    cutlass::epilogue::thread::Identity,
    ElementOutput, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTRoot = cutlass::epilogue::fusion::XeEVT<IdentityCompute, EVTReduce>;

using CollectiveEpilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Xe20, cutlass::arch::OpClassTensorOp,
    TileShape,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementOutput, StrideC, 8,
    ElementOutput, StrideD, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    EVTRoot
  >::CollectiveOp;

using CollectiveMainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Xe20, cutlass::arch::OpClassTensorOp,
    ElementInputA, LayoutA, 8,
    ElementInputB, LayoutB, 8,
    ElementAccumulator,
    TileShape,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

int main(int argc, const char** argv)
{
  int M = 1024, N = 1024, K = 1024;
  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  cutlass::DeviceAllocation<ElementInputA> block_A(M * K);
  cutlass::DeviceAllocation<ElementInputB> block_B(K * N);
  cutlass::DeviceAllocation<ElementOutput> block_D(M * N);
  cutlass::DeviceAllocation<ElementCompute> block_row_sum(M);

  initialize_block(block_A, 2023);
  initialize_block(block_B, 2022);

  // Zero out row_sum
  compat::get_default_queue().memset(block_row_sum.get(), 0, M * sizeof(ElementCompute));
  compat::wait();

  // ColReduction args
  typename ColReduction::Arguments reduction_args;
  reduction_args.ptr_col = block_row_sum.get();
  reduction_args.reduction_identity = ElementCompute(0);
  reduction_args.dCol = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};

  typename Accum::Arguments accum_args{};
  typename EVTReduce::Arguments evt_reduce_args{accum_args, reduction_args};

  typename IdentityCompute::Arguments identity_args{};
  typename EVTRoot::Arguments evt_args{evt_reduce_args, identity_args};

  using EpilogueArguments = typename Gemm::GemmKernel::EpilogueArguments;
  EpilogueArguments epilogue_arguments{
    evt_args,
    nullptr,
    stride_C,
    block_D.get(),
    stride_D
  };

  typename Gemm::GemmKernel::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    epilogue_arguments,
    hw_info
  };

  Gemm gemm_op;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  auto status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    printf("can_implement failed: %d\n", (int)status);
    return 1;
  }
  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm_op.run());
  compat::wait();

  printf("Minimal ColReduction test: SUCCESS (compiled and ran)\n");
  return 0;
}
