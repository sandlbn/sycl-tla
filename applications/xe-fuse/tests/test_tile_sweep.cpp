
// xe-fuse: Tile shape sweep benchmark
//
// Compile: icpx $FLAGS -DKERNEL_ID=N -o sweep_N test_tile_sweep.cpp
//   KERNEL_ID: 0=bare GEMM(bf16), 1=K1(RmsNorm), 2=K3(RoPE), 3=K4(RmsNorm+RoPE), 4=K2(SwiGLU)
//
// Run: ./sweep_N --tile=T --m=4096 --n=4096 --k=4096 --iterations=200
//   T: 0=256x256x32, 1=256x128x32, 2=128x256x32, 3=128x128x32,
//      4=64x256x32,  5=64x128x32,  6=64x64x32

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

#include <cute/tensor.hpp>

#include "sycl_common.hpp"
#include "helper.h"

#if KERNEL_ID == 1
#include "xe-fuse/kernels/gemm_rmsnorm.hpp"
#elif KERNEL_ID == 2
#include "xe-fuse/kernels/gemm_rope.hpp"
#elif KERNEL_ID == 3
#include "xe-fuse/kernels/gemm_rmsnorm_rope.hpp"
#elif KERNEL_ID == 4
#include "xe-fuse/kernels/gemm_rmsnorm_swiglu.hpp"
#endif

#ifndef KERNEL_ID
#error "Define KERNEL_ID: 0=bare, 1=K1, 2=K3, 3=K4, 4=K2(SwiGLU)"
#endif

using namespace cute;

using T0 = Shape<_256, _256, _32>;
using T1 = Shape<_256, _128, _32>;
using T2 = Shape<_128, _256, _32>;
using T3 = Shape<_128, _128, _32>;
using T4 = Shape< _64, _256, _32>;
using T5 = Shape< _64, _128, _32>;
using T6 = Shape< _64,  _64, _32>;

static const char* tile_names[] = {
    "256x256x32", "256x128x32", "128x256x32", "128x128x32",
    "64x256x32",  "64x128x32",  "64x64x32"
};
static const char* kernel_names[] = {
    "Bare_GEMM(bf16)", "K1(RmsNorm)", "K3(RoPE)", "K4(RmsNorm+RoPE)", "K2(SwiGLU)"
};

struct Options {
  int m = 4096, n = 4096, k = 4096, l = 1;
  int iterations = 200;
  int tile = 0;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("m", m, 4096);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("iterations", iterations, 200);
    cmd.get_cmd_line_argument("tile", tile, 0);
  }
};

// ============================================================
// KERNEL_ID == 0: Bare GEMM with bf16 output (via CollectiveBuilder)
// ============================================================
#if KERNEL_ID == 0

template <typename TileShape_>
struct BareGemm {
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementD = cutlass::bfloat16_t;
  using ElementAcc = float;
  using ElementCompute = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using TileShape = TileShape_;
  using StrideC = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideD = cute::Stride<int64_t, cute::Int<1>, int64_t>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Xe20, cutlass::arch::OpClassTensorOp,
      TileShape, cute::Shape<cute::_1, cute::_1, cute::_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementD, StrideC, 8,
      ElementD, StrideD, 8,
      cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Xe20, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8, ElementB, LayoutB, 8,
      ElementAcc, TileShape, cute::Shape<cute::_1, cute::_1, cute::_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename TileShape>
int run(Options& opts) {
  using Config = BareGemm<TileShape>;
  using GemmOp = typename Config::Gemm;
  int M = opts.m, N = opts.n, K = opts.k, L = opts.l;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using StrideA = typename GemmOp::GemmKernel::StrideA;
  using StrideB = typename GemmOp::GemmKernel::StrideB;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, L));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, L));
  auto stride_C = cutlass::make_cute_packed_stride(typename Config::StrideC{}, make_shape(M, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(typename Config::StrideD{}, make_shape(M, N, L));

  cutlass::DeviceAllocation<typename Config::ElementA> block_A(static_cast<size_t>(M) * K * L);
  cutlass::DeviceAllocation<typename Config::ElementB> block_B(static_cast<size_t>(K) * N * L);
  cutlass::DeviceAllocation<typename Config::ElementD> block_D(static_cast<size_t>(M) * N * L);

  initialize_block(block_A, 2023);
  initialize_block(block_B, 2022);

  typename GemmOp::GemmKernel::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, L},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    {{1.0f, 0.0f}, nullptr, stride_C, block_D.get(), stride_D},
    hw_info
  };

  GemmOp gemm_op;
  size_t ws_size = GemmOp::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(ws_size);

  CUTLASS_CHECK(gemm_op.can_implement(arguments));
  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm_op.run());
  compat::wait();

  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < opts.iterations; ++i) gemm_op.run();
  compat::wait();

  float time_s = timer.seconds() / opts.iterations;
  double tflops = (2.0 * M * N * K * L) * 1e-12;
  printf("[Tile %-11s] %-20s %7.3f TFlop/s  %7.4f ms  (%dx%dx%d)\n",
         tile_names[opts.tile], kernel_names[KERNEL_ID],
         tflops / time_s, time_s * 1000, M, N, K);
  printf("SWEEP: kernel=%s tile=%s M=%d N=%d K=%d tflops=%.3f ms=%.4f\n",
         kernel_names[KERNEL_ID], tile_names[opts.tile], M, N, K,
         tflops / time_s, time_s * 1000);
  return 0;
}

// ============================================================
// KERNEL_ID == 1: K1 (GEMM + RmsNorm)
// ============================================================
#elif KERNEL_ID == 1

template <typename TileShape>
int run(Options& opts) {
  using Config = xe_fuse::GemmRmsNorm<
      cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t,
      float, float, float, TileShape>;
  using GemmOp = typename Config::Gemm;
  int M = opts.m, N = opts.n, K = opts.k, L = opts.l;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using StrideA = typename GemmOp::GemmKernel::StrideA;
  using StrideB = typename GemmOp::GemmKernel::StrideB;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, L));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, L));
  auto stride_C = cutlass::make_cute_packed_stride(typename Config::StrideC{}, make_shape(M, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(typename Config::StrideD{}, make_shape(M, N, L));

  cutlass::DeviceAllocation<typename Config::ElementA> block_A(static_cast<size_t>(M) * K * L);
  cutlass::DeviceAllocation<typename Config::ElementB> block_B(static_cast<size_t>(K) * N * L);
  cutlass::DeviceAllocation<typename Config::ElementD> block_D(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<typename Config::ElementScale> block_scale(static_cast<size_t>(M) * L);

  initialize_block(block_A, 2023);
  initialize_block(block_B, 2022);
  initialize_block(block_scale, 42);

  auto evt_args = Config::make_evt_args(block_scale.get(), M);

  typename GemmOp::GemmKernel::EpilogueArguments epilogue_args{
    evt_args, nullptr, stride_C, block_D.get(), stride_D
  };
  typename GemmOp::GemmKernel::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, L},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    epilogue_args, hw_info
  };

  GemmOp gemm_op;
  size_t ws_size = GemmOp::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(ws_size);

  CUTLASS_CHECK(gemm_op.can_implement(arguments));
  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm_op.run());
  compat::wait();

  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < opts.iterations; ++i) gemm_op.run();
  compat::wait();

  float time_s = timer.seconds() / opts.iterations;
  double tflops = (2.0 * M * N * K * L) * 1e-12;
  printf("[Tile %-11s] %-20s %7.3f TFlop/s  %7.4f ms  (%dx%dx%d)\n",
         tile_names[opts.tile], kernel_names[KERNEL_ID],
         tflops / time_s, time_s * 1000, M, N, K);
  printf("SWEEP: kernel=%s tile=%s M=%d N=%d K=%d tflops=%.3f ms=%.4f\n",
         kernel_names[KERNEL_ID], tile_names[opts.tile], M, N, K,
         tflops / time_s, time_s * 1000);
  return 0;
}

// ============================================================
// KERNEL_ID == 2: K3 (GEMM + RoPE)
// ============================================================
#elif KERNEL_ID == 2

template <typename TileShape>
int run(Options& opts) {
  using Config = xe_fuse::GemmRoPE<
      cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t,
      float, float, float, TileShape>;
  using GemmOp = typename Config::Gemm;
  int M = opts.m, N = opts.n, K = opts.k, L = opts.l;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using StrideA = typename GemmOp::GemmKernel::StrideA;
  using StrideB = typename GemmOp::GemmKernel::StrideB;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, L));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, L));
  auto stride_C = cutlass::make_cute_packed_stride(typename Config::StrideC{}, make_shape(M, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(typename Config::StrideD{}, make_shape(M, N, L));
  auto stride_cs = cutlass::make_cute_packed_stride(typename Config::StrideCosSin{}, make_shape(M, N, L));

  cutlass::DeviceAllocation<typename Config::ElementA> block_A(static_cast<size_t>(M) * K * L);
  cutlass::DeviceAllocation<typename Config::ElementB> block_B(static_cast<size_t>(K) * N * L);
  cutlass::DeviceAllocation<typename Config::ElementD> block_D(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<typename Config::ElementCosSin> block_cos_sin(static_cast<size_t>(M) * N * L);

  initialize_block(block_A, 2023);
  initialize_block(block_B, 2022);
  initialize_block(block_cos_sin, 2024);

  auto evt_args = Config::make_evt_args(block_cos_sin.get(), stride_cs);

  typename GemmOp::GemmKernel::EpilogueArguments epilogue_args{
    evt_args, nullptr, stride_C, block_D.get(), stride_D
  };
  typename GemmOp::GemmKernel::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, L},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    epilogue_args, hw_info
  };

  GemmOp gemm_op;
  size_t ws_size = GemmOp::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(ws_size);

  CUTLASS_CHECK(gemm_op.can_implement(arguments));
  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm_op.run());
  compat::wait();

  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < opts.iterations; ++i) gemm_op.run();
  compat::wait();

  float time_s = timer.seconds() / opts.iterations;
  double tflops = (2.0 * M * N * K * L) * 1e-12;
  printf("[Tile %-11s] %-20s %7.3f TFlop/s  %7.4f ms  (%dx%dx%d)\n",
         tile_names[opts.tile], kernel_names[KERNEL_ID],
         tflops / time_s, time_s * 1000, M, N, K);
  printf("SWEEP: kernel=%s tile=%s M=%d N=%d K=%d tflops=%.3f ms=%.4f\n",
         kernel_names[KERNEL_ID], tile_names[opts.tile], M, N, K,
         tflops / time_s, time_s * 1000);
  return 0;
}

// ============================================================
// KERNEL_ID == 3: K4 (GEMM + RmsNorm + RoPE)
// ============================================================
#elif KERNEL_ID == 3

template <typename TileShape>
int run(Options& opts) {
  using Config = xe_fuse::GemmRmsNormRoPE<
      cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t,
      float, float, float, float, TileShape>;
  using GemmOp = typename Config::Gemm;
  int M = opts.m, N = opts.n, K = opts.k, L = opts.l;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using StrideA = typename GemmOp::GemmKernel::StrideA;
  using StrideB = typename GemmOp::GemmKernel::StrideB;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, L));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, L));
  auto stride_C = cutlass::make_cute_packed_stride(typename Config::StrideC{}, make_shape(M, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(typename Config::StrideD{}, make_shape(M, N, L));
  auto stride_cs = cutlass::make_cute_packed_stride(typename Config::StrideCosSin{}, make_shape(M, N, L));

  cutlass::DeviceAllocation<typename Config::ElementA> block_A(static_cast<size_t>(M) * K * L);
  cutlass::DeviceAllocation<typename Config::ElementB> block_B(static_cast<size_t>(K) * N * L);
  cutlass::DeviceAllocation<typename Config::ElementD> block_D(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<typename Config::ElementScale> block_scale(static_cast<size_t>(M) * L);
  cutlass::DeviceAllocation<typename Config::ElementCosSin> block_cos_sin(static_cast<size_t>(M) * N * L);

  initialize_block(block_A, 2023);
  initialize_block(block_B, 2022);
  initialize_block(block_scale, 42);
  initialize_block(block_cos_sin, 2024);

  auto evt_args = Config::make_evt_args(block_scale.get(), M, block_cos_sin.get(), stride_cs);

  typename GemmOp::GemmKernel::EpilogueArguments epilogue_args{
    evt_args, nullptr, stride_C, block_D.get(), stride_D
  };
  typename GemmOp::GemmKernel::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, L},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    epilogue_args, hw_info
  };

  GemmOp gemm_op;
  size_t ws_size = GemmOp::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(ws_size);

  CUTLASS_CHECK(gemm_op.can_implement(arguments));
  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm_op.run());
  compat::wait();

  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < opts.iterations; ++i) gemm_op.run();
  compat::wait();

  float time_s = timer.seconds() / opts.iterations;
  double tflops = (2.0 * M * N * K * L) * 1e-12;
  printf("[Tile %-11s] %-20s %7.3f TFlop/s  %7.4f ms  (%dx%dx%d)\n",
         tile_names[opts.tile], kernel_names[KERNEL_ID],
         tflops / time_s, time_s * 1000, M, N, K);
  printf("SWEEP: kernel=%s tile=%s M=%d N=%d K=%d tflops=%.3f ms=%.4f\n",
         kernel_names[KERNEL_ID], tile_names[opts.tile], M, N, K,
         tflops / time_s, time_s * 1000);
  return 0;
}

// ============================================================
// KERNEL_ID == 4: K2 (GEMM + RmsNorm + SwiGLU)
// ============================================================
#elif KERNEL_ID == 4

template <typename TileShape>
int run(Options& opts) {
  using Config = xe_fuse::GemmRmsNormSwiGLU<
      cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t,
      float, float, float, TileShape>;
  using GemmOp = typename Config::Gemm;
  int M = opts.m, N = opts.n, K = opts.k, L = opts.l;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using StrideA = typename GemmOp::GemmKernel::StrideA;
  using StrideB = typename GemmOp::GemmKernel::StrideB;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, L));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, L));
  auto stride_C = cutlass::make_cute_packed_stride(typename Config::StrideC{}, make_shape(M, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(typename Config::StrideD{}, make_shape(M, N, L));

  cutlass::DeviceAllocation<typename Config::ElementA> block_A(static_cast<size_t>(M) * K * L);
  cutlass::DeviceAllocation<typename Config::ElementB> block_B(static_cast<size_t>(K) * N * L);
  cutlass::DeviceAllocation<typename Config::ElementD> block_D(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<typename Config::ElementScale> block_scale(static_cast<size_t>(M) * L);

  initialize_block(block_A, 2023);
  initialize_block(block_B, 2022);
  initialize_block(block_scale, 42);

  auto evt_args = Config::make_evt_args(block_scale.get(), M);

  typename GemmOp::GemmKernel::EpilogueArguments epilogue_args{
    evt_args, nullptr, stride_C, block_D.get(), stride_D
  };
  typename GemmOp::GemmKernel::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, L},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    epilogue_args, hw_info
  };

  GemmOp gemm_op;
  size_t ws_size = GemmOp::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(ws_size);

  CUTLASS_CHECK(gemm_op.can_implement(arguments));
  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm_op.run());
  compat::wait();

  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < opts.iterations; ++i) gemm_op.run();
  compat::wait();

  float time_s = timer.seconds() / opts.iterations;
  double tflops = (2.0 * M * N * K * L) * 1e-12;
  printf("[Tile %-11s] %-20s %7.3f TFlop/s  %7.4f ms  (%dx%dx%d)\n",
         tile_names[opts.tile], kernel_names[KERNEL_ID],
         tflops / time_s, time_s * 1000, M, N, K);
  printf("SWEEP: kernel=%s tile=%s M=%d N=%d K=%d tflops=%.3f ms=%.4f\n",
         kernel_names[KERNEL_ID], tile_names[opts.tile], M, N, K,
         tflops / time_s, time_s * 1000);
  return 0;
}

#endif

int main(int argc, const char** argv) {
  Options opts;
  opts.parse(argc, argv);

  if (opts.tile < 0 || opts.tile > 6) {
    fprintf(stderr, "Invalid --tile=%d (valid: 0-6)\n", opts.tile);
    return 1;
  }

  switch (opts.tile) {
    case 0: return run<T0>(opts);
    case 1: return run<T1>(opts);
    case 2: return run<T2>(opts);
    case 3: return run<T3>(opts);
    case 4: return run<T4>(opts);
    case 5: return run<T5>(opts);
    case 6: return run<T6>(opts);
  }
  return 1;
}
