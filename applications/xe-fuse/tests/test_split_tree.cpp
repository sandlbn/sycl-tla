
// xe-fuse test: split-tree pattern (dual output) + new activation ops
//
// Tests:
//   1. K0 split-tree: D = gamma * (acc + residual), aux = acc + residual
//   2. GeLU activation fused into GEMM epilogue
//
// Compile with -DTEST_ID=0 for split-tree, -DTEST_ID=1 for GeLU

#include "xe-fuse/builder/epilogue_builder.hpp"

#if TEST_ID == 0
#include "xe-fuse/kernels/gemm_dual_output.hpp"
#endif

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include "sycl_common.hpp"
#include "helper.h"

#include <random>
#include <cmath>

using namespace cute;
using bf16 = cutlass::bfloat16_t;

#ifndef TEST_ID
#define TEST_ID 0
#endif

struct Options {
  int m = 2048, n = 2048, k = 2048, l = 1;
  int iterations = 100;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("m", m, 2048);
    cmd.get_cmd_line_argument("n", n, 2048);
    cmd.get_cmd_line_argument("k", k, 2048);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
  }
};

#if TEST_ID == 0
// ================================================================
// Test 0: Split-tree K0 — dual output
// ================================================================

using K0 = xe_fuse::GemmDualOutput<>;
using GemmOp = K0::Gemm;

int main(int argc, const char** argv) {
  Options opts;
  opts.parse(argc, argv);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  int M = opts.m, N = opts.n, K = opts.k, L = opts.l;

  using StrideA = typename GemmOp::GemmKernel::StrideA;
  using StrideB = typename GemmOp::GemmKernel::StrideB;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, L));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, L));
  auto stride_C = cutlass::make_cute_packed_stride(K0::StrideC{}, make_shape(M, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(K0::StrideD{}, make_shape(M, N, L));
  auto stride_aux = cutlass::make_cute_packed_stride(K0::StrideAux{}, make_shape(M, N, L));

  size_t mn = static_cast<size_t>(M) * N * L;

  cutlass::DeviceAllocation<bf16> block_A(static_cast<size_t>(M) * K * L);
  cutlass::DeviceAllocation<bf16> block_B(static_cast<size_t>(K) * N * L);
  cutlass::DeviceAllocation<bf16> block_D(mn);       // primary output: gamma * (acc + residual)
  cutlass::DeviceAllocation<bf16> block_aux(mn);      // aux output: acc + residual (raw)
  cutlass::DeviceAllocation<bf16> block_residual(mn);
  cutlass::DeviceAllocation<bf16> block_ref_D(mn);
  cutlass::DeviceAllocation<bf16> block_ref_aux(mn);
  cutlass::DeviceAllocation<float> block_gamma(static_cast<size_t>(N) * L);

  initialize_block(block_A, 2023);
  initialize_block(block_B, 2022);
  initialize_block(block_residual, 2021);

  std::vector<float> h_gamma(static_cast<size_t>(N) * L);
  {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.5f, 1.5f);
    for (auto& v : h_gamma) v = dist(rng);
    compat::get_default_queue().memcpy(block_gamma.get(), h_gamma.data(),
                                        h_gamma.size() * sizeof(float));
  }
  compat::wait();

  auto evt_args = K0::make_evt_args(
      block_residual.get(), stride_D,
      block_gamma.get(), N,
      block_aux.get(), stride_aux);

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
  size_t ws = GemmOp::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(ws);

  auto status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "can_implement failed for split-tree K0" << std::endl;
    return 1;
  }
  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm_op.run());
  compat::wait();

  // Reference: float GEMM, then residual add, then gamma scale
  cutlass::DeviceAllocation<float> block_gemm_f32(mn);
  cutlass::TensorRef ref_A(block_A.get(), cutlass::layout::RowMajor::packed({M, K}));
  cutlass::TensorRef ref_B(block_B.get(), cutlass::layout::RowMajor::packed({K, N}));
  cutlass::TensorRef ref_C(block_gemm_f32.get(), cutlass::layout::RowMajor::packed({M, N}));
  cutlass::TensorRef ref_D_f32(block_gemm_f32.get(), cutlass::layout::RowMajor::packed({M, N}));

  cutlass::reference::device::GemmComplex(
    {M, N, K}, float(1), ref_A, cutlass::ComplexTransform::kNone,
    ref_B, cutlass::ComplexTransform::kNone, float(0),
    ref_C, ref_D_f32, float(0), L, M * K, K * N, M * N, M * N);
  compat::wait();

  {
    auto* gemm_ptr = block_gemm_f32.get();
    auto* res_ptr = block_residual.get();
    auto* gamma_ptr = block_gamma.get();
    auto* ref_d = block_ref_D.get();
    auto* ref_a = block_ref_aux.get();
    int n_val = N;

    compat::get_default_queue().parallel_for(
      sycl::range<1>(mn), [=](sycl::id<1> idx) {
        int64_t i = idx[0];
        int col = static_cast<int>(i % n_val);
        float sum = gemm_ptr[i] + static_cast<float>(res_ptr[i]);
        ref_a[i] = static_cast<bf16>(sum);               // aux: raw sum
        ref_d[i] = static_cast<bf16>(sum * gamma_ptr[col]); // primary: gamma * sum
      });
  }
  compat::wait();

  bool d_passed = cutlass::reference::device::BlockCompareRelativelyEqual(
    block_ref_D.get(), block_D.get(), mn,
    static_cast<bf16>(0.05f), static_cast<bf16>(0.05f));
  bool aux_passed = cutlass::reference::device::BlockCompareRelativelyEqual(
    block_ref_aux.get(), block_aux.get(), mn,
    static_cast<bf16>(0.05f), static_cast<bf16>(0.05f));

  std::cout << "Split-tree K0 (dual output):" << std::endl;
  std::cout << "  Primary D (gamma * sum): " << (d_passed ? "Passed" : "Failed") << std::endl;
  std::cout << "  Aux output (raw sum):    " << (aux_passed ? "Passed" : "Failed") << std::endl;

  if (!d_passed || !aux_passed) return 1;

  if (opts.iterations > 0) {
    GPU_Clock timer;
    timer.start();
    for (int i = 0; i < opts.iterations; ++i) gemm_op.run();
    compat::wait();

    float time_s = timer.seconds() / opts.iterations;
    double tflops = (2.0 * M * N * K * L) * 1e-12;
    std::cout << "Problem Size: " << M << 'x' << N << 'x' << K << 'x' << L << std::endl;
    printf("K0_SplitTree: [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / time_s, time_s * 1000);
  }

  return 0;
}

#elif TEST_ID == 1
// ================================================================
// Test 1: GeLU fused into GEMM epilogue
// ================================================================

namespace b = xe_fuse::builder;
using TileShape = Shape<_256, _256, _32>;
using GeLU_EVT = b::GeLU<b::Acc>;
using KernelConfig = b::MakeGemm<GeLU_EVT, bf16, bf16, bf16, float, float, TileShape>;
using GemmOp = typename KernelConfig::Gemm;

int main(int argc, const char** argv) {
  Options opts;
  opts.parse(argc, argv);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  int M = opts.m, N = opts.n, K = opts.k, L = opts.l;

  using StrideA = typename GemmOp::GemmKernel::StrideA;
  using StrideB = typename GemmOp::GemmKernel::StrideB;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, L));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, L));
  auto stride_C = cutlass::make_cute_packed_stride(KernelConfig::StrideC{}, make_shape(M, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(KernelConfig::StrideD{}, make_shape(M, N, L));

  size_t mn = static_cast<size_t>(M) * N * L;

  cutlass::DeviceAllocation<bf16> block_A(static_cast<size_t>(M) * K * L);
  cutlass::DeviceAllocation<bf16> block_B(static_cast<size_t>(K) * N * L);
  cutlass::DeviceAllocation<bf16> block_D(mn);
  cutlass::DeviceAllocation<bf16> block_ref_D(mn);

  initialize_block(block_A, 2023);
  initialize_block(block_B, 2022);
  compat::wait();

  // EVT args: GeLU<Acc> — GeLU has no args, Acc has no args
  typename b::Acc::Arguments acc_args{};
  typename xe_fuse::XeElementwiseCompute<xe_fuse::GeLUFn>::Arguments gelu_args{};
  typename GeLU_EVT::Arguments evt_args{acc_args, gelu_args};

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
  size_t ws = GemmOp::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(ws);

  CUTLASS_CHECK(gemm_op.can_implement(arguments));
  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm_op.run());
  compat::wait();

  // Reference: float GEMM → GeLU
  cutlass::DeviceAllocation<float> block_gemm_f32(mn);
  cutlass::TensorRef ref_A(block_A.get(), cutlass::layout::RowMajor::packed({M, K}));
  cutlass::TensorRef ref_B(block_B.get(), cutlass::layout::RowMajor::packed({K, N}));
  cutlass::TensorRef ref_C(block_gemm_f32.get(), cutlass::layout::RowMajor::packed({M, N}));
  cutlass::TensorRef ref_D_f32(block_gemm_f32.get(), cutlass::layout::RowMajor::packed({M, N}));

  cutlass::reference::device::GemmComplex(
    {M, N, K}, float(1), ref_A, cutlass::ComplexTransform::kNone,
    ref_B, cutlass::ComplexTransform::kNone, float(0),
    ref_C, ref_D_f32, float(0), L, M * K, K * N, M * N, M * N);
  compat::wait();

  {
    auto* src = block_gemm_f32.get();
    auto* ref = block_ref_D.get();
    compat::get_default_queue().parallel_for(
      sycl::range<1>(mn), [=](sycl::id<1> idx) {
        int64_t i = idx[0];
        float x = src[i];
        float gelu = x * 0.5f * (1.0f + sycl::erf(x * 0.7071067811865475f));
        ref[i] = static_cast<bf16>(gelu);
      });
  }
  compat::wait();

  bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(
    block_ref_D.get(), block_D.get(), mn,
    static_cast<bf16>(0.05f), static_cast<bf16>(0.05f));

  std::cout << "GEMM + GeLU: " << (passed ? "Passed" : "Failed") << std::endl;
  if (!passed) return 1;

  if (opts.iterations > 0) {
    GPU_Clock timer;
    timer.start();
    for (int i = 0; i < opts.iterations; ++i) gemm_op.run();
    compat::wait();

    float time_s = timer.seconds() / opts.iterations;
    double tflops = (2.0 * M * N * K * L) * 1e-12;
    std::cout << "Problem Size: " << M << 'x' << N << 'x' << K << 'x' << L << std::endl;
    printf("GEMM_GeLU: [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / time_s, time_s * 1000);
  }

  return 0;
}

#endif
