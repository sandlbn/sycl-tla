
// xe-fuse test: builder API validation
//
// Defines all 5 CODA kernels using the builder aliases and verifies
// they compile and produce correct results. Uses K1 for end-to-end
// correctness check since it's the simplest.

#include "xe-fuse/builder/epilogue_builder.hpp"

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include "sycl_common.hpp"
#include "helper.h"

#include <random>
#include <iostream>

using namespace cute;
namespace b = xe_fuse::builder;

// ============================================================
// Define all 5 kernels using the builder
// ============================================================

using TileShape = Shape<_256, _256, _32>;
using bf16 = cutlass::bfloat16_t;

// K0a: D = gamma[n] * (acc + residual)
using K0a_EVT = b::ScaleCols<b::AddResidual<bf16>, TileShape, float>;
using K0a = b::MakeGemm<K0a_EVT, bf16, bf16, bf16, float, float, TileShape>;

// K1: D = acc * R[m]
using K1_EVT = b::ScaleRows<b::Acc, TileShape, float>;
using K1 = b::MakeGemm<K1_EVT, bf16, bf16, bf16, float, float, TileShape>;

// K2: D = SwiGLU(acc * R[m])
using K2_EVT = b::SwiGLU<b::ScaleRows<b::Acc, TileShape, float>>;
using K2 = b::MakeGemm<K2_EVT, bf16, bf16, bf16, float, float, TileShape>;

// K3: D = RoPE(acc, cos_sin)
using K3_EVT = b::RoPE<float>;
using K3 = b::MakeGemm<K3_EVT, bf16, bf16, bf16, float, float, TileShape>;

// K4: D = RoPE(acc * R[m], cos_sin)
using K4_EVT = b::RoPEComposed<b::ScaleRows<b::Acc, TileShape, float>, float>;
using K4 = b::MakeGemm<K4_EVT, bf16, bf16, bf16, float, float, TileShape>;

// ============================================================
// Correctness test using K1 (simplest: D = acc * R)
// ============================================================

struct Options {
  int m = 4096, n = 4096, k = 4096, l = 1;
  int iterations = 100;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("m", m, 4096);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
  }
};

int main(int argc, const char** argv) {
  Options opts;
  opts.parse(argc, argv);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  int M = opts.m, N = opts.n, K = opts.k, L = opts.l;

  using GemmOp = K1::Gemm;
  using StrideA = typename GemmOp::GemmKernel::StrideA;
  using StrideB = typename GemmOp::GemmKernel::StrideB;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, L));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, L));
  auto stride_C = cutlass::make_cute_packed_stride(K1::StrideC{}, make_shape(M, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(K1::StrideD{}, make_shape(M, N, L));

  cutlass::DeviceAllocation<bf16> block_A(static_cast<size_t>(M) * K * L);
  cutlass::DeviceAllocation<bf16> block_B(static_cast<size_t>(K) * N * L);
  cutlass::DeviceAllocation<bf16> block_D(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<bf16> block_ref_D(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<float> block_scale(static_cast<size_t>(M) * L);

  initialize_block(block_A, 2023);
  initialize_block(block_B, 2022);

  std::vector<float> h_scale(static_cast<size_t>(M) * L);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.5f, 1.5f);
  for (auto& v : h_scale) v = dist(rng);
  compat::get_default_queue().memcpy(block_scale.get(), h_scale.data(), h_scale.size() * sizeof(float));
  compat::wait();

  // Build K1 EVT arguments manually (builder provides types, not arg construction)
  typename b::Acc::Arguments accum_args{};

  typename b::ColBroadcast<0, TileShape, float>::Arguments scale_args;
  scale_args.ptr_col = block_scale.get();
  scale_args.null_default = float(1);
  scale_args.dCol = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};

  typename b::MulOp<>::Arguments mul_args{};

  typename K1_EVT::Arguments evt_args{accum_args, scale_args, mul_args};

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

  // Reference GEMM + scaling
  cutlass::TensorRef ref_A(block_A.get(), cutlass::layout::RowMajor::packed({M, K}));
  cutlass::TensorRef ref_B(block_B.get(), cutlass::layout::RowMajor::packed({K, N}));
  cutlass::TensorRef ref_C(block_ref_D.get(), cutlass::layout::RowMajor::packed({M, N}));
  cutlass::TensorRef ref_D(block_ref_D.get(), cutlass::layout::RowMajor::packed({M, N}));

  cutlass::reference::device::GemmComplex(
    {M, N, K}, float(1), ref_A, cutlass::ComplexTransform::kNone,
    ref_B, cutlass::ComplexTransform::kNone, float(0),
    ref_C, ref_D, float(0), L, M * K, K * N, M * N, M * N);
  compat::wait();

  {
    auto* ref_ptr = block_ref_D.get();
    auto* scale_ptr = block_scale.get();
    int64_t total = static_cast<int64_t>(M) * N * L;
    compat::get_default_queue().parallel_for(
      sycl::range<1>(total),
      [=](sycl::id<1> idx) {
        int64_t i = idx[0];
        int m = static_cast<int>((i / N) % M);
        int batch = static_cast<int>(i / (M * N));
        float val = static_cast<float>(ref_ptr[i]);
        float s = scale_ptr[batch * M + m];
        ref_ptr[i] = static_cast<bf16>(val * s);
      }
    );
  }
  compat::wait();

  bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(
    block_ref_D.get(), block_D.get(), block_D.size(),
    static_cast<bf16>(0.05f), static_cast<bf16>(0.05f));

  std::cout << "Builder K1 correctness: " << (passed ? "Passed" : "Failed") << std::endl;

  // Static checks: verify all 5 kernel types are well-formed
  std::cout << "K0a Gemm kernel size: " << sizeof(typename K0a::GemmKernel) << std::endl;
  std::cout << "K1  Gemm kernel size: " << sizeof(typename K1::GemmKernel) << std::endl;
  std::cout << "K2  Gemm kernel size: " << sizeof(typename K2::GemmKernel) << std::endl;
  std::cout << "K3  Gemm kernel size: " << sizeof(typename K3::GemmKernel) << std::endl;
  std::cout << "K4  Gemm kernel size: " << sizeof(typename K4::GemmKernel) << std::endl;

  if (opts.iterations > 0) {
    GPU_Clock timer;
    timer.start();
    for (int i = 0; i < opts.iterations; ++i) gemm_op.run();
    compat::wait();

    float time_s = timer.seconds() / opts.iterations;
    double tflops = (2.0 * M * N * K * L) * 1e-12;
    printf("Builder K1 perf: [%4.3f]TFlop/s  (%6.4f)ms  %dx%dx%d\n",
           tflops / time_s, time_s * 1000, M, N, K);
  }

  return passed ? 0 : 1;
}
