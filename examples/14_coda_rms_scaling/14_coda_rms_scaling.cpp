
/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief CODA Phase 0A: GEMM with per-row RMS scaling epilogue (K5)

    Implements D[m,n] = acc[m,n] * rms_scale[m]

    This is the simplest CODA kernel: a GEMM whose epilogue multiplies each row
    of the accumulator by a precomputed per-row scalar (the inverse RMS normalization
    factor). Uses raw EVT composition with XeColBroadcast + XeCompute<multiplies>.

    Based on the EVT composition pattern from 05_bmg_gemm_with_epilogue_lincombauxstore.
*/

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

///////////////////////////////////////////////////////////////////////////////////////////////////

struct Options {

  bool help;
  bool error;

  int m, n, k, l, iterations, verify;
  float alpha, beta;

  Options():
    help(false),
    error(false),
    m(4096), n(4096), k(4096), l(1), iterations(100), verify(1),
    alpha(1.f), beta(0.f)
  { }

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 4096);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("verify", verify, 1);
  }

  std::ostream & print_usage(std::ostream &out) const {
    out << "CODA K5: GEMM + Per-Row RMS Scaling\n\n"
      << "  D[m,n] = acc[m,n] * rms_scale[m]\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the L extent (batch count) of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha (applied before scaling)\n"
      << "  --beta=<f32>                Epilogue scalar beta (unused, kept for interface compat)\n"
      << "  --iterations=<int>          Iterations\n"
      << "  --verify=<int>              Specify whether to verify.\n\n";
    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// EVT tree: D = acc * rms_scale[row]
//
// XeEVT<Compute(multiplies),
//   AccFetch,                    // child 0: accumulator
//   XeColBroadcast<rms_scale>   // child 1: per-row vector broadcast across N
// >

using ElementAccumulator = float;
using ElementCompute = float;
using ElementInputA = bfloat16_t;
using ElementInputB = bfloat16_t;
using ElementOutput = bfloat16_t;
using ElementScale = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using TileShape = Shape<_256, _256, _32>;

using StrideC = cute::Stride<int64_t, cute::Int<1>, int64_t>;
using StrideD = cute::Stride<int64_t, cute::Int<1>, int64_t>;

// EVT leaf: fetch accumulator
using Accum = cutlass::epilogue::fusion::XeAccFetch;

// EVT leaf: broadcast per-row scale vector along N dimension
// XeColBroadcast: broadcasts a column vector (one value per M row) across N columns
// Template params: Stages, CtaTileShapeMNK, ElementBroadcast, ElementCompute, StrideMNL, Alignment
using ScaleBroadcast = cutlass::epilogue::fusion::XeColBroadcast<
    0,              // Stages
    TileShape,      // CtaTileShapeMNK
    ElementScale,   // Element type of the scale vector
    ElementCompute, // Compute type
    cute::Stride<cute::Int<1>, cute::Int<0>, int64_t>,  // StrideMNL: stride 1 along M, 0 along N
    128 / cutlass::sizeof_bits_v<ElementScale>           // Alignment
>;

// EVT node: acc * scale
using ScaleCompute = cutlass::epilogue::fusion::XeCompute<
    cutlass::multiplies, ElementOutput, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest
>;

// Full EVT tree: D = acc * scale[row]
using EVTRmsScale = cutlass::epilogue::fusion::XeEVT<
    ScaleCompute,
    Accum,
    ScaleBroadcast
>;

// Build epilogue using CollectiveBuilder with the EVT tree
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
    EVTRmsScale
  >::CollectiveOp;

// Build mainloop
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
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class GemmOp>
struct ExampleRunner {

  using StrideA = typename GemmOp::GemmKernel::StrideA;
  using StrideB = typename GemmOp::GemmKernel::StrideB;
  using ElementA = typename GemmOp::ElementA;
  using ElementB = typename GemmOp::ElementB;

  using CollectiveEpi = typename GemmOp::CollectiveEpilogue;
  using ElementC = typename GemmOp::ElementC;
  using ElementD = typename CollectiveEpi::ElementOutput;
  using ProblemShapeType = typename GemmOp::GemmKernel::ProblemShape;

  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementD> block_D;
  cutlass::DeviceAllocation<ElementD> block_ref_D;
  cutlass::DeviceAllocation<ElementScale> block_scale;  // per-row RMS scale: shape [M * L]

  bool verify(const ProblemShapeType& problem_size) {
    auto [M, N, K, L] = problem_size;

    // Reference: compute acc = A * B, then D[m,n] = acc[m,n] * scale[m]
    // Step 1: compute reference GEMM with alpha=1, beta=0 into ref_D
    cutlass::TensorRef ref_A(block_A.get(), cutlass::layout::RowMajor::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), cutlass::layout::RowMajor::packed({K, N}));
    cutlass::TensorRef ref_C(block_ref_D.get(), cutlass::layout::RowMajor::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), cutlass::layout::RowMajor::packed({M, N}));

    cutlass::reference::device::GemmComplex(
          {M, N, K},
          ElementAccumulator(1),  // alpha
          ref_A,
          cutlass::ComplexTransform::kNone,
          ref_B,
          cutlass::ComplexTransform::kNone,
          ElementAccumulator(0),  // beta
          ref_C,
          ref_D,
          ElementAccumulator(0),
          L,
          M * K,
          K * N,
          M * N,
          M * N
        );

    compat::wait();

    // Step 2: apply per-row scaling on device: ref_D[m,n] *= scale[m]
    {
      auto* ref_ptr = block_ref_D.get();
      auto* scale_ptr = block_scale.get();
      int64_t total = static_cast<int64_t>(M) * N * L;
      int n_val = N;
      int m_val = M;
      compat::get_default_queue().parallel_for(
        sycl::range<1>(total),
        [=](sycl::id<1> idx) {
          int64_t i = idx[0];
          int batch = static_cast<int>(i / (m_val * n_val));
          int m = static_cast<int>((i / n_val) % m_val);
          float val = static_cast<float>(ref_ptr[i]);
          float s = static_cast<float>(scale_ptr[batch * m_val + m]);
          ref_ptr[i] = static_cast<ElementD>(val * s);
        }
      );
    }
    compat::wait();

    bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(
      block_ref_D.get(), block_D.get(), block_D.size(),
      static_cast<ElementD>(0.05f), static_cast<ElementD>(0.05f));

    return passed;
  }

  void initialize(const ProblemShapeType& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    block_A.reset(static_cast<size_t>(M) * K * L);
    block_B.reset(static_cast<size_t>(K) * N * L);
    block_D.reset(static_cast<size_t>(M) * N * L);
    block_ref_D.reset(static_cast<size_t>(M) * N * L);
    block_scale.reset(static_cast<size_t>(M) * L);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);

    // Initialize scale vector with values in [0.5, 1.5] to simulate 1/rms
    std::vector<ElementScale> h_scale(static_cast<size_t>(M) * L);
    std::mt19937 rng(seed + 42);
    std::uniform_real_distribution<float> dist(0.5f, 1.5f);
    for (auto& v : h_scale) {
      v = static_cast<ElementScale>(dist(rng));
    }
    compat::get_default_queue().memcpy(block_scale.get(), h_scale.data(), h_scale.size() * sizeof(ElementScale));
    compat::wait();
  }

  cutlass::Status run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size);

    // Build EVT arguments following the tree structure:
    // EVTRmsScale = XeEVT<ScaleCompute, Accum, ScaleBroadcast>
    // Arguments nesting: { child0_args (Accum), child1_args (ScaleBroadcast), node_args (ScaleCompute) }

    typename Accum::Arguments accum_args{};

    typename ScaleBroadcast::Arguments scale_args;
    scale_args.ptr_col = block_scale.get();
    scale_args.null_default = ElementScale(1);
    scale_args.dCol = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(options.m)};

    typename ScaleCompute::Arguments compute_args{};

    typename EVTRmsScale::Arguments evt_args{
        accum_args,     // child 0: Accum
        scale_args,     // child 1: ScaleBroadcast
        compute_args    // node: ScaleCompute
    };

    using EpilogueArguments = typename GemmOp::GemmKernel::EpilogueArguments;
    EpilogueArguments epilogue_arguments{
      evt_args,
      nullptr,   // C ptr (unused since no source fetch)
      stride_C,
      block_D.get(),
      stride_D
    };

    typename GemmOp::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B.get(), stride_B},
      epilogue_arguments,
      hw_info
    };

    GemmOp gemm_op;

    size_t workspace_size = GemmOp::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm_op.can_implement(arguments));
    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

    // Run
    CUTLASS_CHECK(gemm_op.run());
    compat::wait();

    if (options.verify != 0) {
      bool passed = verify(problem_size);
      std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;
      if (!passed) return cutlass::Status::kErrorInternal;
    } else {
      std::cout << "Disposition is skipped." << std::endl;
    }

    if (options.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int i = 0; i < options.iterations; ++i) {
        gemm_op.run();
      }
      compat::wait();

      float cute_time = timer.seconds() / options.iterations;
      double tflops = (2.0 * options.m * options.n * options.k * options.l) * 1e-12;
      std::cout << "Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
      printf("Cutlass GEMM+RmsScale:        [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / cute_time, cute_time*1000);
    }

    return cutlass::Status::kSuccess;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char** argv)
{
  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  ExampleRunner<Gemm> runner;

  CUTLASS_CHECK(runner.run(options, hw_info));

  return 0;
}
