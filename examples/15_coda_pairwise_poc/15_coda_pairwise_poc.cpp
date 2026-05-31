
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
    \brief CODA Phase 0B: Pairwise Shuffle Feasibility PoC

    Tests whether sub-group shuffles in the GEMM epilogue are performant
    enough for CODA's pairwise operations (RoPE, SwiGLU).

    Implements D[m,n] = acc[m, n ^ 1]  (pairwise column swap via shuffle_xor)

    Xe DPAS CLayout: Layout<Shape<_16, _M>, Stride<_M, _1>>
    Each of 16 sub-group lanes owns one N column. shuffle_xor(val, 1)
    exchanges between lanes (2k, 2k+1), which are N-adjacent column pairs.
*/

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gpu_generics.h"
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
    out << "CODA Phase 0B: Pairwise Shuffle PoC\n\n"
      << "  D[m,n] = acc[m, n^1]  (pairwise column swap)\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the L extent (batch count) of the GEMM\n"
      << "  --iterations=<int>          Iterations\n"
      << "  --verify=<int>              Specify whether to verify.\n\n";
    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Custom EVT visitor: Pairwise Swap via sub-group shuffle_xor
//
// Takes one child (AccFetch) and swaps N-adjacent column pairs.
// Lane k gets lane (k^1)'s value via shfl_xor_sync(mask, val, 1).
//
///////////////////////////////////////////////////////////////////////////////////////////////////

struct PairwiseSwap : cutlass::epilogue::fusion::Sm90VisitorImpl<> {

  using Sm90VisitorImpl<>::Sm90VisitorImpl;

  struct ConsumerStoreCallbacks : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE cutlass::Array<ElementInput, FragmentSize>
    visit(cutlass::Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          cutlass::Array<ElementInput, FragmentSize> const& frg_input) {
      cutlass::Array<ElementInput, FragmentSize> result;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        // Reinterpret as uint32_t for shuffle (works for float and bf16-widened-to-float)
        uint32_t my_bits = reinterpret_cast<const uint32_t&>(frg_input[i]);
        uint32_t partner_bits = shfl_xor_sync(0xFFFFFFFF, my_bits, 1, 16);
        result[i] = reinterpret_cast<const ElementInput&>(partner_bits);
      }

      return result;
    }
  };

  template <
    bool ReferenceSrc,
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(cutlass::epilogue::fusion::ConsumerStoreArgs<Args...> const& args) {
    return ConsumerStoreCallbacks{};
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

using ElementAccumulator = float;
using ElementCompute = float;
using ElementInputA = bfloat16_t;
using ElementInputB = bfloat16_t;
using ElementOutput = bfloat16_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using TileShape = Shape<_256, _256, _32>;

using StrideC = cute::Stride<int64_t, cute::Int<1>, int64_t>;
using StrideD = cute::Stride<int64_t, cute::Int<1>, int64_t>;

// EVT tree: D[m,n] = acc[m, n^1]  (pairwise swap via shuffle)
//
// XeEVT<PairwiseSwap,
//   AccFetch          // child 0: accumulator
// >

using Accum = cutlass::epilogue::fusion::XeAccFetch;

using EVTPairwiseSwap = cutlass::epilogue::fusion::XeEVT<
    PairwiseSwap,
    Accum
>;

// Build epilogue
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
    EVTPairwiseSwap
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

  bool verify(const ProblemShapeType& problem_size) {
    auto [M, N, K, L] = problem_size;

    // Step 1: compute reference GEMM into ref_D
    cutlass::TensorRef ref_A(block_A.get(), cutlass::layout::RowMajor::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), cutlass::layout::RowMajor::packed({K, N}));
    cutlass::TensorRef ref_C(block_ref_D.get(), cutlass::layout::RowMajor::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), cutlass::layout::RowMajor::packed({M, N}));

    cutlass::reference::device::GemmComplex(
          {M, N, K},
          ElementAccumulator(1),
          ref_A,
          cutlass::ComplexTransform::kNone,
          ref_B,
          cutlass::ComplexTransform::kNone,
          ElementAccumulator(0),
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

    // Step 2: pairwise swap columns on device: ref_D[m,n] = gemm[m, n^1]
    {
      auto* ref_ptr = block_ref_D.get();
      int64_t total = static_cast<int64_t>(M) * N * L;
      int n_val = N;
      int m_val = M;
      compat::get_default_queue().parallel_for(
        sycl::range<1>(total),
        [=](sycl::id<1> idx) {
          int64_t i = idx[0];
          int batch = static_cast<int>(i / (m_val * n_val));
          int row = static_cast<int>((i / n_val) % m_val);
          int col = static_cast<int>(i % n_val);
          int partner_col = col ^ 1;
          if (partner_col < n_val) {
            int64_t partner_idx = static_cast<int64_t>(batch) * m_val * n_val +
                                  static_cast<int64_t>(row) * n_val + partner_col;
            // Read from partner column (swap) — need to use a temp to avoid race
            // Since (col, col^1) are symmetric, we can just read the GEMM result at partner
            // But ref_D was already overwritten... we need a copy.
            // Actually, the swap is symmetric: ref_D[m,2k] and ref_D[m,2k+1] are swapped.
            // Since both lanes read from the original and write to the same buffer, we have a race.
            // Fix: do the swap in two steps or use a separate buffer.
          }
        }
      );
    }

    // The in-place swap above has a race condition. Instead, allocate a temp buffer.
    cutlass::DeviceAllocation<ElementD> block_gemm_D;
    block_gemm_D.reset(static_cast<size_t>(M) * N * L);

    // Redo: compute GEMM into block_gemm_D, then swap into block_ref_D
    {
      cutlass::TensorRef ref_D2(block_gemm_D.get(), cutlass::layout::RowMajor::packed({M, N}));
      cutlass::reference::device::GemmComplex(
            {M, N, K},
            ElementAccumulator(1),
            ref_A,
            cutlass::ComplexTransform::kNone,
            ref_B,
            cutlass::ComplexTransform::kNone,
            ElementAccumulator(0),
            ref_C,
            ref_D2,
            ElementAccumulator(0),
            L,
            M * K,
            K * N,
            M * N,
            M * N
          );
    }
    compat::wait();

    // Swap columns pairwise: ref_D[m,n] = gemm_D[m, n^1]
    {
      auto* dst_ptr = block_ref_D.get();
      auto* src_ptr = block_gemm_D.get();
      int64_t total = static_cast<int64_t>(M) * N * L;
      int n_val = N;
      int m_val = M;
      compat::get_default_queue().parallel_for(
        sycl::range<1>(total),
        [=](sycl::id<1> idx) {
          int64_t i = idx[0];
          int col = static_cast<int>(i % n_val);
          int partner_col = col ^ 1;
          if (partner_col < n_val) {
            int64_t base = i - col;
            dst_ptr[i] = src_ptr[base + partner_col];
          } else {
            dst_ptr[i] = src_ptr[i];
          }
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

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
  }

  cutlass::Status run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size);

    // EVT arguments: XeEVT<PairwiseSwap, Accum>
    // Nesting: { child0_args (Accum), node_args (PairwiseSwap) }
    typename Accum::Arguments accum_args{};
    typename PairwiseSwap::Arguments swap_args{};

    typename EVTPairwiseSwap::Arguments evt_args{
        accum_args,     // child 0: Accum
        swap_args       // node: PairwiseSwap
    };

    using EpilogueArguments = typename GemmOp::GemmKernel::EpilogueArguments;
    EpilogueArguments epilogue_arguments{
      evt_args,
      nullptr,   // C ptr (unused)
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
      printf("Cutlass GEMM+PairwiseSwap:    [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / cute_time, cute_time*1000);
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
