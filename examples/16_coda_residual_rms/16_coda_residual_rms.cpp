
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
    \brief CODA Phase 1: K4 + Auxiliary RMS Kernel

    GEMM epilogue (fused):
      D[m,n] = gamma[n] * (acc[m,n] + residual[m,n])

    Auxiliary kernel (separate, following CODA paper architecture):
      rms_scale[m] = 1 / sqrt( mean_n(D[m,n]^2) + eps )

    The rms_scale output feeds into K5 (Phase 0A: GEMM + per-row RMS scaling).
    This two-kernel design matches the CODA paper's architecture where the
    auxiliary reduction bridges K4 and K5.

    EVT tree for GEMM epilogue:
      XeEVT<Multiply,
        XeRowBroadcast(gamma),
        XeEVT<Plus,
          AccFetch,
          XeAuxLoad(residual)
        >
      >
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
    out << "CODA K4 + Aux RMS: GEMM epilogue fusion + auxiliary reduction\n\n"
      << "  D[m,n] = gamma[n] * (acc[m,n] + residual[m,n])\n"
      << "  rms_scale[m] = 1/sqrt(mean_n(D[m,n]^2) + eps)\n\n"
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

using ElementAccumulator = float;
using ElementCompute = float;
using ElementInputA = bfloat16_t;
using ElementInputB = bfloat16_t;
using ElementOutput = bfloat16_t;
using ElementResidual = bfloat16_t;
using ElementGamma = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using TileShape = Shape<_256, _256, _32>;

using StrideC = cute::Stride<int64_t, cute::Int<1>, int64_t>;
using StrideD = cute::Stride<int64_t, cute::Int<1>, int64_t>;

using StrideResidual = cute::Stride<int64_t, cute::Int<1>, int64_t>;

///////////////////////////////////////////////////////////////////////////////////////////////////
// EVT tree for GEMM epilogue
///////////////////////////////////////////////////////////////////////////////////////////////////

using Accum = cutlass::epilogue::fusion::XeAccFetch;

using ResidualLoad = cutlass::epilogue::fusion::XeAuxLoad<
    ElementResidual,
    StrideResidual,
    void,
    128 / cutlass::sizeof_bits_v<ElementResidual>,
    true,
    true   // UseBlock2DCopy
>;

using AddCompute = cutlass::epilogue::fusion::XeCompute<
    cutlass::plus, ElementCompute, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTAdd = cutlass::epilogue::fusion::XeEVT<
    AddCompute,
    Accum,
    ResidualLoad
>;

using GammaBroadcast = cutlass::epilogue::fusion::XeRowBroadcast<
    0, TileShape, ElementGamma, ElementCompute,
    cute::Stride<cute::Int<0>, cute::Int<1>, int64_t>,
    128 / cutlass::sizeof_bits_v<ElementGamma>
>;

using MulCompute = cutlass::epilogue::fusion::XeCompute<
    cutlass::multiplies, ElementOutput, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTK4 = cutlass::epilogue::fusion::XeEVT<
    MulCompute,
    GammaBroadcast,
    EVTAdd
>;

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
    EVTK4
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
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Auxiliary RMS reduction kernel (runs after K4 GEMM)
//
// rms_scale[m] = 1 / sqrt( sum_n(D[m,n]^2) / N + eps )
//
// Uses sub-groups for vectorized reduction across N.
///////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr float kRmsEps = 1e-6f;

void launch_rms_reduction(
    sycl::queue& q,
    ElementOutput const* d_ptr,
    ElementCompute* rms_scale_ptr,
    int M, int N, int L)
{
  constexpr int SG_SIZE = 16;
  int work_groups = M * L;
  int local_size = SG_SIZE;

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(work_groups * local_size, local_size),
      [=](sycl::nd_item<1> item) {
        int row = item.get_group(0);
        int lane = item.get_local_id(0);

        float sum_sq = 0.0f;
        for (int col = lane; col < N; col += SG_SIZE) {
          float val = static_cast<float>(d_ptr[row * N + col]);
          sum_sq += val * val;
        }

        // Sub-group reduction
        auto sg = item.get_sub_group();
        for (int offset = SG_SIZE / 2; offset > 0; offset /= 2) {
          sum_sq += sycl::shift_group_left(sg, sum_sq, offset);
        }

        if (lane == 0) {
          float mean_sq = sum_sq / static_cast<float>(N);
          rms_scale_ptr[row] = 1.0f / sycl::sqrt(mean_sq + kRmsEps);
        }
      }
    );
  });
}

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
  StrideResidual stride_residual;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementD> block_D;
  cutlass::DeviceAllocation<ElementD> block_ref_D;
  cutlass::DeviceAllocation<ElementResidual> block_residual;
  cutlass::DeviceAllocation<ElementGamma> block_gamma;
  cutlass::DeviceAllocation<ElementCompute> block_rms_scale;
  cutlass::DeviceAllocation<ElementCompute> block_ref_rms_scale;

  bool verify(const ProblemShapeType& problem_size) {
    auto [M, N, K, L] = problem_size;

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
          L, M * K, K * N, M * N, M * N
        );
    compat::wait();

    // Reference: D[m,n] = gamma[n] * (acc[m,n] + residual[m,n])
    {
      auto* ref_ptr = block_ref_D.get();
      auto* res_ptr = block_residual.get();
      auto* gamma_ptr = block_gamma.get();
      int64_t total = static_cast<int64_t>(M) * N * L;
      int n_val = N;
      int m_val = M;
      compat::get_default_queue().parallel_for(
        sycl::range<1>(total),
        [=](sycl::id<1> idx) {
          int64_t i = idx[0];
          int col = static_cast<int>(i % n_val);
          int batch = static_cast<int>(i / (m_val * n_val));
          float acc = static_cast<float>(ref_ptr[i]);
          float res = static_cast<float>(res_ptr[i]);
          float g = gamma_ptr[batch * n_val + col];
          ref_ptr[i] = static_cast<ElementD>(g * (acc + res));
        }
      );
    }
    compat::wait();

    // Reference: rms_scale[m] = 1/sqrt(mean(D[m,:]^2) + eps)
    {
      auto* ref_ptr = block_ref_D.get();
      auto* rms_ptr = block_ref_rms_scale.get();
      int n_val = N;
      float eps = kRmsEps;
      compat::get_default_queue().parallel_for(
        sycl::range<1>(static_cast<size_t>(M) * L),
        [=](sycl::id<1> idx) {
          int row = static_cast<int>(idx[0]);
          float sum_sq = 0.0f;
          for (int col = 0; col < n_val; ++col) {
            float val = static_cast<float>(ref_ptr[row * n_val + col]);
            sum_sq += val * val;
          }
          rms_ptr[row] = 1.0f / sycl::sqrt(sum_sq / static_cast<float>(n_val) + eps);
        }
      );
    }
    compat::wait();

    // Check D output
    bool d_passed = cutlass::reference::device::BlockCompareRelativelyEqual(
      block_ref_D.get(), block_D.get(), block_D.size(),
      static_cast<ElementD>(0.05f), static_cast<ElementD>(0.05f));

    // Check RMS scale
    std::vector<ElementCompute> h_rms(static_cast<size_t>(M) * L);
    std::vector<ElementCompute> h_ref_rms(static_cast<size_t>(M) * L);
    compat::get_default_queue().memcpy(h_rms.data(), block_rms_scale.get(),
                                        h_rms.size() * sizeof(ElementCompute));
    compat::get_default_queue().memcpy(h_ref_rms.data(), block_ref_rms_scale.get(),
                                        h_ref_rms.size() * sizeof(ElementCompute));
    compat::wait();

    double max_rel_err = 0.0;
    int err_count = 0;
    for (int row = 0; row < M * L; ++row) {
      if (std::abs(h_ref_rms[row]) > 1e-8f) {
        double rel_err = std::abs(h_rms[row] - h_ref_rms[row]) / std::abs(h_ref_rms[row]);
        max_rel_err = std::max(max_rel_err, rel_err);
        if (rel_err > 0.05) {
          if (err_count < 5) {
            printf("  rms_scale[%d]: kernel=%.6f ref=%.6f rel_err=%.4f\n",
                   row, h_rms[row], h_ref_rms[row], rel_err);
          }
          err_count++;
        }
      }
    }

    bool rms_passed = (err_count == 0);
    printf("D output:     %s\n", d_passed ? "Passed" : "FAILED");
    printf("RMS scale:    %s (max_rel_err=%.6f, errors=%d/%d)\n",
           rms_passed ? "Passed" : "FAILED", max_rel_err, err_count, M * L);

    return d_passed && rms_passed;
  }

  void initialize(const ProblemShapeType& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));
    stride_residual = cutlass::make_cute_packed_stride(StrideResidual{}, cute::make_shape(M, N, L));

    block_A.reset(static_cast<size_t>(M) * K * L);
    block_B.reset(static_cast<size_t>(K) * N * L);
    block_D.reset(static_cast<size_t>(M) * N * L);
    block_ref_D.reset(static_cast<size_t>(M) * N * L);
    block_residual.reset(static_cast<size_t>(M) * N * L);
    block_gamma.reset(static_cast<size_t>(N) * L);
    block_rms_scale.reset(static_cast<size_t>(M) * L);
    block_ref_rms_scale.reset(static_cast<size_t>(M) * L);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_residual, seed + 2021);

    std::vector<ElementGamma> h_gamma(static_cast<size_t>(N) * L);
    std::mt19937 rng(seed + 42);
    std::uniform_real_distribution<float> dist(0.8f, 1.2f);
    for (auto& v : h_gamma) {
      v = static_cast<ElementGamma>(dist(rng));
    }
    compat::get_default_queue().memcpy(block_gamma.get(), h_gamma.data(), h_gamma.size() * sizeof(ElementGamma));
    compat::wait();
  }

  cutlass::Status run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size);

    // EVT arguments
    typename Accum::Arguments accum_args{};

    typename ResidualLoad::Arguments residual_args;
    residual_args.ptr_aux = block_residual.get();
    residual_args.null_default = ElementResidual(0);
    residual_args.dAux = stride_residual;

    typename AddCompute::Arguments add_args{};
    typename EVTAdd::Arguments evt_add_args{accum_args, residual_args, add_args};

    typename GammaBroadcast::Arguments gamma_args;
    gamma_args.ptr_row = block_gamma.get();
    gamma_args.null_default = ElementGamma(1);
    gamma_args.dRow = {cute::Int<0>{}, cute::Int<1>{}, static_cast<int64_t>(options.n)};

    typename MulCompute::Arguments mul_args{};
    typename EVTK4::Arguments evt_args{gamma_args, evt_add_args, mul_args};

    using EpilogueArguments = typename GemmOp::GemmKernel::EpilogueArguments;
    EpilogueArguments epilogue_arguments{
      evt_args,
      nullptr,
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

    // Step 1: K4 GEMM (fused residual + gamma)
    CUTLASS_CHECK(gemm_op.run());
    compat::wait();

    // Step 2: Auxiliary RMS reduction
    auto& q = compat::get_default_queue();
    launch_rms_reduction(q, block_D.get(), block_rms_scale.get(),
                         options.m, options.n, options.l);
    compat::wait();

    if (options.verify != 0) {
      bool passed = verify(problem_size);
      std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;
      if (!passed) return cutlass::Status::kErrorInternal;
    } else {
      std::cout << "Disposition is skipped." << std::endl;
    }

    if (options.iterations > 0) {
      // Benchmark K4 alone
      GPU_Clock timer_k4;
      timer_k4.start();
      for (int i = 0; i < options.iterations; ++i) {
        gemm_op.run();
      }
      compat::wait();
      float k4_time = timer_k4.seconds() / options.iterations;

      // Benchmark aux RMS kernel alone
      GPU_Clock timer_rms;
      timer_rms.start();
      for (int i = 0; i < options.iterations; ++i) {
        launch_rms_reduction(q, block_D.get(), block_rms_scale.get(),
                             options.m, options.n, options.l);
      }
      compat::wait();
      float rms_time = timer_rms.seconds() / options.iterations;

      // Benchmark K4 + aux together
      GPU_Clock timer_total;
      timer_total.start();
      for (int i = 0; i < options.iterations; ++i) {
        gemm_op.run();
        launch_rms_reduction(q, block_D.get(), block_rms_scale.get(),
                             options.m, options.n, options.l);
      }
      compat::wait();
      float total_time = timer_total.seconds() / options.iterations;

      double tflops = (2.0 * options.m * options.n * options.k * options.l) * 1e-12;
      std::cout << "Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
      printf("K4 GEMM only:                 [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / k4_time, k4_time*1000);
      printf("Aux RMS kernel:               (%6.4f)ms\n", rms_time*1000);
      printf("K4 + Aux total:               [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / total_time, total_time*1000);
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
