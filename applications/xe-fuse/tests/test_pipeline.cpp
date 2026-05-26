
// xe-fuse pipeline comparison: Fused vs Unfused
//
// Simulates a transformer layer's GEMM+epilogue chain:
//   Phase A: GEMM1 → residual_add + gamma → rstd → GEMM2 * rstd → SwiGLU
//   Phase B: GEMM3 → residual_add + gamma → rstd → GEMM4 * rstd → RoPE
//
// Fused path uses xe-fuse kernels (K0a, K2, K3 with rstd).
// Unfused path uses bare GEMM + standalone element-wise kernels.
// Compares end-to-end latency.

#include "xe-fuse/kernels/gemm_residual_gamma.hpp"
#include "xe-fuse/kernels/gemm_rmsnorm.hpp"
#include "xe-fuse/kernels/gemm_rmsnorm_swiglu.hpp"
#include "xe-fuse/kernels/gemm_rope.hpp"
#include "xe-fuse/kernels/compute_rstd.hpp"

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

#include "sycl_common.hpp"
#include "helper.h"

#include <random>
#include <cmath>

using namespace cute;

struct Options {
  int m = 4096, n = 4096, k = 4096, l = 1;
  int iterations = 200;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("m", m, 4096);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("iterations", iterations, 200);
  }
};

// Bare GEMM (no epilogue fusion)
using BareGemmConfig = xe_fuse::GemmRmsNorm<>;  // reuse config but we'll run without epilogue
using BareGemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>,
      typename xe_fuse::GemmRmsNorm<>::CollectiveMainloop,
      typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Xe20, cutlass::arch::OpClassTensorOp,
        cute::Shape<cute::_256, cute::_256, cute::_32>,
        cute::Shape<cute::_1, cute::_1, cute::_1>,
        cutlass::epilogue::collective::EpilogueTileAuto,
        float, float,
        cutlass::bfloat16_t, cute::Stride<int64_t, cute::Int<1>, int64_t>, 8,
        cutlass::bfloat16_t, cute::Stride<int64_t, cute::Int<1>, int64_t>, 8,
        cutlass::epilogue::collective::EpilogueScheduleAuto
      >::CollectiveOp
    >>;

using ElementD = cutlass::bfloat16_t;
using StrideD = cute::Stride<int64_t, cute::Int<1>, int64_t>;

// Fused kernel types
using K0a = xe_fuse::GemmResidualGamma<>;
using K1  = xe_fuse::GemmRmsNorm<>;
using K2  = xe_fuse::GemmRmsNormSwiGLU<>;
using K3  = xe_fuse::GemmRoPE<>;

int main(int argc, const char** argv) {
  Options opts;
  opts.parse(argc, argv);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  int M = opts.m, N = opts.n, K = opts.k, L = opts.l;
  size_t mn = static_cast<size_t>(M) * N * L;
  size_t mk = static_cast<size_t>(M) * K * L;
  size_t kn = static_cast<size_t>(K) * N * L;

  using StrideA = typename K0a::Gemm::GemmKernel::StrideA;
  using StrideB = typename K0a::Gemm::GemmKernel::StrideB;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, L));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, L));
  auto stride_C = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, L));

  // Allocate buffers (reused across phases)
  cutlass::DeviceAllocation<ElementD> block_A(mk);
  cutlass::DeviceAllocation<ElementD> block_B(kn);
  cutlass::DeviceAllocation<ElementD> block_D(mn);
  cutlass::DeviceAllocation<ElementD> block_residual(mn);
  cutlass::DeviceAllocation<ElementD> block_tmp(mn);
  cutlass::DeviceAllocation<float> block_gamma(static_cast<size_t>(N) * L);
  cutlass::DeviceAllocation<float> block_scale(static_cast<size_t>(M) * L);
  cutlass::DeviceAllocation<float> block_cos_sin(mn);

  initialize_block(block_A, 2023);
  initialize_block(block_B, 2022);
  initialize_block(block_residual, 2021);

  // Initialize gamma (per-column weights)
  {
    std::vector<float> h_gamma(static_cast<size_t>(N) * L);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.5f, 1.5f);
    for (auto& v : h_gamma) v = dist(rng);
    compat::get_default_queue().memcpy(block_gamma.get(), h_gamma.data(),
                                        h_gamma.size() * sizeof(float));
  }

  // Initialize scale (per-row RMS normalization, pre-computed)
  {
    std::vector<float> h_scale(static_cast<size_t>(M) * L);
    std::mt19937 rng(43);
    std::uniform_real_distribution<float> dist(0.5f, 1.5f);
    for (auto& v : h_scale) v = v;
    for (auto& v : h_scale) v = dist(rng);
    compat::get_default_queue().memcpy(block_scale.get(), h_scale.data(),
                                        h_scale.size() * sizeof(float));
  }

  // Initialize cos_sin
  {
    std::vector<float> h_cs(mn);
    for (int batch = 0; batch < L; ++batch)
      for (int m = 0; m < M; ++m)
        for (int kp = 0; kp < N / 2; ++kp) {
          float freq = 1.0f / std::pow(10000.0f, 2.0f * kp / static_cast<float>(N));
          float angle = static_cast<float>(m) * freq;
          size_t base = static_cast<size_t>(batch) * M * N + static_cast<size_t>(m) * N;
          h_cs[base + 2 * kp] = std::cos(angle);
          h_cs[base + 2 * kp + 1] = std::sin(angle);
        }
    compat::get_default_queue().memcpy(block_cos_sin.get(), h_cs.data(), mn * sizeof(float));
  }
  compat::wait();

  auto stride_cs = cutlass::make_cute_packed_stride(K3::StrideCosSin{}, make_shape(M, N, L));

  // ============================================================
  // Setup fused kernels
  // ============================================================

  // K0a: D = gamma * (acc + residual)
  auto k0a_evt = K0a::make_evt_args(block_residual.get(), stride_D, block_gamma.get(), N);
  typename K0a::Gemm::GemmKernel::EpilogueArguments k0a_epi{
    k0a_evt, nullptr, stride_C, block_D.get(), stride_D};
  typename K0a::Gemm::GemmKernel::Arguments k0a_args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, L}, {block_A.get(), stride_A, block_B.get(), stride_B},
    k0a_epi, hw_info};
  K0a::Gemm k0a_op;
  size_t k0a_ws_size = K0a::Gemm::get_workspace_size(k0a_args);
  cutlass::device_memory::allocation<uint8_t> k0a_ws(k0a_ws_size);
  CUTLASS_CHECK(k0a_op.can_implement(k0a_args));
  CUTLASS_CHECK(k0a_op.initialize(k0a_args, k0a_ws.get()));

  // K2: D = SwiGLU(acc * R)
  auto k2_evt = K2::make_evt_args(block_scale.get(), M);
  typename K2::Gemm::GemmKernel::EpilogueArguments k2_epi{
    k2_evt, nullptr, stride_C, block_D.get(), stride_D};
  typename K2::Gemm::GemmKernel::Arguments k2_args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, L}, {block_A.get(), stride_A, block_B.get(), stride_B},
    k2_epi, hw_info};
  K2::Gemm k2_op;
  size_t k2_ws_size = K2::Gemm::get_workspace_size(k2_args);
  cutlass::device_memory::allocation<uint8_t> k2_ws(k2_ws_size);
  CUTLASS_CHECK(k2_op.can_implement(k2_args));
  CUTLASS_CHECK(k2_op.initialize(k2_args, k2_ws.get()));

  // K3: D = RoPE(acc, cos_sin)
  auto k3_evt = K3::make_evt_args(block_cos_sin.get(), stride_cs);
  typename K3::Gemm::GemmKernel::EpilogueArguments k3_epi{
    k3_evt, nullptr, stride_C, block_D.get(), stride_D};
  typename K3::Gemm::GemmKernel::Arguments k3_args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, L}, {block_A.get(), stride_A, block_B.get(), stride_B},
    k3_epi, hw_info};
  K3::Gemm k3_op;
  size_t k3_ws_size = K3::Gemm::get_workspace_size(k3_args);
  cutlass::device_memory::allocation<uint8_t> k3_ws(k3_ws_size);
  CUTLASS_CHECK(k3_op.can_implement(k3_args));
  CUTLASS_CHECK(k3_op.initialize(k3_args, k3_ws.get()));

  // Bare GEMM for unfused path
  typename BareGemm::GemmKernel::EpilogueArguments bare_epi{
    {}, nullptr, stride_C, block_D.get(), stride_D};
  typename BareGemm::GemmKernel::Arguments bare_args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, L}, {block_A.get(), stride_A, block_B.get(), stride_B},
    bare_epi, hw_info};
  BareGemm bare_op;
  size_t bare_ws_size = BareGemm::get_workspace_size(bare_args);
  cutlass::device_memory::allocation<uint8_t> bare_ws(bare_ws_size);
  CUTLASS_CHECK(bare_op.can_implement(bare_args));
  CUTLASS_CHECK(bare_op.initialize(bare_args, bare_ws.get()));

  auto q = compat::get_default_queue();

  // ============================================================
  // Standalone element-wise kernels (unfused path)
  // ============================================================
  auto standalone_residual_gamma = [&]() {
    auto* d = block_D.get();
    auto* r = block_residual.get();
    auto* g = block_gamma.get();
    int n_val = N;
    int64_t total = static_cast<int64_t>(M) * N * L;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
      int64_t i = idx[0];
      int col = static_cast<int>(i % n_val);
      float val = static_cast<float>(d[i]) + static_cast<float>(r[i]);
      d[i] = static_cast<ElementD>(val * g[col]);
    });
  };

  auto standalone_rmsnorm_scale = [&]() {
    auto* d = block_D.get();
    auto* s = block_scale.get();
    int n_val = N, m_val = M;
    int64_t total = static_cast<int64_t>(M) * N * L;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
      int64_t i = idx[0];
      int m = static_cast<int>((i / n_val) % m_val);
      int batch = static_cast<int>(i / (m_val * n_val));
      d[i] = static_cast<ElementD>(static_cast<float>(d[i]) * s[batch * m_val + m]);
    });
  };

  auto standalone_swiglu = [&]() {
    auto* d = block_D.get();
    int n_val = N;
    int64_t total = static_cast<int64_t>(M) * N * L;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
      int64_t i = idx[0];
      int col = static_cast<int>(i % n_val);
      int64_t row_base = i - col;
      int even_col = col & ~1;
      int odd_col = even_col + 1;
      if (odd_col < n_val) {
        float gate = static_cast<float>(d[row_base + even_col]);
        float up = static_cast<float>(d[row_base + odd_col]);
        float silu_gate = gate / (1.0f + sycl::exp(-gate));
        d[i] = static_cast<ElementD>(silu_gate * up);
      }
    });
  };

  auto standalone_rope = [&]() {
    auto* d = block_D.get();
    auto* t = block_tmp.get();
    auto* cs = block_cos_sin.get();
    int n_val = N;
    int64_t total = static_cast<int64_t>(M) * N * L;
    // Copy D to tmp first (need read-only source)
    q.memcpy(t, d, mn * sizeof(ElementD));
    q.wait();
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
      int64_t i = idx[0];
      int col = static_cast<int>(i % n_val);
      int64_t row_base = i - col;
      int even_col = col & ~1;
      int odd_col = even_col + 1;
      if (odd_col < n_val) {
        float x_even = static_cast<float>(t[row_base + even_col]);
        float x_odd = static_cast<float>(t[row_base + odd_col]);
        float cos_val = cs[row_base + even_col];
        float sin_val = cs[row_base + odd_col];
        float out;
        if ((col & 1) == 0)
          out = x_even * cos_val + x_odd * sin_val;
        else
          out = -x_even * sin_val + x_odd * cos_val;
        d[i] = static_cast<ElementD>(out);
      }
    });
  };

  // ============================================================
  // Warmup
  // ============================================================
  for (int i = 0; i < 5; ++i) {
    bare_op.run(); k0a_op.run(); k2_op.run(); k3_op.run();
  }
  compat::wait();

  // ============================================================
  // Benchmark: Fused pipeline
  // K0a → rstd → K2 → K0a → rstd → K3
  // ============================================================
  {
    GPU_Clock timer;
    timer.start();
    for (int i = 0; i < opts.iterations; ++i) {
      k0a_op.run();                                                          // GEMM + residual + gamma
      xe_fuse::launch_compute_rstd(q, block_D.get(), block_scale.get(), M, N, L);
      k2_op.run();                                                           // GEMM + RMSNorm + SwiGLU
      k0a_op.run();                                                          // GEMM + residual + gamma
      xe_fuse::launch_compute_rstd(q, block_D.get(), block_scale.get(), M, N, L);
      k3_op.run();                                                           // GEMM + RoPE
    }
    compat::wait();
    float time_ms = timer.seconds() * 1000.0f / opts.iterations;
    double tflops_total = (4.0 * 2.0 * M * N * K * L) * 1e-12;  // 4 GEMMs
    printf("Fused pipeline:    %7.3f ms  (4 GEMMs: %.1f TFlop/s effective)\n",
           time_ms, tflops_total / (time_ms * 1e-3));
  }

  // ============================================================
  // Benchmark: Unfused pipeline
  // bare_GEMM + residual_gamma + rstd + bare_GEMM + rmsnorm + swiglu
  // + bare_GEMM + residual_gamma + rstd + bare_GEMM + rmsnorm + rope
  // ============================================================
  {
    GPU_Clock timer;
    timer.start();
    for (int i = 0; i < opts.iterations; ++i) {
      // Phase A
      bare_op.run();                                                         // GEMM1
      standalone_residual_gamma();                                           // + residual + gamma
      q.wait();
      xe_fuse::launch_compute_rstd(q, block_D.get(), block_scale.get(), M, N, L);
      q.wait();
      bare_op.run();                                                         // GEMM2
      standalone_rmsnorm_scale();                                            // * rstd
      standalone_swiglu();                                                   // SwiGLU
      q.wait();

      // Phase B
      bare_op.run();                                                         // GEMM3
      standalone_residual_gamma();                                           // + residual + gamma
      q.wait();
      xe_fuse::launch_compute_rstd(q, block_D.get(), block_scale.get(), M, N, L);
      q.wait();
      bare_op.run();                                                         // GEMM4
      standalone_rmsnorm_scale();                                            // * rstd
      standalone_rope();                                                     // RoPE
      q.wait();
    }
    compat::wait();
    float time_ms = timer.seconds() * 1000.0f / opts.iterations;
    double tflops_total = (4.0 * 2.0 * M * N * K * L) * 1e-12;
    printf("Unfused pipeline:  %7.3f ms  (4 GEMMs: %.1f TFlop/s effective)\n",
           time_ms, tflops_total / (time_ms * 1e-3));
  }

  // ============================================================
  // Benchmark: Bare GEMMs only (lower bound)
  // ============================================================
  {
    GPU_Clock timer;
    timer.start();
    for (int i = 0; i < opts.iterations; ++i) {
      bare_op.run();
      bare_op.run();
      bare_op.run();
      bare_op.run();
    }
    compat::wait();
    float time_ms = timer.seconds() * 1000.0f / opts.iterations;
    double tflops_total = (4.0 * 2.0 * M * N * K * L) * 1e-12;
    printf("4x bare GEMM:      %7.3f ms  (%.1f TFlop/s)\n",
           time_ms, tflops_total / (time_ms * 1e-3));
  }

  // ============================================================
  // Benchmark: Standalone ops only (memory overhead)
  // ============================================================
  {
    GPU_Clock timer;
    timer.start();
    for (int i = 0; i < opts.iterations; ++i) {
      standalone_residual_gamma(); q.wait();
      xe_fuse::launch_compute_rstd(q, block_D.get(), block_scale.get(), M, N, L); q.wait();
      standalone_rmsnorm_scale(); q.wait();
      standalone_swiglu(); q.wait();
      standalone_residual_gamma(); q.wait();
      xe_fuse::launch_compute_rstd(q, block_D.get(), block_scale.get(), M, N, L); q.wait();
      standalone_rmsnorm_scale(); q.wait();
      standalone_rope(); q.wait();
    }
    compat::wait();
    float time_ms = timer.seconds() * 1000.0f / opts.iterations;
    printf("Standalone ops:    %7.3f ms  (memory-bound overhead eliminated by fusion)\n", time_ms);
  }

  printf("\n");
  double tensor_bytes = static_cast<double>(M) * N * L * sizeof(ElementD);
  printf("Tensor size: %dx%d = %.1f MB (bf16)\n", M, N, tensor_bytes / (1024.0 * 1024.0));
  printf("Each standalone op reads+writes: %.1f MB\n", 2.0 * tensor_bytes / (1024.0 * 1024.0));

  return 0;
}
