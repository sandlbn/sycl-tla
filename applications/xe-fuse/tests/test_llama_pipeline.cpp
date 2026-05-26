
// xe-fuse: Full LLaMA transformer layer pipeline test
//
// Chain: compute_rstd → K4(Q) → K1(V) → [mock attention] →
//        K0(output proj + residual, dual output) → compute_rstd → K2(FFN)
//
// Each step verified against float32 reference.
// A real layer also has K4 for K — same kernel as Q, omitted for compile time.

#include "xe-fuse/kernels/gemm_rmsnorm_rope.hpp"
#include "xe-fuse/kernels/gemm_rmsnorm.hpp"
#include "xe-fuse/kernels/gemm_dual_output.hpp"
#include "xe-fuse/kernels/gemm_rmsnorm_swiglu.hpp"
#include "xe-fuse/kernels/compute_rstd.hpp"
#include "xe-fuse/standalone/ops.hpp"

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
#include <iostream>

using namespace cute;
using bf16 = cutlass::bfloat16_t;

using K4_Config = xe_fuse::GemmRmsNormRoPE<>;
using K1_Config = xe_fuse::GemmRmsNorm<>;
using K0_Config = xe_fuse::GemmDualOutput<>;
using K2_Config = xe_fuse::GemmRmsNormSwiGLU<>;

struct BareGemm {
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementD = cutlass::bfloat16_t;
  using ElementAcc = float;
  using ElementCompute = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using TileShape = cute::Shape<cute::_256, cute::_256, cute::_32>;
  using StrideC = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideD = cute::Stride<int64_t, cute::Int<1>, int64_t>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Xe20, cutlass::arch::OpClassTensorOp,
      TileShape, cute::Shape<cute::_1, cute::_1, cute::_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementD, StrideC, 8, ElementD, StrideD, 8,
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

struct Options {
  int m = 1024, h = 1024, inter = 2048, l = 1;
  int iterations = 0;
  float eps = 1e-6f;

  void parse(int argc, const char** args) {
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("m", m, 1024);
    cmd.get_cmd_line_argument("h", h, 1024);
    cmd.get_cmd_line_argument("inter", inter, 2048);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("iterations", iterations, 0);
  }
};

bool check(const char* name, bf16 const* ref, bf16 const* actual, size_t count,
           float rtol = 0.05f) {
  bool ok = cutlass::reference::device::BlockCompareRelativelyEqual(
    ref, actual, count, static_cast<bf16>(rtol), static_cast<bf16>(rtol));
  std::cout << "  " << name << ": " << (ok ? "Passed" : "FAILED") << std::endl;
  return ok;
}

int main(int argc, const char** argv) {
  Options opts;
  opts.parse(argc, argv);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  int M = opts.m, H = opts.h, I = opts.inter, L = opts.l;
  int N_ffn = 2 * I;
  auto q = compat::get_default_queue();

  std::cout << "============================================================" << std::endl;
  std::cout << "xe-fuse: LLaMA Layer Pipeline Test" << std::endl;
  std::cout << "  M=" << M << " H=" << H << " I=" << I << " (N_ffn=" << N_ffn << ")" << std::endl;
  std::cout << "============================================================" << std::endl;

  // === Allocations ===
  size_t mh = static_cast<size_t>(M) * H * L;
  size_t hh = static_cast<size_t>(H) * H * L;
  size_t h_nffn = static_cast<size_t>(H) * N_ffn * L;
  size_t m_nffn = static_cast<size_t>(M) * N_ffn * L;

  // Inputs & weights
  cutlass::DeviceAllocation<bf16> x(mh);
  cutlass::DeviceAllocation<bf16> W_qk(hh);
  cutlass::DeviceAllocation<bf16> W_v(hh);
  cutlass::DeviceAllocation<bf16> W_o(hh);
  cutlass::DeviceAllocation<bf16> W_ffn(h_nffn);
  cutlass::DeviceAllocation<float> gamma(static_cast<size_t>(H) * L);
  cutlass::DeviceAllocation<float> cos_sin(mh);

  // Fused pipeline outputs
  cutlass::DeviceAllocation<float> R1(static_cast<size_t>(M) * L);
  cutlass::DeviceAllocation<bf16> Q_out(mh);
  cutlass::DeviceAllocation<bf16> V_out(mh);
  cutlass::DeviceAllocation<bf16> residual1(mh);
  cutlass::DeviceAllocation<bf16> raw_sum(mh);
  cutlass::DeviceAllocation<float> R2(static_cast<size_t>(M) * L);
  cutlass::DeviceAllocation<bf16> ffn_out(m_nffn);

  // Reference temporaries
  size_t max_mn = std::max(mh, m_nffn);
  cutlass::DeviceAllocation<float> ref_f32(max_mn);
  cutlass::DeviceAllocation<bf16> ref_bf16(max_mn);
  cutlass::DeviceAllocation<bf16> ref_bf16_aux(mh);
  cutlass::DeviceAllocation<bf16> tmp_buf(mh);

  // Initialize
  initialize_block(x, 2023);
  initialize_block(W_qk, 2024);
  initialize_block(W_v, 2025);
  initialize_block(W_o, 2026);
  initialize_block(W_ffn, 2027);

  {
    std::vector<float> h_gamma(static_cast<size_t>(H) * L);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.5f, 1.5f);
    for (auto& v : h_gamma) v = dist(rng);
    q.memcpy(gamma.get(), h_gamma.data(), h_gamma.size() * sizeof(float));
  }
  {
    std::vector<float> h_cs(mh);
    std::mt19937 rng(2024);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_cs) v = dist(rng);
    q.memcpy(cos_sin.get(), h_cs.data(), h_cs.size() * sizeof(float));
  }
  compat::wait();

  // === Setup all GEMM ops ===

  // Common strides for [M,H] GEMM
  auto stride_A_mh = cutlass::make_cute_packed_stride(
    typename K4_Config::Gemm::GemmKernel::StrideA{}, make_shape(M, H, L));
  auto stride_B_hh = cutlass::make_cute_packed_stride(
    typename K4_Config::Gemm::GemmKernel::StrideB{}, make_shape(H, H, L));
  auto stride_D_mh = cutlass::make_cute_packed_stride(
    K4_Config::StrideD{}, make_shape(M, H, L));
  auto stride_cs = cutlass::make_cute_packed_stride(
    K4_Config::StrideCosSin{}, make_shape(M, H, L));

  // K4 — GEMM + RMSNorm + RoPE (for Q)
  typename K4_Config::Gemm k4_op;
  auto k4_evt = K4_Config::make_evt_args(R1.get(), M, cos_sin.get(), stride_cs);
  typename K4_Config::Gemm::GemmKernel::EpilogueArguments k4_epi{
    k4_evt, nullptr, stride_D_mh, Q_out.get(), stride_D_mh};
  typename K4_Config::Gemm::GemmKernel::Arguments k4_args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, H, H, L},
    {x.get(), stride_A_mh, W_qk.get(), stride_B_hh},
    k4_epi, hw_info};
  cutlass::device_memory::allocation<uint8_t> k4_ws(K4_Config::Gemm::get_workspace_size(k4_args));
  CUTLASS_CHECK(k4_op.can_implement(k4_args));
  CUTLASS_CHECK(k4_op.initialize(k4_args, k4_ws.get()));

  // K1 — GEMM + RMSNorm (for V)
  typename K1_Config::Gemm k1_op;
  auto k1_evt = K1_Config::make_evt_args(R1.get(), M);
  typename K1_Config::Gemm::GemmKernel::EpilogueArguments k1_epi{
    k1_evt, nullptr, stride_D_mh, V_out.get(), stride_D_mh};
  typename K1_Config::Gemm::GemmKernel::Arguments k1_args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, H, H, L},
    {x.get(), stride_A_mh, W_v.get(), stride_B_hh},
    k1_epi, hw_info};
  cutlass::device_memory::allocation<uint8_t> k1_ws(K1_Config::Gemm::get_workspace_size(k1_args));
  CUTLASS_CHECK(k1_op.can_implement(k1_args));
  CUTLASS_CHECK(k1_op.initialize(k1_args, k1_ws.get()));

  // K0 — GEMM + residual + dual output (output projection)
  typename K0_Config::Gemm k0_op;
  auto stride_aux = cutlass::make_cute_packed_stride(K0_Config::StrideAux{}, make_shape(M, H, L));
  auto k0_evt = K0_Config::make_evt_args(
    x.get(), stride_D_mh, gamma.get(), H, raw_sum.get(), stride_aux);
  typename K0_Config::Gemm::GemmKernel::EpilogueArguments k0_epi{
    k0_evt, nullptr, stride_D_mh, residual1.get(), stride_D_mh};
  typename K0_Config::Gemm::GemmKernel::Arguments k0_args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, H, H, L},
    {V_out.get(), stride_A_mh, W_o.get(), stride_B_hh},
    k0_epi, hw_info};
  cutlass::device_memory::allocation<uint8_t> k0_ws(K0_Config::Gemm::get_workspace_size(k0_args));
  CUTLASS_CHECK(k0_op.can_implement(k0_args));
  CUTLASS_CHECK(k0_op.initialize(k0_args, k0_ws.get()));

  // K2 — GEMM + RMSNorm + SwiGLU (FFN)
  typename K2_Config::Gemm k2_op;
  auto stride_A_mh_k2 = cutlass::make_cute_packed_stride(
    typename K2_Config::Gemm::GemmKernel::StrideA{}, make_shape(M, H, L));
  auto stride_B_ffn = cutlass::make_cute_packed_stride(
    typename K2_Config::Gemm::GemmKernel::StrideB{}, make_shape(N_ffn, H, L));
  auto stride_D_ffn = cutlass::make_cute_packed_stride(
    K2_Config::StrideD{}, make_shape(M, N_ffn, L));
  auto k2_evt = K2_Config::make_evt_args(R2.get(), M);
  typename K2_Config::Gemm::GemmKernel::EpilogueArguments k2_epi{
    k2_evt, nullptr, stride_D_ffn, ffn_out.get(), stride_D_ffn};
  typename K2_Config::Gemm::GemmKernel::Arguments k2_args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N_ffn, H, L},
    {residual1.get(), stride_A_mh_k2, W_ffn.get(), stride_B_ffn},
    k2_epi, hw_info};
  cutlass::device_memory::allocation<uint8_t> k2_ws(K2_Config::Gemm::get_workspace_size(k2_args));
  CUTLASS_CHECK(k2_op.can_implement(k2_args));
  CUTLASS_CHECK(k2_op.initialize(k2_args, k2_ws.get()));

  // === Run fused pipeline ===
  std::cout << "\n=== Running fused pipeline ===" << std::endl;

  // Step 1: compute_rstd(x) → R1
  xe_fuse::launch_compute_rstd(q, x.get(), R1.get(), M, H, L, opts.eps);
  compat::wait();

  // Step 2: Q = K4(x @ W_qk, R1, cos_sin)
  CUTLASS_CHECK(k4_op.run());

  // Step 3: V = K1(x @ W_v, R1)
  CUTLASS_CHECK(k1_op.run());
  compat::wait();

  // Step 4: mock attention (attn_out = V, already in V_out)

  // Step 5: K0(V @ W_o + x, gamma) → residual1, raw_sum
  CUTLASS_CHECK(k0_op.run());
  compat::wait();

  // Step 6: compute_rstd(raw_sum) → R2
  xe_fuse::launch_compute_rstd(q, raw_sum.get(), R2.get(), M, H, L, opts.eps);
  compat::wait();

  // Step 7: ffn = K2(residual1 @ W_ffn, R2)
  CUTLASS_CHECK(k2_op.run());
  compat::wait();

  std::cout << "Pipeline complete.\n" << std::endl;

  // === Verify each step ===
  bool all_passed = true;

  // --- Step 2: K4 reference ---
  std::cout << "--- Step 2: Q = K4(x @ W_qk, R1, cos_sin) ---" << std::endl;
  {
    cutlass::TensorRef rA(x.get(), cutlass::layout::RowMajor::packed({M, H}));
    cutlass::TensorRef rB(W_qk.get(), cutlass::layout::RowMajor::packed({H, H}));
    cutlass::TensorRef rC(ref_f32.get(), cutlass::layout::RowMajor::packed({M, H}));
    cutlass::TensorRef rD(ref_f32.get(), cutlass::layout::RowMajor::packed({M, H}));
    cutlass::reference::device::GemmComplex(
      {M, H, H}, 1.0f, rA, cutlass::ComplexTransform::kNone,
      rB, cutlass::ComplexTransform::kNone, 0.0f, rC, rD, 0.0f,
      L, M*H, H*H, M*H, M*H);
    compat::wait();

    auto* gemm = ref_f32.get();
    auto* r1 = R1.get();
    auto* cs = cos_sin.get();
    auto* out = ref_bf16.get();
    int h_val = H;
    q.parallel_for(sycl::range<1>(mh), [=](sycl::id<1> idx) {
      int64_t i = idx[0];
      int row = static_cast<int>(i / h_val);
      int col = static_cast<int>(i % h_val);
      int pair_base = (col / 2) * 2;
      int64_t even_idx = static_cast<int64_t>(row) * h_val + pair_base;
      int64_t odd_idx = even_idx + 1;

      float even_scaled = gemm[even_idx] * r1[row];
      float odd_scaled = gemm[odd_idx] * r1[row];
      float cos_v = cs[even_idx];
      float sin_v = cs[odd_idx];

      float result = (col % 2 == 0)
        ? ( even_scaled * cos_v + odd_scaled * sin_v)
        : (-even_scaled * sin_v + odd_scaled * cos_v);
      out[i] = static_cast<bf16>(result);
    });
    compat::wait();
    all_passed &= check("K4 (GEMM+RMSNorm+RoPE) → Q", ref_bf16.get(), Q_out.get(), mh);
  }

  // --- Step 3: K1 reference ---
  std::cout << "--- Step 3: V = K1(x @ W_v, R1) ---" << std::endl;
  {
    cutlass::TensorRef rA(x.get(), cutlass::layout::RowMajor::packed({M, H}));
    cutlass::TensorRef rB(W_v.get(), cutlass::layout::RowMajor::packed({H, H}));
    cutlass::TensorRef rC(ref_f32.get(), cutlass::layout::RowMajor::packed({M, H}));
    cutlass::TensorRef rD(ref_f32.get(), cutlass::layout::RowMajor::packed({M, H}));
    cutlass::reference::device::GemmComplex(
      {M, H, H}, 1.0f, rA, cutlass::ComplexTransform::kNone,
      rB, cutlass::ComplexTransform::kNone, 0.0f, rC, rD, 0.0f,
      L, M*H, H*H, M*H, M*H);
    compat::wait();

    auto* gemm = ref_f32.get();
    auto* r1 = R1.get();
    auto* out = ref_bf16.get();
    int h_val = H;
    q.parallel_for(sycl::range<1>(mh), [=](sycl::id<1> idx) {
      int64_t i = idx[0];
      int row = static_cast<int>(i / h_val);
      out[i] = static_cast<bf16>(gemm[i] * r1[row]);
    });
    compat::wait();
    all_passed &= check("K1 (GEMM+RMSNorm) → V", ref_bf16.get(), V_out.get(), mh);
  }

  // --- Step 5: K0 reference ---
  std::cout << "--- Step 5: K0(V @ W_o + x) → residual1, raw_sum ---" << std::endl;
  {
    cutlass::TensorRef rA(V_out.get(), cutlass::layout::RowMajor::packed({M, H}));
    cutlass::TensorRef rB(W_o.get(), cutlass::layout::RowMajor::packed({H, H}));
    cutlass::TensorRef rC(ref_f32.get(), cutlass::layout::RowMajor::packed({M, H}));
    cutlass::TensorRef rD(ref_f32.get(), cutlass::layout::RowMajor::packed({M, H}));
    cutlass::reference::device::GemmComplex(
      {M, H, H}, 1.0f, rA, cutlass::ComplexTransform::kNone,
      rB, cutlass::ComplexTransform::kNone, 0.0f, rC, rD, 0.0f,
      L, M*H, H*H, M*H, M*H);
    compat::wait();

    auto* gemm = ref_f32.get();
    auto* x_ptr = x.get();
    auto* gamma_ptr = gamma.get();
    auto* ref_d = ref_bf16.get();
    auto* ref_a = ref_bf16_aux.get();
    int h_val = H;
    q.parallel_for(sycl::range<1>(mh), [=](sycl::id<1> idx) {
      int64_t i = idx[0];
      int col = static_cast<int>(i % h_val);
      float sum = gemm[i] + static_cast<float>(x_ptr[i]);
      ref_a[i] = static_cast<bf16>(sum);
      ref_d[i] = static_cast<bf16>(sum * gamma_ptr[col]);
    });
    compat::wait();
    all_passed &= check("K0 primary D (gamma * sum)", ref_bf16.get(), residual1.get(), mh);
    all_passed &= check("K0 aux (raw sum)", ref_bf16_aux.get(), raw_sum.get(), mh);
  }

  // --- Step 7: K2 reference ---
  std::cout << "--- Step 7: ffn = K2(residual1 @ W_ffn, R2) ---" << std::endl;
  {
    cutlass::TensorRef rA(residual1.get(), cutlass::layout::RowMajor::packed({M, H}));
    cutlass::TensorRef rB(W_ffn.get(), cutlass::layout::RowMajor::packed({H, N_ffn}));
    cutlass::TensorRef rC(ref_f32.get(), cutlass::layout::RowMajor::packed({M, N_ffn}));
    cutlass::TensorRef rD(ref_f32.get(), cutlass::layout::RowMajor::packed({M, N_ffn}));
    cutlass::reference::device::GemmComplex(
      {M, N_ffn, H}, 1.0f, rA, cutlass::ComplexTransform::kNone,
      rB, cutlass::ComplexTransform::kNone, 0.0f, rC, rD, 0.0f,
      L, M*H, H*N_ffn, M*N_ffn, M*N_ffn);
    compat::wait();

    auto* gemm = ref_f32.get();
    auto* r2 = R2.get();
    auto* out = ref_bf16.get();
    int n_val = N_ffn;
    q.parallel_for(sycl::range<1>(m_nffn), [=](sycl::id<1> idx) {
      int64_t i = idx[0];
      int row = static_cast<int>(i / n_val);
      int col = static_cast<int>(i % n_val);
      int pair_base = (col / 2) * 2;
      int64_t gate_idx = static_cast<int64_t>(row) * n_val + pair_base;
      int64_t up_idx = gate_idx + 1;

      float gate = gemm[gate_idx] * r2[row];
      float up = gemm[up_idx] * r2[row];
      float silu_gate = gate / (1.0f + sycl::exp(-gate));
      out[i] = static_cast<bf16>(silu_gate * up);
    });
    compat::wait();
    all_passed &= check("K2 (GEMM+RMSNorm+SwiGLU) → ffn", ref_bf16.get(), ffn_out.get(), m_nffn);
  }

  // === Summary ===
  std::cout << "\n============================================================" << std::endl;
  std::cout << "Pipeline: " << (all_passed ? "ALL PASSED" : "SOME FAILED") << std::endl;
  std::cout << "============================================================" << std::endl;

  // === Benchmark ===
  if (opts.iterations > 0) {
    std::cout << "\n=== Benchmark: " << opts.iterations << " iterations ===" << std::endl;
    std::cout << "Pipeline: " << M << "x" << H << " hidden, " << I << " intermediate\n" << std::endl;

    double flops = 2.0 * M * (3.0 * H * H + static_cast<double>(N_ffn) * H) * L;

    // --- Fused pipeline ---
    xe_fuse::launch_compute_rstd(q, x.get(), R1.get(), M, H, L, opts.eps);
    k4_op.run(); k1_op.run(); k0_op.run();
    xe_fuse::launch_compute_rstd(q, raw_sum.get(), R2.get(), M, H, L, opts.eps);
    k2_op.run();
    compat::wait();

    GPU_Clock timer;
    timer.start();
    for (int i = 0; i < opts.iterations; ++i) {
      xe_fuse::launch_compute_rstd(q, x.get(), R1.get(), M, H, L, opts.eps);
      k4_op.run(); k1_op.run(); k0_op.run();
      xe_fuse::launch_compute_rstd(q, raw_sum.get(), R2.get(), M, H, L, opts.eps);
      k2_op.run();
    }
    compat::wait();
    float fused_s = timer.seconds() / opts.iterations;

    printf("Fused pipeline (GEMM + epilogue):\n");
    printf("  Time: %.4f ms\n", fused_s * 1000);
    printf("  Aggregate: %.3f TFlop/s\n", flops * 1e-12 / fused_s);

    // --- Unfused pipeline (bare GEMM + standalone ops) ---
    using BareOp = typename BareGemm::Gemm;
    using BareKernel = typename BareGemm::Gemm::GemmKernel;

    auto bare_sA_mh = cutlass::make_cute_packed_stride(
      typename BareKernel::StrideA{}, make_shape(M, H, L));
    auto bare_sB_hh = cutlass::make_cute_packed_stride(
      typename BareKernel::StrideB{}, make_shape(H, H, L));
    auto bare_sD_mh = cutlass::make_cute_packed_stride(
      BareGemm::StrideD{}, make_shape(M, H, L));
    auto bare_sB_ffn = cutlass::make_cute_packed_stride(
      typename BareKernel::StrideB{}, make_shape(N_ffn, H, L));
    auto bare_sD_ffn = cutlass::make_cute_packed_stride(
      BareGemm::StrideD{}, make_shape(M, N_ffn, L));

    typename BareKernel::Arguments args_qk{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, H, H, L},
      {x.get(), bare_sA_mh, W_qk.get(), bare_sB_hh},
      {{1.0f, 0.0f}, nullptr, bare_sD_mh, tmp_buf.get(), bare_sD_mh},
      hw_info};
    typename BareKernel::Arguments args_v{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, H, H, L},
      {x.get(), bare_sA_mh, W_v.get(), bare_sB_hh},
      {{1.0f, 0.0f}, nullptr, bare_sD_mh, V_out.get(), bare_sD_mh},
      hw_info};
    typename BareKernel::Arguments args_o{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, H, H, L},
      {V_out.get(), bare_sA_mh, W_o.get(), bare_sB_hh},
      {{1.0f, 0.0f}, nullptr, bare_sD_mh, residual1.get(), bare_sD_mh},
      hw_info};
    typename BareKernel::Arguments args_ffn{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N_ffn, H, L},
      {residual1.get(), bare_sA_mh, W_ffn.get(), bare_sB_ffn},
      {{1.0f, 0.0f}, nullptr, bare_sD_ffn, ffn_out.get(), bare_sD_ffn},
      hw_info};

    BareOp bare_qk, bare_v, bare_o, bare_ffn;
    cutlass::device_memory::allocation<uint8_t> ws_qk(BareOp::get_workspace_size(args_qk));
    cutlass::device_memory::allocation<uint8_t> ws_v(BareOp::get_workspace_size(args_v));
    cutlass::device_memory::allocation<uint8_t> ws_o(BareOp::get_workspace_size(args_o));
    cutlass::device_memory::allocation<uint8_t> ws_ffn(BareOp::get_workspace_size(args_ffn));
    CUTLASS_CHECK(bare_qk.can_implement(args_qk));
    CUTLASS_CHECK(bare_qk.initialize(args_qk, ws_qk.get()));
    CUTLASS_CHECK(bare_v.can_implement(args_v));
    CUTLASS_CHECK(bare_v.initialize(args_v, ws_v.get()));
    CUTLASS_CHECK(bare_o.can_implement(args_o));
    CUTLASS_CHECK(bare_o.initialize(args_o, ws_o.get()));
    CUTLASS_CHECK(bare_ffn.can_implement(args_ffn));
    CUTLASS_CHECK(bare_ffn.initialize(args_ffn, ws_ffn.get()));

    size_t mh_bytes = mh * sizeof(bf16);

    // Warm-up unfused
    xe_fuse::launch_compute_rstd(q, x.get(), R1.get(), M, H, L, opts.eps);
    bare_qk.run();
    xe_fuse::standalone::scale_rows(q, tmp_buf.get(), R1.get(), M, H, L);
    xe_fuse::standalone::rope(q, Q_out.get(), tmp_buf.get(), cos_sin.get(), M, H, L);
    bare_v.run();
    xe_fuse::standalone::scale_rows(q, V_out.get(), R1.get(), M, H, L);
    bare_o.run();
    xe_fuse::standalone::add_residual(q, residual1.get(), x.get(), M, H, L);
    q.memcpy(raw_sum.get(), residual1.get(), mh_bytes);
    xe_fuse::standalone::scale_cols(q, residual1.get(), gamma.get(), M, H, L);
    xe_fuse::launch_compute_rstd(q, raw_sum.get(), R2.get(), M, H, L, opts.eps);
    bare_ffn.run();
    xe_fuse::standalone::scale_rows(q, ffn_out.get(), R2.get(), M, N_ffn, L);
    xe_fuse::standalone::swiglu(q, ffn_out.get(), M, N_ffn, L);
    compat::wait();

    timer.start();
    for (int i = 0; i < opts.iterations; ++i) {
      xe_fuse::launch_compute_rstd(q, x.get(), R1.get(), M, H, L, opts.eps);
      bare_qk.run();
      xe_fuse::standalone::scale_rows(q, tmp_buf.get(), R1.get(), M, H, L);
      xe_fuse::standalone::rope(q, Q_out.get(), tmp_buf.get(), cos_sin.get(), M, H, L);
      bare_v.run();
      xe_fuse::standalone::scale_rows(q, V_out.get(), R1.get(), M, H, L);
      bare_o.run();
      xe_fuse::standalone::add_residual(q, residual1.get(), x.get(), M, H, L);
      q.memcpy(raw_sum.get(), residual1.get(), mh_bytes);
      xe_fuse::standalone::scale_cols(q, residual1.get(), gamma.get(), M, H, L);
      xe_fuse::launch_compute_rstd(q, raw_sum.get(), R2.get(), M, H, L, opts.eps);
      bare_ffn.run();
      xe_fuse::standalone::scale_rows(q, ffn_out.get(), R2.get(), M, N_ffn, L);
      xe_fuse::standalone::swiglu(q, ffn_out.get(), M, N_ffn, L);
    }
    compat::wait();
    float unfused_s = timer.seconds() / opts.iterations;

    printf("\nUnfused pipeline (bare GEMM + standalone ops):\n");
    printf("  Time: %.4f ms\n", unfused_s * 1000);
    printf("  Aggregate: %.3f TFlop/s\n", flops * 1e-12 / unfused_s);

    float speedup = unfused_s / fused_s;
    float savings = (1.0f - fused_s / unfused_s) * 100.0f;
    printf("\nFusion speedup: %.2fx (fused saves %.1f%% of pipeline time)\n", speedup, savings);
    std::cout << "Note: real layer has 2x K4 (Q+K). Shown: 1x K4 + 1x K1 + 1x K0 + 1x K2" << std::endl;
  }

  return all_passed ? 0 : 1;
}
