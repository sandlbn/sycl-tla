#pragma once

// K0a_CR: gemm_residual_gamma + fused ColReduction
//
// D[m,n] = gamma[n] * (acc[m,n] + residual[m,n])
// sum_sq[m] = sum_n(D[m,n]^2) via atomic_add  (fused into epilogue)
//
// Eliminates the standalone compute_rstd kernel. After this GEMM,
// only a tiny M-element rsqrt is needed:
//   R[m] = 1/sqrt(sum_sq[m]/N + eps)
//
// Key design: residual is loaded via the C matrix (XeSrcFetch), NOT
// XeAuxLoad. This avoids the IGC bug where AuxLoad + ColReduction
// in the same kernel causes "Constraints for inline assembly" errors.
//
// EVT tree (SplitTreeVisitor):
//   InputTree:  gamma[n] * (acc + SrcFetch)       → D values
//   OutputTree: Identity(Acc)                      → store D
//   AuxOutTree: ColReduction(Acc * Acc)            → atomic_add sum_sq[m]

#include "xe-fuse/builder/epilogue_builder.hpp"

#include <sycl/sycl.hpp>

namespace xe_fuse {

template <
  typename ElementA_        = cutlass::bfloat16_t,
  typename ElementB_        = cutlass::bfloat16_t,
  typename ElementD_        = cutlass::bfloat16_t,
  typename ElementResidual_ = cutlass::bfloat16_t,
  typename ElementGamma_    = float,
  typename ElementAcc_      = float,
  typename ElementCompute_  = float,
  typename TileShape_       = cute::Shape<cute::_256, cute::_256, cute::_32>
>
struct GemmResidualGammaColRed {
  using ElementA        = ElementA_;
  using ElementB        = ElementB_;
  using ElementD        = ElementD_;
  using ElementResidual = ElementResidual_;
  using ElementGamma    = ElementGamma_;
  using ElementAcc      = ElementAcc_;
  using ElementCompute  = ElementCompute_;
  using TileShape       = TileShape_;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;

  using StrideC        = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideD        = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideResidual = cute::Stride<int64_t, cute::Int<1>, int64_t>;

  // C matrix source fetch — loads residual through the standard C-load path.
  // This avoids AuxLoad, which triggers an IGC bug when combined with ColReduction.
  using SrcFetch = cutlass::epilogue::fusion::Sm90SrcFetch<ElementResidual>;

  // ── InputTree: gamma[n] * (acc + C) ──
  // acc = raw GEMM accumulator, C = residual (loaded via SrcFetch)
  using InputTree = builder::EVT<
      builder::MulOp<ElementCompute, ElementCompute>,
      builder::RowBroadcast<0, TileShape, ElementGamma, ElementCompute>,
      builder::EVT<
          builder::AddOp<ElementCompute, ElementCompute>,
          builder::Acc,
          SrcFetch
      >
  >;

  // ── OutputTree: pass-through to D store ──
  using IdentityOp = cutlass::epilogue::fusion::XeCompute<
      cutlass::epilogue::thread::Identity,
      ElementD, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  using OutputTree = builder::EVT<IdentityOp, builder::Acc>;

  // ── AuxOutTree: ColReduction of D² per row ──
  using SquaredInput = builder::Mul<builder::Acc, builder::Acc,
                                     ElementCompute, ElementCompute>;

  using ColRed = cutlass::epilogue::fusion::XeColReduction<
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

  using ColRedTree = builder::EVT<ColRed, SquaredInput>;

  // ── Combined: SplitTree(InputTree, OutputTree, ColRedTree) ──
  using EVT = cutlass::epilogue::fusion::Sm90SplitTreeVisitor<
      InputTree, OutputTree, ColRedTree>;

  using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Xe20, cutlass::arch::OpClassTensorOp,
      TileShape,
      cute::Shape<cute::_1, cute::_1, cute::_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementResidual, StrideC, 8,
      ElementD, StrideD, 8,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EVT
    >::CollectiveOp;

  using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Xe20, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAcc,
      TileShape,
      cute::Shape<cute::_1, cute::_1, cute::_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static typename EVT::Arguments make_evt_args(
      ElementGamma const* gamma_ptr, int N,
      ElementCompute* sum_sq_ptr, int M) {

    // InputTree: Mul<RowBroadcast<gamma>, Add<Acc, SrcFetch>>
    typename builder::Acc::Arguments accum_args{};
    typename SrcFetch::Arguments src_args{};  // no arguments needed, reads from ptr_C

    typename builder::AddOp<>::Arguments add_args{};
    using AddTree = builder::EVT<builder::AddOp<ElementCompute, ElementCompute>,
                                  builder::Acc, SrcFetch>;
    typename AddTree::Arguments add_tree_args{accum_args, src_args, add_args};

    typename builder::RowBroadcast<0, TileShape, ElementGamma, ElementCompute>::Arguments gamma_args;
    gamma_args.ptr_row = gamma_ptr;
    gamma_args.null_default = ElementGamma(1);
    gamma_args.dRow = {cute::Int<0>{}, cute::Int<1>{}, static_cast<int64_t>(N)};

    typename builder::MulOp<>::Arguments mul_args{};
    typename InputTree::Arguments input_args{gamma_args, add_tree_args, mul_args};

    // OutputTree: EVT<Identity, Acc>
    typename builder::Acc::Arguments out_acc_args{};
    typename IdentityOp::Arguments identity_args{};
    typename OutputTree::Arguments output_args{out_acc_args, identity_args};

    // ColRedTree: EVT<ColRed, Mul<Acc, Acc>>
    typename builder::Acc::Arguments sq_acc1_args{};
    typename builder::Acc::Arguments sq_acc2_args{};
    typename builder::MulOp<>::Arguments sq_mul_args{};
    typename SquaredInput::Arguments sq_args{sq_acc1_args, sq_acc2_args, sq_mul_args};

    typename ColRed::Arguments col_red_args;
    col_red_args.ptr_col = sum_sq_ptr;
    col_red_args.reduction_identity = ElementCompute(0);
    col_red_args.dCol = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};

    typename ColRedTree::Arguments col_red_tree_args{sq_args, col_red_args};

    // SplitTreeVisitor args order: InputTree, AuxOutTrees..., OutputTree
    return {input_args, col_red_tree_args, output_args};
  }
};

// Tiny post-GEMM kernel: convert sum_sq → rstd.
// R[m] = 1 / sqrt(sum_sq[m] / N + eps)
template <typename ElementOutput = float>
void launch_rsqrt_from_sum_sq(
    sycl::queue& q,
    float const* sum_sq_ptr,
    ElementOutput* rstd_ptr,
    int M, int N, int L,
    float eps = 1e-6f)
{
  int total = M * L;
  q.parallel_for(
    sycl::range<1>(total),
    [=](sycl::id<1> idx) {
      int i = static_cast<int>(idx[0]);
      float mean_sq = sum_sq_ptr[i] / static_cast<float>(N);
      rstd_ptr[i] = static_cast<ElementOutput>(
          1.0f / sycl::sqrt(mean_sq + eps));
    }
  );
}

}  // namespace xe_fuse
