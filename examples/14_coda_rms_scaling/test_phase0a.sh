#!/bin/bash
# =============================================================================
# CODA Phase 0A: GEMM + Per-Row RMS Scaling (K5)
#
# Validates EVT composition workflow by building the simplest CODA kernel:
#   D[m,n] = acc[m,n] * rms_scale[m]
#
# Usage:
#   sbatch test_phase0a.sh
# =============================================================================
#SBATCH --job-name=coda_phase0a
#SBATCH --partition=bmtxg31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=coda_phase0a_%j.out

set -e

SYCL_TLA_DIR="/data/nfs_home/mspoczy/upstream/sycl-tla"
BUILD_DIR="/tmp/coda_phase0a_build_$$"
mkdir -p "$BUILD_DIR"

# Intel env (2025.3 — 2026.0 Level Zero runtime incompatible with node GPU drivers)
set +u
source /swtools/intel/compiler/2025.3/env/vars.sh
source /swtools/intel-gpu/latest/intel_gpu_vars.sh
export LD_LIBRARY_PATH="/swtools/intel/compiler/2025.3/lib:/swtools/intel/2025.3/lib:$LD_LIBRARY_PATH"
set -u

# IGC 256-GRF mode
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
export IGC_VectorAliasBBThreshold=100000000000

COMMON_FLAGS="-fsycl \
    -DCUTLASS_ENABLE_SYCL \
    -DSYCL_INTEL_TARGET \
    -I $SYCL_TLA_DIR/include \
    -I $SYCL_TLA_DIR/tools/util/include \
    -I $SYCL_TLA_DIR/examples/common \
    -I $SYCL_TLA_DIR/applications \
    -I /swtools/intel/mkl/latest/include \
    -O2 -std=c++17 \
    -fno-sycl-instrument-device-code \
    -fsycl-targets=spir64_gen \
    -Xsycl-target-backend=spir64_gen -device\ bmg-g31 \
    -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate"

echo "============================================================"
echo "CODA Phase 0A: GEMM + Per-Row RMS Scaling"
echo "============================================================"
echo "Node: $(hostname)"
echo "icpx: $(which icpx)"
icpx --version 2>&1 | head -1
echo ""

echo "=== Device diagnostics ==="
echo "--- /dev/dri ---"
ls -la /dev/dri/ 2>&1 || echo "No /dev/dri"
echo "--- sycl-ls (all devices) ---"
ONEAPI_DEVICE_SELECTOR="" sycl-ls 2>&1 || echo "(sycl-ls not available)"
echo "--- Level Zero loader ---"
ldconfig -p 2>/dev/null | grep ze_loader || echo "ze_loader not in ldconfig"
ls /swtools/intel-gpu/latest/lib/libze_loader* 2>/dev/null || echo "no ze_loader in intel-gpu"
echo "--- GPU kernel modules ---"
lsmod 2>/dev/null | grep -E "i915|xe " || echo "no GPU kernel modules"
echo ""

# ============================================================
# Compile K5: GEMM + RMS Scaling
# ============================================================
echo "Compiling CODA K5 (GEMM + RMS scaling)..."
BINARY="$BUILD_DIR/14_coda_rms_scaling"

icpx -fsycl \
    -DCUTLASS_ENABLE_SYCL \
    -DSYCL_INTEL_TARGET \
    -I "$SYCL_TLA_DIR/include" \
    -I "$SYCL_TLA_DIR/tools/util/include" \
    -I "$SYCL_TLA_DIR/examples/common" \
    -I "$SYCL_TLA_DIR/applications" \
    -I "/swtools/intel/mkl/latest/include" \
    -O2 -std=c++17 \
    -fno-sycl-instrument-device-code \
    -fsycl-targets=spir64_gen \
    -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
    -Xspirv-translator "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate" \
    -o "$BINARY" \
    "$SYCL_TLA_DIR/examples/14_coda_rms_scaling/14_coda_rms_scaling.cpp"

echo "Build succeeded: $BINARY"
ls -lh "$BINARY"
echo ""

# ============================================================
# Compile bare GEMM baseline (example 00)
# ============================================================
echo "Compiling baseline GEMM..."
BASELINE="$BUILD_DIR/00_bmg_gemm"

icpx -fsycl \
    -DCUTLASS_ENABLE_SYCL \
    -DSYCL_INTEL_TARGET \
    -I "$SYCL_TLA_DIR/include" \
    -I "$SYCL_TLA_DIR/tools/util/include" \
    -I "$SYCL_TLA_DIR/examples/common" \
    -I "$SYCL_TLA_DIR/applications" \
    -I "/swtools/intel/mkl/latest/include" \
    -O2 -std=c++17 \
    -fno-sycl-instrument-device-code \
    -fsycl-targets=spir64_gen \
    -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
    -Xspirv-translator "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate" \
    -o "$BASELINE" \
    "$SYCL_TLA_DIR/examples/00_bmg_gemm/00_bmg_gemm.cpp"

echo "Build succeeded: $BASELINE"
echo ""

# ============================================================
# Correctness verification
# ============================================================
echo "============================================================"
echo "=== Correctness: K5 with verification ==="
echo "============================================================"
"$BINARY" --m=4096 --n=4096 --k=4096 --iterations=10 --verify=1
echo ""

echo "--- Smaller shape ---"
"$BINARY" --m=1024 --n=1024 --k=1024 --iterations=10 --verify=1
echo ""

echo "--- Non-square ---"
"$BINARY" --m=2048 --n=4096 --k=1024 --iterations=10 --verify=1
echo ""

# ============================================================
# Performance: K5 vs bare GEMM
# ============================================================
echo "============================================================"
echo "=== Performance: K5 (GEMM+RMS scale) vs bare GEMM ==="
echo "============================================================"

for M in 4096 5120 8192 16384; do
    N=4096
    K=4096
    echo "--- ${M}x${N}x${K} ---"
    echo "Bare GEMM:"
    "$BASELINE" --m=$M --n=$N --k=$K --iterations=100 --verify=0
    echo "K5 (GEMM+RMS scaling):"
    "$BINARY" --m=$M --n=$N --k=$K --iterations=100 --verify=0
    echo ""
done

echo "============================================================"
echo "Phase 0A DONE"
echo "============================================================"

# Cleanup
rm -rf "$BUILD_DIR"
