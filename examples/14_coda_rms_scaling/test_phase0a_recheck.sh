#!/bin/bash
#SBATCH --job-name=coda_phase0a_recheck
#SBATCH --partition=bmtxg31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=coda_phase0a_recheck_%j.out

set -e

SYCL_TLA_DIR="/data/nfs_home/mspoczy/upstream/sycl-tla"
BUILD_DIR="/tmp/coda_phase0a_recheck_$$"
mkdir -p "$BUILD_DIR"

set +u
source /swtools/intel/compiler/2025.3/env/vars.sh
source /swtools/intel-gpu/latest/intel_gpu_vars.sh
export LD_LIBRARY_PATH="/swtools/intel/compiler/2025.3/lib:/swtools/intel/2025.3/lib:$LD_LIBRARY_PATH"
set -u

export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
export IGC_VectorAliasBBThreshold=100000000000

echo "=== Phase 0A Recheck: Reproducibility + Iter Sweep ==="
echo "Node: $(hostname)"

# Compile
BINARY="$BUILD_DIR/14_coda_rms_scaling"
BASELINE="$BUILD_DIR/00_bmg_gemm"

icpx -fsycl \
    -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
    -I "$SYCL_TLA_DIR/include" -I "$SYCL_TLA_DIR/tools/util/include" \
    -I "$SYCL_TLA_DIR/examples/common" -I "$SYCL_TLA_DIR/applications" \
    -I "/swtools/intel/mkl/latest/include" \
    -O2 -std=c++17 -fno-sycl-instrument-device-code \
    -fsycl-targets=spir64_gen \
    -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
    -Xspirv-translator "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate" \
    -o "$BINARY" "$SYCL_TLA_DIR/examples/14_coda_rms_scaling/14_coda_rms_scaling.cpp"

icpx -fsycl \
    -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
    -I "$SYCL_TLA_DIR/include" -I "$SYCL_TLA_DIR/tools/util/include" \
    -I "$SYCL_TLA_DIR/examples/common" -I "$SYCL_TLA_DIR/applications" \
    -I "/swtools/intel/mkl/latest/include" \
    -O2 -std=c++17 -fno-sycl-instrument-device-code \
    -fsycl-targets=spir64_gen \
    -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
    -Xspirv-translator "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate" \
    -o "$BASELINE" "$SYCL_TLA_DIR/examples/00_bmg_gemm/00_bmg_gemm.cpp"

echo "Build done."
echo ""

# Test 1: Multiple short runs at 4096 to check variance
echo "=== Run-to-run variance at 4096x4096x4096 ==="
for run in 1 2 3; do
    echo "--- Run $run ---"
    echo "Bare GEMM:"
    "$BASELINE" --m=4096 --n=4096 --k=4096 --iterations=50 --verify=0
    echo "K5 (RMS scaling):"
    "$BINARY" --m=4096 --n=4096 --k=4096 --iterations=50 --verify=0
done
echo ""

# Test 2: Different iteration counts to check thermal
echo "=== Iteration count sweep at 4096x4096x4096 ==="
for iters in 10 20 50 100 200; do
    echo "--- $iters iterations ---"
    echo "Bare GEMM:"
    "$BASELINE" --m=4096 --n=4096 --k=4096 --iterations=$iters --verify=0
    echo "K5:"
    "$BINARY" --m=4096 --n=4096 --k=4096 --iterations=$iters --verify=0
done
echo ""

# Test 3: All shapes with consistent 50 iterations
echo "=== All shapes (50 iterations) ==="
for M in 2048 4096 5120 8192 16384; do
    echo "--- ${M}x4096x4096 ---"
    echo "Bare:"
    "$BASELINE" --m=$M --n=4096 --k=4096 --iterations=50 --verify=0
    echo "K5:"
    "$BINARY" --m=$M --n=4096 --k=4096 --iterations=50 --verify=0
done

echo ""
echo "=== Recheck DONE ==="

rm -rf "$BUILD_DIR"
