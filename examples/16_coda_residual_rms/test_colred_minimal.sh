#!/bin/bash
#SBATCH --job-name=colred_test
#SBATCH --partition=bmtxg31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --output=colred_minimal_%j.out

set -e

SYCL_TLA_DIR="/data/nfs_home/mspoczy/upstream/sycl-tla"
BUILD_DIR="/tmp/colred_test_$$"
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

echo "=== Minimal XeColReduction test ==="
echo "icpx: $(which icpx)"
icpx --version 2>&1 | head -1
echo ""

echo "Compiling minimal ColReduction test..."
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
    -o "$BUILD_DIR/colred_minimal" \
    "$SYCL_TLA_DIR/examples/16_coda_residual_rms/test_colred_minimal.cpp"

echo "Build succeeded!"
"$BUILD_DIR/colred_minimal"

rm -rf "$BUILD_DIR"
echo "=== DONE ==="
