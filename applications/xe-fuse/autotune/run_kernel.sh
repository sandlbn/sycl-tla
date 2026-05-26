#!/bin/bash
# =============================================================================
# xe-fuse autotune: compile and benchmark a generated kernel
#
# Usage (standalone):
#   sbatch run_kernel.sh <source.cpp> [extra_args...]
#
# Usage (from runner.py):
#   Called automatically with generated source path
#
# Output is structured for machine parsing:
#   BUILD: OK|FAILED
#   SPILLS: <count>|none
#   <kernel_name>: [<tflops>]TFlop/s  (<time>)ms
# =============================================================================
#SBATCH --job-name=xe_fuse_autotune
#SBATCH --partition=bmtxg31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=autotune_%j.out

set -e

KERNEL_SRC="${1:?Usage: sbatch run_kernel.sh <source.cpp> [args...]}"
shift
EXTRA_ARGS="$@"

SYCL_TLA_DIR="/data/nfs_home/mspoczy/upstream/sycl-tla"
XE_FUSE_DIR="$SYCL_TLA_DIR/applications/xe-fuse"
BUILD_DIR="/tmp/xe_fuse_autotune_$$"
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

COMMON_FLAGS="-fsycl \
    -DCUTLASS_ENABLE_SYCL \
    -DSYCL_INTEL_TARGET \
    -I $SYCL_TLA_DIR/include \
    -I $SYCL_TLA_DIR/tools/util/include \
    -I $SYCL_TLA_DIR/examples/common \
    -I $SYCL_TLA_DIR/applications \
    -I $XE_FUSE_DIR/include \
    -I /swtools/intel/mkl/latest/include \
    -O2 -std=c++17 \
    -fno-sycl-instrument-device-code \
    -fsycl-targets=spir64_gen \
    -Xsycl-target-backend=spir64_gen \"-device bmg-g31\" \
    -Xspirv-translator \"-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate\""

BINARY="$BUILD_DIR/kernel"

echo "=== xe-fuse autotune ==="
echo "Source: $KERNEL_SRC"
echo "Node: $(hostname)"
echo ""

# Build
echo "--- Compiling ---"
eval icpx $COMMON_FLAGS -o "$BINARY" "$KERNEL_SRC" > "$BUILD_DIR/build.log" 2>&1
BUILD_RC=$?

if [ $BUILD_RC -ne 0 ]; then
    echo "BUILD: FAILED"
    cat "$BUILD_DIR/build.log"
    rm -rf "$BUILD_DIR"
    exit 1
fi

echo "BUILD: OK"
SPILLS=$(grep -o "spilled around [0-9]*" "$BUILD_DIR/build.log" 2>/dev/null | head -1 || true)
if [ -n "$SPILLS" ]; then
    echo "SPILLS: $(echo $SPILLS | grep -o '[0-9]*')"
else
    echo "SPILLS: none"
fi
echo ""

# Run benchmark at multiple sizes
for SIZE in "4096 4096 4096" "8192 4096 4096" "16384 4096 4096"; do
    read M N K <<< "$SIZE"
    echo "--- ${M}x${N}x${K} ---"
    "$BINARY" --m=$M --n=$N --k=$K --iterations=200 --verify=0 $EXTRA_ARGS
    echo ""
done

echo "=== autotune DONE ==="

rm -rf "$BUILD_DIR"
