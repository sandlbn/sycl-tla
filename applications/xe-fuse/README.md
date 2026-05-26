# xe-fuse

GEMM epilogue fusion framework for Intel Xe GPUs, built on sycl-tla.

Fuses memory-bound Transformer operations (RMSNorm, SwiGLU, RoPE, GeLU, residual-add, etc.) into GEMM epilogues — the ops execute on data still in registers from the accumulator, avoiding separate kernel launches and global memory round-trips.

## Quick Start

### Run a full model pipeline

```bash
# Generate and benchmark a full Transformer layer for LLaMA 3 8B
sbatch tests/run_model_pipeline.sh llama3_8b 2048 100

# Or for Gemma 2 9B (uses GeGLU instead of SwiGLU)
sbatch tests/run_model_pipeline.sh gemma2_9b 2048 100
```

The pipeline generator produces model-specific C++ from architecture presets,
compiles it, and benchmarks fused vs unfused execution on real model dimensions.

### Using a preset kernel

```bash
# Generate a fused GEMM + RMSNorm + SwiGLU kernel
uv run autotune/generate_kernel.py --preset k2 -o /tmp/kernel.cpp

# Compile
icpx -fsycl -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
    -I $SYCL_TLA/include -I $SYCL_TLA/tools/util/include \
    -I $SYCL_TLA/examples/common -I $SYCL_TLA/applications \
    -I applications/xe-fuse/include \
    -O2 -std=c++17 -fsycl-targets=spir64_gen \
    -o /tmp/kernel /tmp/kernel.cpp

# Run
/tmp/kernel --m=4096 --n=4096 --k=4096 --iterations=200
```

### Available kernel presets

```
uv run autotune/generate_kernel.py --list-presets

GEMM epilogue presets:
  k1               D = acc * R[m]                         (RMSNorm)
  k0a              D = gamma[n] * (acc + residual)         (residual + weight)
  k2               D = SwiGLU(acc * R[m])                  (RMSNorm + SwiGLU)
  k2_geglu         D = GeGLU(acc * R[m])                   (RMSNorm + GeGLU)
  k3               D = RoPE(acc, cos_sin)                  (positional encoding)
  k4v2             D = RoPE(acc * R[m], cos_sin)           (RMSNorm + RoPE, merged)

Standalone presets (unfused baselines):
  sa_scale_rows     D[m,n] *= scale[m]
  sa_residual_gamma D = gamma[n] * (D + residual)
  sa_swiglu         D = SwiGLU(D)
  sa_rope_scaled    D = RoPE(D * scale[m], cos_sin)
```

### Available model presets

```
uv run autotune/generate_pipeline.py --list-presets

Available model presets:
  Name             Model                    H   H_kv      I      FFN  RoPE
  llama3_8b        LLaMA 3 8B            4096   1024  14336   swiglu  yes
  llama2_7b        LLaMA 2 7B            4096   4096  11008   swiglu  yes
  llama3_70b       LLaMA 3 70B           8192   1024  28672   swiglu  yes
  gemma2_9b        Gemma 2 9B            3584   2048  14336    geglu  yes
  gemma2_27b       Gemma 2 27B           4608   2048  36864    geglu  yes
  mistral_7b       Mistral 7B            4096   1024  14336   swiglu  yes
  qwen25_7b        Qwen 2.5 7B           3584    512  18944   swiglu  yes
  qwen25_72b       Qwen 2.5 72B          8192   1024  29568   swiglu  yes
  phi3_mini        Phi-3 Mini 3.8B       3072   3072   8192   swiglu  yes
  phi3_medium      Phi-3 Medium 14B      5120   5120  17920   swiglu  yes
```

Each preset defines H (hidden dim), H_kv (KV head dim for GQA), I (intermediate/FFN dim),
activation type, and whether RoPE is used. The pipeline generator maps these to the correct
kernel variants automatically.

### Writing a kernel in C++

```cpp
#include "xe-fuse/builder/epilogue_builder.hpp"
namespace b = xe_fuse::builder;

using TileShape = cute::Shape<cute::_256, cute::_256, cute::_32>;

// Pick your epilogue — just compose the ops you need:
using EVT = b::GeLU<b::ScaleRows<b::Acc, TileShape, float>>;   // GEMM + RMSNorm + GeLU

// Build the full GEMM kernel:
using Kernel = b::MakeGemm<EVT, bf16, bf16, bf16, float, float, TileShape>;
using Gemm = typename Kernel::Gemm;
```

## Builder API Reference

### Data Sources

| Alias | Description |
|-------|-------------|
| `Acc` | GEMM accumulator (frg_acc) |
| `AuxLoad<Element>` | Load M×N auxiliary tensor via Block2D |
| `ColBroadcast<Idx, TS, Element>` | Per-row vector: `scale[m]` broadcast across columns |
| `RowBroadcast<Idx, TS, Element>` | Per-column vector: `scale[n]` broadcast across rows |

### Binary Ops

| Alias | Formula |
|-------|---------|
| `Mul<A, B>` | `A * B` element-wise |
| `Add<A, B>` | `A + B` element-wise |
| `ScaleRows<Input, TS, E>` | `Input * scale[m]` — per-row scaling (RMSNorm) |
| `ScaleCols<Input, TS, E>` | `gamma[n] * Input` — per-column scaling |
| `AddResidual<E>` | `acc + AuxLoad(residual)` |
| `BiasAdd<TS, E>` | `acc + bias[n]` — per-column bias |

### Activation Functions

| Alias | Formula | Used By |
|-------|---------|---------|
| `GeLU<Input>` | `x * 0.5 * (1 + erf(x/√2))` | BERT, GPT-2/3/4, Gemma |
| `GeLUTanh<Input>` | tanh approximation of GeLU | Common fast variant |
| `SiLU<Input>` | `x * sigmoid(x)` (Swish) | LLaMA, Mistral |
| `ReLU<Input>` | `max(0, x)` | Older models |
| `Sigmoid<Input>` | `1 / (1 + exp(-x))` | General |

### Pairwise Ops (lane shuffle)

| Alias | Formula | Notes |
|-------|---------|-------|
| `SwiGLU<Input>` | `silu(gate) * up` on adjacent pairs | LLaMA, Mistral, Qwen |
| `GeGLU<Input>` | `gelu(gate) * up` on adjacent pairs | Gemma 2 |
| `RoPE<ECosSin>` | Rotary position embedding on acc | Direct on accumulator |
| `RoPEComposed<Input, ECosSin>` | RoPE on pre-processed input | Two-child visitor |
| `RoPEScaled<TS, EScale, ECosSin>` | Merged scale + RoPE | Flat tree, fewer dispatches |

### Patterns

| Alias | Description |
|-------|-------------|
| `DualOutput<Input, Output, AuxE>` | Split-tree: evaluate Input once, store to aux buffer, apply Output for primary D |

### Kernel Builder

```cpp
// MakeGemm<EVT, ElementA, ElementB, ElementD, ElementAcc, ElementCompute, TileShape>
using K = b::MakeGemm<EVT, bf16, bf16, bf16, float, float, TileShape>;
using Gemm = typename K::Gemm;  // ready to instantiate and run
```

## Composition Examples

```cpp
// BERT: GEMM + bias + GeLU
using EVT = b::GeLU<b::BiasAdd<TileShape, float>>;

// LLaMA / Mistral / Qwen: GEMM + RMSNorm + SwiGLU (pairwise)
using EVT = b::SwiGLU<b::ScaleRows<b::Acc, TileShape, float>>;

// Gemma 2: GEMM + RMSNorm + GeGLU (pairwise)
using EVT = b::GeGLU<b::ScaleRows<b::Acc, TileShape, float>>;

// GPT-2: GEMM + bias + GeLU_tanh
using EVT = b::GeLUTanh<b::BiasAdd<TileShape, float>>;

// K0 dual output: store raw sum for rstd + gamma-weighted primary output
using Input = b::AddResidual<bf16>;
using Output = b::ScaleCols<b::Acc, TileShape, float>;
using EVT = b::DualOutput<Input, Output, bf16>;

// Custom: GEMM + residual + RMSNorm + RoPE
using Step1 = b::AddResidual<bf16>;
using Step2 = b::ScaleRows<Step1, TileShape, float>;
using EVT = b::RoPEComposed<Step2, float>;
```

## Standalone Ops

For unfused baselines or non-GEMM use:

```cpp
#include "xe-fuse/standalone/ops.hpp"

auto q = compat::get_default_queue();
xe_fuse::standalone::scale_rows(q, data, scale, M, N, L);
xe_fuse::standalone::gelu(q, data, M, N, L);
xe_fuse::standalone::swiglu(q, data, out, M, N, L);
xe_fuse::standalone::geglu(q, data, out, M, N, L);
xe_fuse::standalone::rope_scaled(q, data, tmp, scale, cos_sin, M, N, L);
```

## Code Generation

The `autotune/` directory contains Python tooling for generating kernel and pipeline benchmarks:

| Tool | Description |
|------|-------------|
| `generate_kernel.py` | Single-kernel C++ from preset or JSON spec |
| `generate_pipeline.py` | Full model pipeline C++ from architecture preset |
| `model_presets.py` | Model architecture configs (LLaMA, Gemma, Mistral, Qwen, Phi-3) |
| `pipeline_template.cpp.j2` | Jinja2 template for pipeline benchmarks |
| `kernel_template.cpp.j2` | Jinja2 template for single-kernel benchmarks |
| `run_kernel.sh` | sbatch wrapper: compile + benchmark with structured output |

The pipeline generator maps model architectures to kernel variants:
- SwiGLU models (LLaMA, Mistral, Qwen) use K2 with `b::SwiGLU<>`
- GeGLU models (Gemma 2) use K2 with `b::GeGLU<>`
- GQA models get correct H_kv dimensions for V/K projections
- RoPE is conditionally included based on architecture

## File Layout

```
xe-fuse/
├── include/xe-fuse/
│   ├── builder/epilogue_builder.hpp    — Builder API (start here)
│   ├── visitors/
│   │   ├── xe_elementwise_compute.hpp  — Unary ops: GeLU, SiLU, ReLU, Sigmoid
│   │   ├── xe_pairwise_compute.hpp     — Pairwise: SwiGLU, GeGLU, generic pairs
│   │   └── xe_rope_compute.hpp         — RoPE: 3 visitor variants
│   ├── kernels/                        — Complete GEMM+epilogue configs
│   └── standalone/ops.hpp              — Element-wise SYCL baselines
├── autotune/
│   ├── generate_kernel.py              — Single-kernel code generator
│   ├── generate_pipeline.py            — Model pipeline code generator
│   ├── model_presets.py                — Architecture configs (10 models)
│   ├── pipeline_template.cpp.j2        — Pipeline Jinja2 template
│   ├── kernel_template.cpp.j2          — Kernel Jinja2 template
│   └── run_kernel.sh                   — sbatch compile + run wrapper
├── tests/
│   ├── run_model_pipeline.sh           — sbatch: generate → compile → test
│   └── run_llama_pipeline.sh           — Original LLaMA-specific test
```

## Performance (BMG G31, bf16, 256×256×32 tiles)

### Per-kernel overhead vs bare GEMM (~148 TFlop/s at 4096³)

| Kernel | Overhead | Spills |
|--------|----------|--------|
| GeLU | ~0% | 0 |
| K1 RMSNorm | ~8% | 22 |
| K0a Residual+Gamma | ~7% | 0 |
| K0 SplitTree (dual output) | ~9% | 0 |
| K2 RMSNorm+SwiGLU | ~9% | 15 |
| K2 RMSNorm+GeGLU | ~9% | 15 |
| K3 RoPE | ~11% | 0 |
| K4 RMSNorm+RoPE | ~15-18% | 17 |

### Model pipeline results (M=2048)

| Model | Fused (TFlop/s) | Unfused (TFlop/s) | Speedup |
|-------|-----------------|-------------------|---------|
| LLaMA 3 8B (SwiGLU) | 128 | 80 | 1.60x |
| Gemma 2 9B (GeGLU) | 108 | 74 | 1.46x |
| Qwen 2.5 7B (SwiGLU) | 123 | 74 | 1.66x |
| Phi-3 Mini 3.8B (SwiGLU) | 120 | 69 | 1.74x |

Fusion vs standalone kernels: **~4x less time** on post-GEMM ops.
Full-layer pipeline speedup: **46-74%** depending on model architecture.

Phi-3 shows the largest speedup (1.74x) due to MHA (no GQA) — all projections use the
full hidden dimension, maximizing the benefit of fused epilogues on every GEMM.
