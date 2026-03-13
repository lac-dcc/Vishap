# MLIR Optimization Pipeline Template

This directory contains a minimal, self-contained template for developing and
evaluating custom optimization passes on MLIR linalg programs. It compiles and
runs a baseline program, applies an optimization, then compiles and runs the
optimized version — reporting both outputs and a speedup measurement.

## Files

| File | Description |
|---|---|
| `conv2d.mlir` | Benchmark program: a large 2D convolution (~184M MACs) |
| `run_pipeline.sh` | Script that runs the full baseline vs optimized pipeline |

## Usage

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

## The Benchmark: `conv2d.mlir`

The benchmark performs a 2D convolution with:

- Input: `1 × 32 × 128 × 128` (batch × channels × height × width)
- Filter: `64 × 32 × 3 × 3` (out_channels × in_channels × kH × kW)
- Output: `1 × 64 × 126 × 126` (valid convolution, no padding)

This amounts to roughly **184 million multiply-accumulate operations**, which
is large enough to produce meaningful timing measurements beyond JIT startup
overhead. The input is filled with `1.0` and the filter with `0.01`, so the
expected output for any element is `2.88` (32 channels × 9 kernel positions ×
1.0 × 0.01). Only the first output element is printed, to avoid flooding the
terminal.

The benchmark uses `linalg.fill` and `tensor.empty` rather than large `dense`
constants, which is the idiomatic modern MLIR style and gives optimization
passes more linalg ops to reason about.

## The Pipeline: `run_pipeline.sh`

The script runs two back-to-back pipelines and compares their performance.

### Baseline pipeline

```
conv2d.mlir
    │
    ▼  mlir-opt
    │  --one-shot-bufferize
    │  --convert-linalg-to-loops      ← plain scf loops, no tiling
    │  --lower-affine
    │  --convert-scf-to-cf
    │  --convert-cf-to-llvm
    │  --convert-vector-to-llvm
    │  --convert-arith-to-llvm
    │  --convert-func-to-llvm
    │  --finalize-memref-to-llvm
    │  --reconcile-unrealized-casts
    ▼
baseline_lowered.mlir
    │
    ▼  mlir-runner
    output + timing
```

### Optimized pipeline

```
conv2d.mlir
    │
    ▼  mlir-opt
    │  --one-shot-bufferize
    │  --convert-linalg-to-affine-loops   ← affine loops (required for tiling)
    │  --affine-loop-tile="tile-size=8"   ← YOUR PASS GOES HERE
    │  --lower-affine
    │  --convert-scf-to-cf
    │  --convert-cf-to-llvm
    │  --convert-vector-to-llvm
    │  --convert-arith-to-llvm
    │  --convert-func-to-llvm
    │  --finalize-memref-to-llvm
    │  --reconcile-unrealized-casts
    ▼
optimized_lowered.mlir
    │
    ▼  mlir-runner
    output + timing
```

The two pipelines differ in exactly one place: the baseline uses
`--convert-linalg-to-loops` and no tiling, while the optimized path uses
`--convert-linalg-to-affine-loops` followed by `--affine-loop-tile`. The
rest of the lowering is identical, ensuring the performance comparison is fair.

### Why affine loops for tiling?

`--affine-loop-tile` requires affine loops because it needs to statically
reason about loop bounds in order to tile safely. Plain `scf` loops do not
carry enough information. After tiling, `--lower-affine` converts the tiled
affine loops back to scf, and the rest of the pipeline proceeds as normal.

### What tiling does

Without tiling, the loop nest iterates over the full 126×126×64 output in
one sweep, causing large amounts of cache pressure. With `tile-size=8`, the
loop nest is restructured into cache-friendly 8×8×8 chunks:

```
// Before tiling
for out_h in [0..126):
  for out_w in [0..126):
    for out_c in [0..64):
      ... accumulate ...

// After tiling (tile-size=8)
for out_h in [0..126) step 8:
  for out_w in [0..126) step 8:
    for out_c in [0..64) step 8:
      for out_h' in [0..8):       // inner tile — fits in L1/L2 cache
        for out_w' in [0..8):
          for out_c' in [0..8):
            ... accumulate ...
```

Each inner tile operates on a small working set that fits in cache, reducing
memory traffic for large tensors.

## Adapting This Template for Your Own Pass

This pipeline is designed to be a reusable template. To evaluate your own
custom pass, make two edits to `run_pipeline.sh`:

**1. Add your pass to `lower_optimized()`**, replacing or augmenting
`--affine-loop-tile`:

```bash
lower_optimized() {
  local input=$1
  local output=$2
  $MLIR_OPT "$input" \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --convert-linalg-to-affine-loops \
    --my-custom-pass \                 # <-- your pass here
    --lower-affine \
    ...
}
```

**2. Update the description echo** in Step 3 to name your pass.

Everything else — timing, output verification, speedup calculation, and
cleanup — works automatically.

### Important constraint: stay above bufferization

Passes must operate **before** bufferization (i.e., at the tensor/linalg
level) or **after** bufferization (at the memref level), but not across both.
The boundary is `--one-shot-bufferize`, which converts tensors to memrefs.
Most linalg optimization passes (fusion, tiling, padding) should run before
bufferization. Memory allocator passes typically run after.

If your pass runs before bufferization, place it before
`--one-shot-bufferize`. If it runs after, place it between
`--one-shot-bufferize` and `--convert-linalg-to-affine-loops`.

## Requirements

- LLVM/MLIR built from source (tested with LLVM 23)
- `mlir-opt` and `mlir-runner` on `PATH`
- `libmlir_runner_utils.so` and `libmlir_c_runner_utils.so` available
  (paths are set at the top of `run_pipeline.sh`)
