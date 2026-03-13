# Conv2D MLIR Benchmark

This directory contains a 2D convolution benchmark implemented in MLIR's `linalg`
dialect, compiled and run via two different pipelines: **mlir-opt + mlir-runner**
and **IREE**. Both compute the same Sobel edge detection convolution on a 4×4 input,
producing a 2×2 output.

## Files

| File | Description |
|---|---|
| `conv2d_llvm.mlir` | Source for the mlir-opt + mlir-runner pipeline (uses `printMemrefF32`) |
| `conv2d_iree.mlir` | Source for the IREE pipeline (returns tensor as output) |
| `run_conv2d.sh` | Script to compile and run either pipeline |

## Usage

```bash
./run_conv2d.sh mlir   # compile and run using mlir-opt + mlir-runner
./run_conv2d.sh iree   # compile and run using iree-compile + iree-run-module
```

## Expected Output

Both pipelines should produce the same result — the Sobel filter responding
uniformly to the left-to-right gradient in the input:

```
[[[[-8, -8],
   [-8, -8]]]]
```

---

## Pipeline 1: mlir-opt + mlir-runner

This is a **manual, educational pipeline** where every lowering pass is driven
explicitly. It is the "manual transmission" of MLIR compilation — you control
exactly which transformations happen and in what order.

### Source file: `conv2d_llvm.mlir`

The source uses tensors and calls `@printMemrefF32` to print the result.
`mlir-runner` executes in-process via JIT and uses shared runtime libraries
to provide print utilities.

### Compilation

```bash
mlir-opt conv2d_llvm.mlir \
  --one-shot-bufferize="bufferize-function-boundaries" \
  --convert-linalg-to-loops \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --finalize-memref-to-llvm \
  --reconcile-unrealized-casts \
  -o conv2d_lowered.mlir
```

Each flag is an explicit transformation pass:

| Pass | What it does |
|---|---|
| `--one-shot-bufferize` | Converts tensors to memrefs (heap buffers) |
| `--convert-linalg-to-loops` | Lowers `linalg.conv_2d_nchw_fchw` to explicit `scf` loop nests |
| `--lower-affine` | Lowers affine maps to standard arithmetic |
| `--convert-scf-to-cf` | Lowers structured control flow (`scf`) to basic blocks (`cf`) |
| `--convert-cf-to-llvm` | Lowers `cf.br`, `cf.cond_br` to LLVM branches |
| `--convert-arith-to-llvm` | Lowers arithmetic ops to LLVM IR |
| `--convert-func-to-llvm` | Lowers `func.func` and `func.call` to LLVM IR |
| `--finalize-memref-to-llvm` | Lowers memref descriptors to LLVM structs/pointers |
| `--reconcile-unrealized-casts` | Cleans up any leftover type cast placeholders |

The output is a fully lowered MLIR file in the **LLVM dialect**, ready for JIT execution.

### Execution

```bash
mlir-runner conv2d_lowered.mlir \
  --entry-point-result=void \
  --shared-libs=/path/to/libmlir_runner_utils.so \
  --shared-libs=/path/to/libmlir_c_runner_utils.so
```

`mlir-runner` JIT-compiles the LLVM dialect IR in-process and executes the
`@main` function. The shared libraries provide the `printMemrefF32` symbol
used to print the result.

**Important:** `--shared-libs` takes literal file paths — it does **not** use
`LD_LIBRARY_PATH`. Always pass full paths.

### Lowering ladder

```
linalg (conv2d)
    ↓  --convert-linalg-to-loops
scf (for loops)
    ↓  --convert-scf-to-cf
cf (basic blocks + branches)
    ↓  --convert-cf-to-llvm / --convert-arith-to-llvm / --convert-func-to-llvm
LLVM dialect
    ↓  mlir-runner (JIT)
Native execution
```

---

## Pipeline 2: IREE

IREE is a **full production compiler and runtime stack** built on top of MLIR.
It is the "automatic transmission" — you hand it your model and it handles
the entire compilation pipeline internally, targeting a specific hardware backend.

### Source file: `conv2d_iree.mlir`

The source returns the output tensor directly as a function return value.
IREE's runtime (`iree-run-module`) automatically prints all return values,
so no `printMemrefF32` call is needed. Unranked tensors (`tensor<*xf32>`)
are **not supported** by IREE — all tensors must be fully ranked and statically
shaped.

### Compilation

```bash
iree-compile conv2d_iree.mlir \
  --iree-hal-target-backends=llvm-cpu \
  -o conv2d.vmfb
```

`iree-compile` runs its own multi-stage internal pipeline:

| Stage | What happens |
|---|---|
| Ingestion | Parses linalg MLIR input |
| Optimization | Applies fusion, tiling, and vectorization |
| Flow → Stream → HAL | Lowers through IREE's own dialect stack |
| Backend codegen | Generates code for the target (`llvm-cpu`, `vulkan`, `cuda`, etc.) |
| Packaging | Bundles everything into a `.vmfb` flatbuffer binary |

The output `.vmfb` is a **self-contained deployable artifact** containing
compiled code and metadata.

### Execution

```bash
iree-run-module \
  --module=conv2d.vmfb \
  --function=main
```

`iree-run-module` loads the `.vmfb` via IREE's VM runtime, which handles
memory management and scheduling. Return values are printed automatically.

### Lowering ladder

```
linalg (conv2d)
    ↓  iree-compile (internal passes)
Flow dialect
    ↓
Stream dialect
    ↓
HAL dialect (Hardware Abstraction Layer)
    ↓  backend codegen (llvm-cpu / vulkan / cuda / ...)
.vmfb (VM flatbuffer binary)
    ↓  iree-run-module
Native execution via IREE VM runtime
```

---

## Comparison

| Aspect | mlir-opt + mlir-runner | IREE |
|---|---|---|
| Pass control | Manual, explicit | Automatic |
| Output format | LLVM dialect MLIR | Binary `.vmfb` flatbuffer |
| Execution model | JIT via shared libs | VM-based runtime |
| Optimization | Minimal / manual | Full (tiling, fusion, vectorization) |
| Hardware targets | Host CPU only | CPU, GPU, accelerators |
| Unranked tensors | Supported | Not supported |
| Print mechanism | `printMemrefF32` via shared lib | Automatic from return values |
| Best for | Learning, research | Production deployment |
