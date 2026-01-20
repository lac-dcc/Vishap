# Value-Domain Analysis for TOSA

The goal of this project is to implement a **Value-Domain Analysis** pass for tensor compilers using the MLIR TOSA dialect. This analysis maps tensor slices—subsets of a tensor with one fewer dimension—to their specific min-max ranges.

For example, a 3D tensor  is analyzed as a series of 2D slices. If a slice contains only zeros, as seen in the "zero-table" results, the analysis identifies this sparsity precisely.

## Optimization Use Cases

By understanding the value range of each slice, a compiler can perform several high-impact optimizations:

* **Per-Channel Quantization:** Automatically determine the scale and zero-point for INT8 quantization based on the dynamic range of individual filters (slices).
* **Layer Elimination:** If a slice's range is strictly positive (e.g., ), a subsequent ReLU layer is redundant and can be eliminated for that slice.
* **Sparsity-Aware Execution:** Identify "zero-slices" (where Min=0 and Max=0) to skip computation or use sparse-tensor storage formats.
* **Bit-Width Optimization:** Determine if a slice can fit into a smaller data type (e.g., FP16 or INT16) without loss of precision.
* **Dead Code Elimination:** If the value range of a tensor is constant across its entire domain, downstream operations that branch on those values can be simplified.

## Build

To build the VISHAP optimizer, ensure you have MLIR 22.0 configured and run:

```bash
mkdir build
cd build
cmake ..
make

```

## Run

You can run the value profiler on any TOSA-based MLIR file. For example, to analyze the 3D tensor  provided in the test suite:

```bash
./bin/vishap-opt ../tests/value_profiler/t0.mlir

```

### Example Output

The tool analyzes slices along the first dimension. For a hypothetical `3 * 2 * 2` tensor:

*  **Slice 0:** Min=0.0, Max=2.0 (Contains values like 1 and 2) 
*  **Slice 1:** Min=0.0, Max=0.0 (Zero-slice) 
*  **Slice 2:** Min=0.0, Max=4.0 (Contains values like 3 and 4) 
