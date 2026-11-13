<p align="center">
  <img alt="Project Banner" src="assets/images/Banner.png" width="90%" height="auto"/></br>
</p>

The goal of this project is to implement a **Value-Domain Analysis** pass for
tensor compilers using the MLIR [StableHLO](https://github.com/openxla/stablehlo) dialect. This analysis maps tensor
slices (subsets of a tensor with one fewer dimension) to their statistical
information about their values.

## Optimization Use Cases

By understanding the distribution of each slice, a compiler can potentially
perform several high-impact optimizations:

* **Quantization:** Use the estimated information to estimate the scale and zero-point
for INT8 quantization.
* **Layer Elimination:** If all slices for a tensor are strictly positive, a subsequent ReLU
layer is redundant and can be eliminated for that slice.
* **Bit-Width Optimization:** Determine if a slice can fit into a smaller data type (e.g., FP16 or INT16) without loss of precision.
* **Dead Code Elimination:** If the value range of a tensor is constant across its entire domain, downstream operations that branch on those values can be simplified.

## Dependencies
### Submodules
Vishap depends on a few third-party repositories, which are placed in the
[`external/`](external/) directory. Make sure to initialize them when cloning the
repository:

```bash
git submodule update --init
```

### Software Dependencies

These are the software requirements for building and running the code for Vishap:

| Dependency | Version   | Installation Link                                                   |
|------------|-----------|---------------------------------------------------------------------|
| Python     | >= 3.10   | [python.org](https://www.python.org/downloads/release/python-3100/) |
| CMake      | >= 3.20   | [cmake.org](https://cmake.org/install/)                             |
| Ninja      | >= 1.10   | [ninja-build GitHub](https://github.com/ninja-build/ninja/releases) |

Vishap has currently only been tested for Linux. We do not recommend using Windows.

### Python
We use Python scripts to apply quantization, and we also rely on [MLIR Python Bindings](https://mlir.llvm.org/docs/Bindings/Python/).
Make sure to install all Python dependencies before running the project. We suggest creating
a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### LLVM

Our project uses MLIR and depends on a specific LLVM commit to build all of our
dependencies, so we recommend building LLVM from source. The commit can be found at
[`build_tools/llvm_version.txt`](build_tools/llvm_version.txt).

When building LLVM, be sure to enable the MLIR project in the `LLVM_ENABLE_PROJECTS` CMake variable,
and to enable Python bindings. You should build within an environment that
has all necessary [Python dependencies](#python).

Here is a suggested build command:
```bash
git clone https://github.com/llvm/llvm-project.git
(hash="$(cat <path_to_vishap>/build_tools/llvm_version.txt)"; cd llvm-project && git fetch origin "$hash" && git checkout "$hash")
mkdir build && cd build
cmake -G Ninja ../llvm \
  -DCMAKE_INSTALL_PREFIX=. \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_RTTI=ON
ninja install
```

For more information on how to build LLVM, you can check [Prof. Fernando Pereira's video](https://www.youtube.com/watch?v=l0LI_7KeFtw) on the topic.

## Build

To build the Vishap MLIR tools, you can use the `setup.sh` script available
in the [`scripts/`](scripts/) directory. Make sure to run it from an environment
with all necessary [Python dependencies](#python), and to point it to your
LLVM installation directory:

```bash
export LLVM_BUILD_DIR=/path/to/llvm/build
./scripts/setup.sh
ninja -C build/
```

### Tests

We have a set of LIT tests ([`tests/`](tests/)) for our MLIR passes under the
`check-vishap` target. You should always run them after making modifications to
the project:

```bash
ninja -C build/ check-vishap
```

### Run
The main entry point for Vishap's MLIR passes is the `vishap-opt` binary, which
is placed in the directory where you have built Vishap. For instance, you can
run the distribution static analysis with the `--annotate-distributions` pass.
If you have the following MLIR example (remember, the analysis works over the
StableHLO dialect):

```mlir
// add.mlir
func.func @add() -> tensor<2x2xf32> {
  %0 = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %1 = stablehlo.constant dense<[[4.0, 3.0], [2.0, 1.0]]> : tensor<2x2xf32>
  %res = stablehlo.add %0, %1 : tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}
```

You can run the analysis with: `vishap-opt --annotate-distributions add.mlir`.
This will produce a version of this same graph, with the distribution
estimated by the static analysis:

```mlir
func.func @add() -> tensor<2x2xf32> attributes {vishap.distribution = [[4.0, 6.0, 5.0, 0.5]]} {
  %cst = stablehlo.constant {vishap.distribution = [[1.0, 4.0, 2.5, 1.25]]}
         dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %cst_0 = stablehlo.constant {vishap.distribution = [[1.0, 4.0, 2.5, 1.25]]}
           dense<[[4.0, 3.0], [2.0, 1.0]]> : tensor<2x2xf32>
  %0 = stablehlo.add %cst, %cst_0 {vishap.distribution = [[4.0, 6.0, 5.0, 0.5]]} : tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
```

The `vishap.distribution` attribute contains a list of 4-valued lists. Each
4-valued list is associated with the respective operation result. The 4 values
are floating-point numbers which mean, in order, the estimated minimum, maximum
mean and variance for that result.
