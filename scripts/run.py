import argparse
import ctypes
import os
import numpy as np
import sys

# This script uses MLIR Python bindings. To run it, you must have a LLVM build
# with MLIR_ENABLE_BINDINGS_PYTHON enabled. You also need to update your
# PYTHONPATH. More information here: https://mlir.llvm.org/docs/Bindings/Python/
from mlir.execution_engine import ExecutionEngine
from mlir.ir import Module, Context, UnitAttr
from mlir.passmanager import PassManager
from mlir.runtime import get_ranked_memref_descriptor, ranked_memref_to_numpy


def _parse_network_to_llvm(ctx: Context, filename: str) -> Module:
    LINALG_TO_LLVM_PIPELINE = """builtin.module(
        func.func(
            canonicalize,
            cse
        ),
        one-shot-bufferize{bufferize-function-boundaries},
        convert-linalg-to-loops,
        convert-scf-to-cf,
        convert-cf-to-llvm,
        expand-strided-metadata,
        lower-affine,
        finalize-memref-to-llvm,
        convert-math-to-llvm,
        convert-math-to-libm,
        convert-arith-to-llvm,
        convert-index-to-llvm,
        convert-func-to-llvm,
        reconcile-unrealized-casts
    )"""

    assert os.path.exists(filename), f"{filename} is not a valid file"

    module = Module.parseFile(filename, ctx)

    entry_func = module.body.operations[0]
    entry_func.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    pm = PassManager.parse(LINALG_TO_LLVM_PIPELINE)
    pm.run(module.operation)

    return module


def _init_input(shape: tuple[int], dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.floating):
        return np.random.random(shape).astype(dtype)
    else:
        return np.random.randint(-50, 50, size=shape, dtype=dtype)


def _init_output(shape: tuple[int], dtype: np.dtype) -> np.ndarray:
    out_arr = np.zeros(shape, dtype=dtype)
    return out_arr


def _array_to_ranked_memref_ptr(array: np.ndarray):
    return ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(array)))


def _parse_io(io_desc: str) -> list[tuple[tuple[int], str]]:
    shapes_and_types = []
    for desc in io_desc.split(","):
        shape = desc.split("x")
        dtype = shape[-1]
        shape = tuple(int(dim) for dim in shape[:-1])
        shapes_and_types.append((shape, dtype))

    return shapes_and_types


def _create_io_arrays(io_desc: str, is_input: bool) -> list[np.ndarray]:
    assert io_desc, "I/O description cannot be empty"

    type_map = {
        "f32": np.float32,
        "f64": np.float64,
        "i8": np.int8,
        "i16": np.int16,
        "i32": np.int32,
        "i64": np.int64,
        "si8": np.int8,
        "si16": np.int16,
        "si32": np.int32,
        "si64": np.int64,
        "ui8": np.uint8,
        "ui16": np.uint16,
        "ui32": np.uint32,
        "ui64": np.uint64,
    }

    tensors = []
    for shape, dtype in _parse_io(io_desc):
        assert dtype in type_map, f"Unsupported data type: {dtype}"
        initializer = _init_input if is_input else _init_output
        tensors.append(initializer(shape, type_map[dtype]))

    return tensors


def _create_io_memref_ptrs(io_arrays: list[np.ndarray]):
    return [_array_to_ranked_memref_ptr(arr) for arr in io_arrays]


def _print_summary(title: str, array: np.ndarray):
    print(title)
    print("  Min:", array.min())
    print("  Max:", array.max())
    print("  Mean:", array.mean())
    print("  Variance:", array.var())


def _parse_args():
    parser = argparse.ArgumentParser(description="Utility script to MLIR networks.")
    parser.add_argument(
        "input_model", type=str, help="input MLIR model", metavar="model_path"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=str,
        help="output filename",
        default="",
        metavar="filename",
    )
    parser.add_argument(
        "--function",
        type=str,
        help="entry function for execution (default: %(default)s)",
        default="main",
    )
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--output", type=str, default="")

    return parser.parse_args()


def _main():
    llvm_build_dir = os.getenv("LLVM_BUILD_DIR")
    assert llvm_build_dir, "'LLVM_BUILD_DIR' env variable must be set"
    assert os.path.exists(llvm_build_dir), f"{llvm_build_dir} is not a valid directory"

    MLIR_SHARED_LIBS = [
        os.path.join(llvm_build_dir, "lib/libmlir_runner_utils.so"),
        os.path.join(llvm_build_dir, "lib/libmlir_c_runner_utils.so"),
    ]

    args = _parse_args()

    with Context() as ctx:
        module = _parse_network_to_llvm(ctx, args.input_model)
        engine = ExecutionEngine(module, opt_level=3, shared_libs=MLIR_SHARED_LIBS)

    input_arrays = _create_io_arrays(args.input, is_input=True)
    input_memref_ptrs = _create_io_memref_ptrs(input_arrays)
    output_arrays = _create_io_arrays(args.output, is_input=False)
    output_memref_ptrs = _create_io_memref_ptrs(output_arrays)
    engine.invoke(
        args.function,
        *output_memref_ptrs,
        *input_memref_ptrs,
    )

    print("=" * 10, "Inputs summary", "=" * 10)
    for idx, input_array in enumerate(input_arrays):
        _print_summary(f"Input {idx}", input_array)

    print("=" * 10, "Outputs summary", "=" * 10)
    for idx, output_memref_ptr in enumerate(output_memref_ptrs):
        output_array = ranked_memref_to_numpy(output_memref_ptr[0]).astype(np.float32)
        _print_summary(f"Output {idx}", output_array)

    return 0


if __name__ == "__main__":
    sys.exit(_main())
