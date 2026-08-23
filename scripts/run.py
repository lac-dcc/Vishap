import argparse
import csv
import ctypes
import logging
import numpy as np
import os

from pathlib import Path
from utils import (
    Stats,
    TMP_DIR,
    BUILD_DIR,
    annotate_distributions,
    ensure_vishap_bindings_on_path,
    get_array_stats,
    get_llvm_build_dir,
    load_input_array,
    parse_io,
)

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    ensure_vishap_bindings_on_path()
    from vishap.execution_engine import ExecutionEngine
    from vishap.ir import UnitAttr, RankedTensorType, FloatType, WalkResult
    from vishap.passmanager import PassManager
    from vishap.runtime import get_ranked_memref_descriptor, ranked_memref_to_numpy
except ImportError:
    logger.error(
        "Vishap bindings are not available. Ensure you have built Vishap with MLIR_ENABLE_BINDINGS_PYTHON."
    )
    exit(1)

VISHAP_TMP_DIR = TMP_DIR / "vishap"


def _parse_network_to_llvm(module):
    STABLEHLO_TO_LLVM_PIPELINE = """builtin.module(            \
        func.func(                                             \
            add-probe-calls,                                   \
            canonicalize,                                      \
            cse                                                \
        ),                                                     \
        stablehlo-convert-to-signless,                         \
        func.func(stablehlo-legalize-to-linalg),               \
        empty-tensor-to-alloc-tensor,                          \
        one-shot-bufferize{bufferize-function-boundaries},     \
        probe-lower-to-func-calls,                             \
        convert-linalg-to-loops,                               \
        convert-scf-to-cf,                                     \
        expand-strided-metadata,                               \
        lower-affine,                                          \
        finalize-memref-to-llvm,                               \
        convert-math-to-llvm,                                  \
        convert-math-to-libm,                                  \
        convert-arith-to-llvm,                                 \
        convert-index-to-llvm,                                 \
        convert-cf-to-llvm,                                    \
        func.func(                                             \
            llvm-request-c-wrappers                            \
        ),                                                     \
        convert-func-to-llvm,                                  \
        reconcile-unrealized-casts,                            \
        func.func(                                             \
            canonicalize,                                      \
            cse                                                \
        )                                                      \
    )"""

    PassManager.parse(STABLEHLO_TO_LLVM_PIPELINE).run(module.operation)
    entry_func = module.body.operations[0]
    entry_func.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    return module


def _init_input(shape: tuple[int], dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.floating):
        # FIXME: Add option to create input with outliers?
        return np.random.random(shape).astype(dtype)
    else:
        return np.random.randint(-50, 50, size=shape, dtype=dtype)


def _init_output(shape: tuple[int], dtype: np.dtype) -> np.ndarray:
    out_arr = np.zeros(shape, dtype=dtype)
    return out_arr


def _array_to_ranked_memref_ptr(array: np.ndarray):
    return ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(array)))


def _create_io_arrays(
    io_desc: str, is_input: bool, input_file: str | None = None
) -> list[np.ndarray]:
    assert io_desc, "I/O description cannot be empty"

    tensors = []
    for idx, (shape, dtype) in enumerate(parse_io(io_desc)):
        if is_input and input_file is not None and idx == 0:
            # FIXME: for now, only support image input for networks with
            # single inputs
            tensors.append(load_input_array(input_file, shape, dtype))
        else:
            initializer = _init_input if is_input else _init_output
            tensors.append(initializer(shape, dtype))

    return tensors


def _create_io_memref_ptrs(io_arrays: list[np.ndarray]):
    return [_array_to_ranked_memref_ptr(arr) for arr in io_arrays]


def _run_network(
    annotated_module,
    input_arrays: list[np.ndarray],
    output_arrays: list[np.ndarray],
    function: str,
) -> list[Stats]:
    llvm_build_dir = get_llvm_build_dir()
    MLIR_SHARED_LIBS = [
        str(llvm_build_dir / "lib/libmlir_runner_utils.so"),
        str(llvm_build_dir / "lib/libmlir_c_runner_utils.so"),
        str(BUILD_DIR / "lib/Runtime/libVishapRuntime.so"),
    ]

    with annotated_module.context:
        llvm_module = _parse_network_to_llvm(annotated_module)
        engine = ExecutionEngine(llvm_module, opt_level=3, shared_libs=MLIR_SHARED_LIBS)
    input_memref_ptrs = _create_io_memref_ptrs(input_arrays)
    output_memref_ptrs = _create_io_memref_ptrs(output_arrays)
    engine.invoke(
        function,
        *output_memref_ptrs,
        *input_memref_ptrs,
    )

    output_stats = [
        get_array_stats(ranked_memref_to_numpy(ptr[0]).astype(np.float32))
        for ptr in output_memref_ptrs
    ]

    return output_stats


def _compute_mape(actual: np.float64, estimate: np.float64) -> np.float64:
    """Compute Mean Absolute Percentage Error (MAPE)"""

    if not np.isfinite(actual) or not np.isfinite(estimate):
        # Return NaN error for NaN/Inf values
        return np.nan

    diff = np.abs(estimate - actual)
    threshold = 1.0e-6
    if diff <= threshold:
        # Avoid division by zero when actual value is zero
        return 0.0

    eps = 2.0e-16
    denominator = np.abs(estimate) if estimate != 0 else eps
    return diff / denominator


def _compute_ratio(actual: np.float64, estimate: np.float64) -> np.float64:
    """Compute ratio of estimate to actual value"""

    if not np.isfinite(actual) or not np.isfinite(estimate):
        # Return NaN ratio for NaN/Inf values
        return np.nan

    threshold = 1.0e-6
    if np.abs(actual - estimate) <= threshold:
        # Treat values as equals when difference is smaller than threshold.
        # This helps to avoid division by zero
        return 1.0

    return estimate / actual


def _compare_stats(
    actual: Stats, estimate: Stats
) -> tuple[np.float64, np.float64, np.float64, np.float64]:
    min_gt, max_gt, mean_gt, var_gt = actual
    min_est, max_est, mean_est, var_est = estimate
    min_mape = _compute_mape(min_gt, min_est)
    max_mape = _compute_mape(max_gt, max_est)
    mean_mape = _compute_mape(mean_gt, mean_est)
    var_ratio = _compute_ratio(var_gt, var_est)

    print(
        f"  Actual min: {min_gt}, Estimated min: {min_est}, MAPE min: {_compute_mape(min_gt, min_est)}"
    )
    print(
        f"  Actual max: {max_gt}, Estimated max: {max_est}, MAPE max: {_compute_mape(max_gt, max_est)}"
    )
    print(
        f"  Actual mean: {mean_gt}, Estimated mean: {mean_est}, MAPE mean: {_compute_mape(mean_gt, mean_est)}"
    )
    print(
        f"  Actual variance: {var_gt}, Estimated variance: {var_est}, Ratio variance: {_compute_ratio(var_gt, var_est)}"
    )

    return min_mape, max_mape, mean_mape, var_ratio


def _output_csv_results(
    out_file: str,
    results: list[tuple[str, np.float64, np.float64, np.float64, np.float64]],
):
    results_dir = TMP_DIR / "results"
    os.makedirs(results_dir, exist_ok=True)

    header = ["op_name", "min_mape", "max_mape", "mean_mape", "var_ratio"]
    with open(results_dir / out_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(results)


def _is_float_ranked_tensor_type(type_) -> bool:
    """Identify float tensors. Only float tensors are observed by probe."""
    return isinstance(type_, RankedTensorType) and isinstance(
        type_.element_type, FloatType
    )


def _collect_vishap_estimations(annotated_module) -> tuple[list[str], list[Stats]]:
    """Collect Vishap distribution estimations from MLIR module"""
    entry_func = annotated_module.body.operations[0]

    op_names = []
    estimated_dists = []
    vishap_dist_attr = "vishap.distribution"

    def collect_dists(op):
        if vishap_dist_attr not in op.attributes:
            return WalkResult.ADVANCE

        dists = list(op.operation.attributes[vishap_dist_attr])
        results = list(op.results)
        # Mirror AddProbeCalls: one observation per float ranked-tensor
        # result. Skips func.func (no results) and non-float results such
        # as the i64 constant retroactively annotated from convert.
        if len(dists) != len(results):
            return WalkResult.ADVANCE

        for dist, result in zip(dists, results):
            if not _is_float_ranked_tensor_type(result.type):
                continue
            op_names.append(op.operation.name)
            estimated_dists.append(tuple(np.float64(val) for val in dist))
        return WalkResult.ADVANCE

    entry_func.operation.walk(collect_dists)

    return op_names, estimated_dists


def _compare_results(
    op_names: list[str], estimated_dists: list[Stats], csv_filename: str
):
    # Collect true distributions from probe report
    true_dists = []
    # FIXME: Should this file name be hardcoded?
    with open("vishap_probe_report.csv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        # Ignore CSV header
        next(reader)
        for row in reader:
            # Ignore opID and resultID
            true_dists.append(tuple(np.float64(val) for val in row[2:]))

    assert len(true_dists) == len(estimated_dists), (
        "Number of estimated distributions does not match number of reported "
        f"distributions ({len(estimated_dists)} estimated vs {len(true_dists)} reported)"
    )

    results = []

    print("=" * 10, "Distribution comparison", "=" * 10)
    for idx, (op_name, actual_stats, est_stats) in enumerate(
        zip(op_names, true_dists, estimated_dists)
    ):
        print(f"{idx} - Distribution for {op_name}")
        min_mape, max_mape, mean_mape, var_ratio = _compare_stats(
            actual_stats, est_stats
        )
        results.append((op_name, min_mape, max_mape, mean_mape, var_ratio))

    _output_csv_results(csv_filename, results)


def _parse_args():
    parser = argparse.ArgumentParser(description="Utility script to MLIR networks.")
    parser.add_argument(
        "input_model", type=str, help="input MLIR model", metavar="model_path"
    )
    parser.add_argument(
        "--function",
        type=str,
        help="entry function for execution (default: %(default)s)",
        default="main",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="comma separated list of input tensor shapes and types (e.g., 1x2x3xf32,4x5xi8)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="comma separated list of output tensor shapes and types (e.g., 1x2xui8,3x4x5xf32)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="optional image or .npy path; replaces random data for the first --input tensor",
    )

    return parser.parse_args()


def _main():
    args = _parse_args()

    input_arrays = _create_io_arrays(
        args.input, is_input=True, input_file=args.input_file
    )
    output_arrays = _create_io_arrays(args.output, is_input=False)
    input_stats = [get_array_stats(arr) for arr in input_arrays]

    # Save input arrays for later use
    for idx, arr in enumerate(input_arrays):
        inputs_dir = VISHAP_TMP_DIR / "inputs"
        os.makedirs(inputs_dir, exist_ok=True)
        out_path = (
            inputs_dir / Path(args.input_model).with_suffix(f".input{idx}.npy").name
        )
        np.save(out_path, arr)

    # Run Vishap and collect estimations
    with open(args.input_model, "rb") as f:
        input_asm = f.read()
    annotated_module = annotate_distributions(input_asm, input_stats)
    op_names, estimated_dists = _collect_vishap_estimations(annotated_module)

    # Run network and collect real distributions with probe
    _run_network(annotated_module, input_arrays, output_arrays, args.function)

    # Compare and report results
    results_filename = Path(args.input_model).with_suffix(".csv").name
    _compare_results(op_names, estimated_dists, results_filename)


if __name__ == "__main__":
    _main()
