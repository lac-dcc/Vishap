import argparse
import csv
import logging
import numpy as np
import os
import warnings

from pathlib import Path
from mlir_runtime import (
    io_arrays_from_entry_func,
    parse_probe_report_observations,
    run_probed_network,
    is_float_ranked_tensor_type,
)
from utils import (
    Stats,
    TMP_DIR,
    annotate_distributions,
    collect_input_paths,
    ensure_vishap_bindings_on_path,
    get_array_stats,
    parse_vishap_module,
)

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    ensure_vishap_bindings_on_path()
    from vishap.ir import WalkResult
except ImportError:
    logger.error(
        "Vishap bindings are not available. Ensure you have built Vishap with MLIR_ENABLE_BINDINGS_PYTHON."
    )
    exit(1)

VISHAP_TMP_DIR = TMP_DIR / "vishap"


def _init_input(shape: tuple[int], dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.floating):
        # FIXME: Add option to create input with outliers?
        return np.random.random(shape).astype(dtype)
    else:
        return np.random.randint(-50, 50, size=shape, dtype=dtype)


def _run_network(
    annotated_module,
    input_arrays: list[np.ndarray],
    output_arrays: list[np.ndarray],
    report_file: Path,
) -> list[Stats]:
    return run_probed_network(
        annotated_module,
        input_arrays,
        output_arrays,
        all_float_ops=False,
        report_file=report_file,
    )


# Fraction of the observed tensor std used to floor the std log-ratio. Keeps
# the ratio finite for zero-variance slices (dead channels, padding) while
# still reporting a large miss when the estimate is far from zero.
STD_FLOOR_FRACTION = 1.0e-3


class Scale:
    """Normalization constants of one observed tensor.

    Every row of an observation (tensor-level and per-slice) is normalized by
    the *tensor-level* observed statistics, so slice errors are comparable
    across slices and stay well-defined for degenerate slices (constant
    values, zero variance).

    `range` is `max - min` of the observed tensor, falling back to its
    magnitude for constant tensors; NaN for an all-zero tensor. `std` is
    the observed tensor std, falling back to `range` when zero.
    """

    def __init__(self, summary: Stats):
        min_actual, max_actual, _, var_actual = summary
        span = max_actual - min_actual
        if np.isfinite(span) and span > 0:
            self.range = np.float64(span)
        else:
            magnitude = max(np.abs(min_actual), np.abs(max_actual))
            self.range = np.float64(magnitude) if magnitude > 0 else np.float64(np.nan)

        std = np.sqrt(var_actual) if np.isfinite(var_actual) and var_actual > 0 else 0.0
        self.std = np.float64(std) if std > 0 else self.range
        self.std_floor = STD_FLOOR_FRACTION * self.std

    def is_degenerate(self) -> bool:
        return not np.isfinite(self.range)


def _normalized_error(
    actual: np.float64, estimate: np.float64, scale: np.float64
) -> np.float64:
    """Signed ``(estimate - actual) / scale``; NaN for non-finite inputs."""
    if not np.isfinite(actual) or not np.isfinite(estimate) or not np.isfinite(scale):
        return np.float64(np.nan)
    return (estimate - actual) / scale


def _std_log2_ratio(
    var_actual: np.float64, var_est: np.float64, floor: np.float64
) -> np.float64:
    """`log2((std_est + floor) / (std_actual + floor))`.

    Symmetric under averaging (an estimate 2x too large gives +1, 2x too
    small gives -1).
    NaN for non-finite inputs and for negative variance estimates, which are
    not a std at all and are counted separately by the caller.
    """
    if (
        not np.isfinite(var_actual)
        or not np.isfinite(var_est)
        or not np.isfinite(floor)
        or var_est < 0
    ):
        return np.float64(np.nan)
    std_actual = np.sqrt(max(var_actual, 0.0))
    std_est = np.sqrt(var_est)
    return np.log2((std_est + floor) / (std_actual + floor))


# Metrics of one (actual, estimate) pair. All are signed and dimensionless:
#   min_err:        (min_est - min_actual) / range.  < 0 is a loose (sound)
#                   lower bound, > 0 is an unsound one.
#   max_err:        (max_est - max_actual) / range.  > 0 is loose, < 0 unsound.
#   mean_err:       (mean_est - mean_actual) / std, in observed std units.
#   std_log2_ratio: log2 of estimated over observed std (floored).
Metrics = tuple[np.float64, np.float64, np.float64, np.float64]


def _compare_stats(
    actual: Stats, estimate: Stats, scale: Scale, verbose: bool = True
) -> Metrics:
    min_actual, max_actual, mean_actual, var_actual = actual
    min_est, max_est, mean_est, var_est = estimate
    min_err = _normalized_error(min_actual, min_est, scale.range)
    max_err = _normalized_error(max_actual, max_est, scale.range)
    mean_err = _normalized_error(mean_actual, mean_est, scale.std)
    std_ratio = _std_log2_ratio(var_actual, var_est, scale.std_floor)

    if verbose:
        print(
            f"  Actual min: {min_actual}, Estimated min: {min_est}, "
            f"range-normalized error: {min_err}"
        )
        print(
            f"  Actual max: {max_actual}, Estimated max: {max_est}, "
            f"range-normalized error: {max_err}"
        )
        print(
            f"  Actual mean: {mean_actual}, Estimated mean: {mean_est}, "
            f"error in std units: {mean_err}"
        )
        print(
            f"  Actual variance: {var_actual}, Estimated variance: {var_est}, "
            f"std log2 ratio: {std_ratio}"
        )

    return min_err, max_err, mean_err, std_ratio


def _is_unsound(actual: Stats, estimate: Stats) -> bool:
    """True when the estimated [min, max] does not contain the observed one."""
    min_actual, max_actual, _, _ = actual
    min_est, max_est, _, _ = estimate
    if not all(np.isfinite(v) for v in (min_actual, max_actual, min_est, max_est)):
        return False
    return bool(min_est > min_actual or max_est < max_actual)


# One comparison row: (op_id, result_id, op_name, axis, slice, *Metrics).
# ``op_id``/``result_id`` are the probe identifiers of the observed result and
# identify it uniquely; ``op_name`` is the op type and is informational only.
# Tensor-level rows use axis = slice = -1.
ResultRow = tuple[
    int, int, str, int, int, np.float64, np.float64, np.float64, np.float64
]
# Index of the first metric in a ResultRow.
METRICS_START = 5

# Aggregated row written to the results CSV. ``num_inputs`` is how many inputs
# produced a row for this (op_id, result_id, axis, slice); metrics are
# nanmeans over them.
AggregatedRow = tuple[
    int, int, str, int, int, np.float64, np.float64, np.float64, np.float64, int
]

METRIC_NAMES = ("min_err", "max_err", "mean_err", "std_log2_ratio")


def _output_csv_results(out_file: str, results: list[AggregatedRow]):
    results_dir = TMP_DIR / "results"
    os.makedirs(results_dir, exist_ok=True)

    header = [
        "op_id",
        "result_id",
        "op_name",
        "axis",
        "slice",
        *METRIC_NAMES,
        "num_inputs",
    ]
    with open(results_dir / out_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(results)


def _aggregate_rows(per_input: list[list[ResultRow]]) -> list[AggregatedRow]:
    """Mean each metric across inputs, keyed by (op_id, result_id, axis, slice).

    Key order follows the first input that produced the key. The op name is
    taken from that first row (it is the same for every input).
    """
    Key = tuple[int, int, int, int]
    order: list[Key] = []
    names: dict[Key, str] = {}
    buckets: dict[Key, list[tuple]] = {}
    for rows in per_input:
        for op_id, result_id, op_name, axis, slice_idx, *metrics in rows:
            key = (op_id, result_id, axis, slice_idx)
            if key not in buckets:
                order.append(key)
                names[key] = op_name
                buckets[key] = []
            buckets[key].append(metrics)

    aggregated: list[AggregatedRow] = []
    for op_id, result_id, axis, slice_idx in order:
        key = (op_id, result_id, axis, slice_idx)
        means = _nanmean_rows(buckets[key])
        aggregated.append(
            (op_id, result_id, names[key], axis, slice_idx, *means, len(buckets[key]))
        )
    return aggregated


def _nanmean_rows(rows: list[tuple]) -> np.ndarray:
    """Column-wise nanmean; all-NaN columns yield NaN without a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.asarray(rows, dtype=np.float64), axis=0)


def _nanmedian_rows(rows: list[tuple]) -> np.ndarray:
    """Column-wise nanmedian; all-NaN columns yield NaN without a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(np.asarray(rows, dtype=np.float64), axis=0)


def _format_metric_means(rows: list[tuple]) -> str:
    """``name=mean`` pairs of the signed metric means over ``rows``."""
    means = _nanmean_rows(rows)
    return ", ".join(f"{name}={val}" for name, val in zip(METRIC_NAMES, means))


def _tensor_level_summary(rows: list[ResultRow]) -> str:
    """Summary of the tensor-level (axis = slice = -1) metrics.

    Reports the mean and median absolute value of each metric. Absolute
    values keep loose and unsound bounds from cancelling across ops; the
    median is robust to the few ops whose estimate blows up.
    """
    tensor_rows = [row[METRICS_START:] for row in rows if row[3] == -1 and row[4] == -1]
    if not tensor_rows:
        return "no tensor-level rows"
    abs_rows = [tuple(np.abs(v) for v in row) for row in tensor_rows]
    abs_means = _nanmean_rows(abs_rows)
    abs_medians = _nanmedian_rows(abs_rows)
    return f"tensor-level |metric| over {len(tensor_rows)} ops: " + ", ".join(
        f"{name} mean={mean} median={median}"
        for name, mean, median in zip(METRIC_NAMES, abs_means, abs_medians)
    )


# Per-axis slice estimations for one result: one list of Stats per axis. An
# empty list means the axis is uniform (described by the tensor summary).
ResultAxes = list[list[Stats]]


def _collect_vishap_estimations(
    annotated_module,
) -> tuple[list[str], list[Stats], list[ResultAxes]]:
    """Collect Vishap distribution estimations from MLIR module.

    Returns, per observed result: the op name, the tensor-level summary, and
    the per-axis slice estimations from ``vishap.axis_distribution`` (empty
    for every axis when the attribute is absent).
    """
    entry_func = annotated_module.body.operations[0]

    op_names = []
    estimated_dists = []
    estimated_axes: list[ResultAxes] = []
    vishap_dist_attr = "vishap.distribution"
    vishap_axis_dist_attr = "vishap.axis_distribution"

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

        if vishap_axis_dist_attr in op.attributes:
            axis_dists = list(op.operation.attributes[vishap_axis_dist_attr])
        else:
            axis_dists = [None] * len(results)

        for dist, result_axes, result in zip(dists, axis_dists, results):
            if not is_float_ranked_tensor_type(result.type):
                continue
            op_names.append(op.operation.name)
            estimated_dists.append(tuple(np.float64(val) for val in dist))
            if result_axes is None:
                estimated_axes.append([[] for _ in range(len(result.type.shape))])
            else:
                estimated_axes.append(
                    [
                        [
                            tuple(np.float64(val) for val in slice_dist)
                            for slice_dist in axis
                        ]
                        for axis in result_axes
                    ]
                )
        return WalkResult.ADVANCE

    entry_func.operation.walk(collect_dists)

    return op_names, estimated_dists, estimated_axes


def _compare_results(
    op_names: list[str],
    estimated_dists: list[Stats],
    estimated_axes: list[ResultAxes],
    report_file: Path,
    verbose: bool = True,
) -> list[ResultRow]:
    # Collect true distributions (tensor-level and per-slice) from the probe
    # report
    observations = parse_probe_report_observations(report_file)

    assert len(observations) == len(estimated_dists), (
        "Number of estimated distributions does not match number of reported "
        f"distributions ({len(estimated_dists)} estimated vs "
        f"{len(observations)} reported)"
    )

    results: list[ResultRow] = []
    # Degenerate cases that the metrics cannot express and so return NaN for.
    num_degenerate_tensors = 0
    num_negative_var = 0
    num_unsound = 0
    num_compared = 0

    if verbose:
        print("=" * 10, "Distribution comparison", "=" * 10)
    for idx, (op_name, obs, est_stats, est_axes) in enumerate(
        zip(op_names, observations, estimated_dists, estimated_axes)
    ):
        if verbose:
            print(
                f"{idx} - Distribution for {op_name} "
                f"(op {obs.op_id}, result {obs.result_id})"
            )
        scale = Scale(obs.summary)
        if scale.is_degenerate():
            num_degenerate_tensors += 1
            if verbose:
                print("  (all-zero observed tensor: normalized metrics are NaN)")

        def compare(actual: Stats, estimate: Stats, show: bool) -> Metrics:
            nonlocal num_negative_var, num_unsound, num_compared
            num_compared += 1
            if np.isfinite(estimate[3]) and estimate[3] < 0:
                num_negative_var += 1
            if _is_unsound(actual, estimate):
                num_unsound += 1
            return _compare_stats(actual, estimate, scale, verbose=show)

        tensor_metrics = compare(obs.summary, est_stats, verbose)
        results.append((obs.op_id, obs.result_id, op_name, -1, -1, *tensor_metrics))

        # Per-slice comparison. The static estimate of a slice is the slice
        # estimation when the axis is non-uniform, and the tensor summary
        # otherwise (mirroring TensorDistribution::slice). Slice rows are
        # normalized by the tensor-level observed scale so they are
        # comparable across slices and defined for constant slices.
        for axis in sorted(obs.axes):
            actual_slices = obs.axes[axis]
            est_slices = est_axes[axis] if axis < len(est_axes) else []
            uniform = not est_slices
            if not uniform and len(est_slices) != len(actual_slices):
                logger.warning(
                    "Skipping axis %d of %s: %d estimated slices vs %d observed",
                    axis,
                    op_name,
                    len(est_slices),
                    len(actual_slices),
                )
                continue

            slice_rows: list[Metrics] = []
            for slice_idx, actual in enumerate(actual_slices):
                estimate = est_stats if uniform else est_slices[slice_idx]
                slice_rows.append(compare(actual, estimate, False))
                results.append(
                    (obs.op_id, obs.result_id, op_name, axis, slice_idx, *slice_rows[-1])
                )

            if verbose:
                print(
                    f"  Axis {axis} ({len(actual_slices)} slices, "
                    f"{'uniform' if uniform else 'per-slice'} estimate): "
                    f"signed means {_format_metric_means(slice_rows)}"
                )

    if num_compared:
        logger.info(
            "Compared %d (tensor + slice) pairs: %d unsound [min, max] "
            "(%.1f%%), %d negative variance estimates, %d all-zero observed "
            "tensors",
            num_compared,
            num_unsound,
            100.0 * num_unsound / num_compared,
            num_negative_var,
            num_degenerate_tensors,
        )

    return results


def _parse_args():
    parser = argparse.ArgumentParser(description="Utility script to MLIR networks.")
    parser.add_argument(
        "input_model", type=str, help="input MLIR model", metavar="model_path"
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="optional image or .npy path; replaces random data for the first function argument",
    )
    inputs.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="directory of images or .npy files; each input is compared and the metrics are averaged",
    )
    parser.add_argument(
        "--num-inputs",
        type=int,
        default=None,
        help="compare only the first N inputs from --input-dir, in sorted order (default: all)",
    )

    return parser.parse_args()


def _input_paths(args) -> list[Path | None]:
    """Inputs to compare. ``None`` means randomly generated data."""
    if args.num_inputs is not None and args.input_dir is None:
        raise SystemExit("--num-inputs requires --input-dir")
    if args.num_inputs is not None and args.num_inputs < 1:
        raise SystemExit(f"--num-inputs must be >= 1, got {args.num_inputs}")

    if args.input_dir is None:
        return [Path(args.input_file) if args.input_file else None]

    paths = collect_input_paths(args.input_dir)
    if args.num_inputs is not None:
        paths = paths[: args.num_inputs]
    if not paths:
        raise SystemExit(f"No inputs selected from {args.input_dir}")
    return paths


def _report_path(model: str, input_file: Path | None) -> Path:
    """Per-input probe report so successive inputs do not overwrite each other."""
    model_stem = Path(model).stem
    if input_file is None:
        name = f"{model_stem}.probe_report.csv"
    else:
        name = f"{model_stem}.{input_file.stem}.probe_report.csv"
    return VISHAP_TMP_DIR / "reports" / name


def _compare_one_input(
    input_asm: bytes,
    module,
    args,
    input_file: Path | None,
    verbose: bool,
    log_shapes: bool = False,
) -> list[ResultRow]:
    try:
        input_arrays, output_arrays = io_arrays_from_entry_func(
            module,
            input_file=input_file,
            input_initializer=_init_input,
        )
    except ValueError as err:
        raise SystemExit(str(err)) from None
    if log_shapes:
        logger.info(
            "Inferred inputs: %s",
            [(tuple(arr.shape), arr.dtype.name) for arr in input_arrays],
        )
        logger.info(
            "Inferred outputs: %s",
            [(tuple(arr.shape), arr.dtype.name) for arr in output_arrays],
        )
    input_stats = [get_array_stats(arr) for arr in input_arrays]

    # Save input arrays for later use
    inputs_dir = VISHAP_TMP_DIR / "inputs"
    os.makedirs(inputs_dir, exist_ok=True)
    model_stem = Path(args.input_model).stem
    input_tag = input_file.stem if input_file else "random"
    for idx, arr in enumerate(input_arrays):
        np.save(inputs_dir / f"{model_stem}.{input_tag}.input{idx}.npy", arr)

    # Estimates depend on this input's distribution, so re-annotate per input.
    annotated_module = annotate_distributions(
        input_asm, input_stats, annotate_axes=True
    )
    op_names, estimated_dists, estimated_axes = _collect_vishap_estimations(
        annotated_module
    )

    reports_dir = VISHAP_TMP_DIR / "reports"
    os.makedirs(reports_dir, exist_ok=True)
    report_file = _report_path(args.input_model, input_file)
    _run_network(
        annotated_module, input_arrays, output_arrays, report_file
    )

    return _compare_results(
        op_names, estimated_dists, estimated_axes, report_file, verbose=verbose
    )


def _main():
    args = _parse_args()
    input_paths = _input_paths(args)
    verbose = len(input_paths) == 1

    with open(args.input_model, "rb") as f:
        input_asm = f.read()

    module = parse_vishap_module(input_asm)

    per_input: list[list[ResultRow]] = []
    for index, input_file in enumerate(input_paths, start=1):
        label = str(input_file) if input_file else "random input"
        if not verbose:
            print(f"[{index}/{len(input_paths)}] {label}")
        rows = _compare_one_input(
            input_asm, module, args, input_file, verbose, log_shapes=index == 1
        )
        per_input.append(rows)
        if not verbose:
            print(f"  {_tensor_level_summary(rows)}")

    aggregated = _aggregate_rows(per_input)
    results_filename = Path(args.input_model).with_suffix(".csv").name
    _output_csv_results(results_filename, aggregated)
    print(
        f"Wrote {len(aggregated)} aggregated rows "
        f"({len(input_paths)} input(s)) to {TMP_DIR / 'results' / results_filename}"
    )


if __name__ == "__main__":
    _main()
