import argparse
import copy
import csv
import io
import logging
import numpy as np
import onnx
import onnxruntime as ort
import os
import sys
import tempfile
import time

from dataclasses import dataclass
from enum import Enum, IntEnum
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.calibrate import (
    CalibrationMethod,
    CalibrationDataReader,
    MinMaxCalibrater,
    TensorData,
    TensorsData,
    create_calibrator,
)
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
from onnxruntime.quantization.quant_utils import (
    load_model_with_shape_infer,
    update_opset_version,
)
from onnxruntime.quantization.registry import QDQRegistry, QLinearOpsRegistry
from pathlib import Path

from onnx2mlir import convert_onnx_to_mlir, load_onnx_model
from mlir_runtime import (
    CONSTANT_OP_NAMES,
    mixed_candidate_op_types,
    mixed_candidate_predecessors,
)
from utils import (
    Stats,
    TMP_DIR,
    VISHAP_AXIS_DIST_ATTR,
    VISHAP_DIST_ATTR,
    annotate_distributions,
    annotate_distributions_on_module,
    get_array_stats,
    is_image_path,
    is_nchw_layout,
    parse_onnx_loc_name,
    parse_vishap_module,
    preprocess_image,
    spatial_size,
    collect_input_paths,
)

DEFAULT_SPATIAL_SIZE = 224

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

QUANT_TMP_DIR = TMP_DIR / "quantization"


class LogicalResult(IntEnum):
    SUCCESS = 0
    FAILURE = 1


class RealDataCalibrationReader(CalibrationDataReader):
    """
    A data reader that feeds real batches of data to the calibrator.
    """

    def __init__(
        self,
        model_path: str,
        data_batches: list[np.ndarray | dict[str, np.ndarray]],
    ):
        self.data_batches = data_batches
        self.iterator = iter(self.data_batches)

        # A bare array is wrapped with the model's first input name. Named
        # feeds already carry every input and are returned unchanged.
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = session.get_inputs()[0].name

    def get_next(self) -> dict | None:
        """
        ORT calls this method repeatedly until it returns None.
        Each dictionary returned represents one batch of calibration data.
        """
        try:
            batch = next(self.iterator)
        except StopIteration:
            # Returning None tells the calibrator that we are out of data
            return None
        return _as_session_feed(batch, self.input_name)


class QuantMethod(Enum):
    calibration = "calibration"
    vishap = "vishap"
    mixed = "mixed"
    random = "random"

    def __str__(self):
        return self.value


class RangeMode(Enum):
    """How estimated per-axis slice distributions become a quant range.

    `lambda`: union of per-slice lambda clips (mean ± λ√variance).
    `percentile`: one percentile of the raw per-slice min and max estimates.
    """

    lambda_clip = "lambda"
    percentile = "percentile"

    def __str__(self):
        return self.value


def _ort_dtype_to_numpy(ort_type: str) -> np.dtype:
    if ort_type == "tensor(int64)":
        return np.int64
    if ort_type == "tensor(int32)":
        return np.int32
    if ort_type == "tensor(float16)":
        return np.float16
    return np.float32


def _as_session_feed(
    batch: np.ndarray | dict[str, np.ndarray], input_name: str
) -> dict[str, np.ndarray]:
    """Return a named ORT feed. A bare array uses `input_name`."""
    if isinstance(batch, dict):
        return batch
    return {input_name: batch}


@dataclass(frozen=True)
class ModelInputSpec:
    """One concrete ONNX Runtime input."""

    name: str
    shape: list[int]
    dtype: np.dtype


def _concrete_input_shape(
    raw_shape: list, model_path: str, input_name: str
) -> list[int]:
    """Resolve a model input shape.

    A 4D image may leave the batch and spatial sizes symbolic: batch becomes
    1 and each spatial dim becomes `DEFAULT_SPATIAL_SIZE`. Any other symbolic
    dimension is an error.
    """
    shape: list[int] = []
    for index, dim in enumerate(raw_shape):
        if isinstance(dim, str) or dim is None:
            if len(raw_shape) != 4:
                raise ValueError(
                    f"Model input '{input_name}' of {model_path} has symbolic "
                    f"shape {raw_shape}; only 4D image inputs may omit dimensions"
                )
            shape.append(1 if index == 0 else DEFAULT_SPATIAL_SIZE)
        else:
            shape.append(int(dim))
    return shape


def get_model_input_specs(model_path: str) -> list[ModelInputSpec]:
    """Return one concrete spec per ONNX Runtime input, in model order."""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    specs: list[ModelInputSpec] = []
    for inp in session.get_inputs():
        shape = _concrete_input_shape(list(inp.shape), model_path, inp.name)
        dtype = _ort_dtype_to_numpy(inp.type)
        if len(shape) == 4:
            layout = "NCHW" if is_nchw_layout(shape) else "NHWC"
            logger.info(
                f"Model input '{inp.name}': shape={shape}, "
                f"layout={layout}, dtype={dtype}"
            )
        else:
            logger.info(f"Model input '{inp.name}': shape={shape}, dtype={dtype}")
        specs.append(ModelInputSpec(inp.name, shape, dtype))
    return specs


def _is_single_image_input(specs: list[ModelInputSpec]) -> bool:
    """True when the model has one 4D floating-point input."""
    if len(specs) != 1 or len(specs[0].shape) != 4:
        return False
    return np.issubdtype(specs[0].dtype, np.floating)


def _shuffle_paths(paths: list, seed: int | None) -> list:
    """Return paths shuffled with seed, or unchanged when seed is None."""
    if seed is None or len(paths) <= 1:
        return paths
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    return [paths[i] for i in order]


def _load_image_feeds(
    path: Path,
    spec: ModelInputSpec,
    max_images: int | None,
    seed: int | None,
    model_path: str,
) -> list[dict[str, np.ndarray]]:
    """Load .npy tensors or images for a single 4D input and name each one."""
    shape = spec.shape
    is_nchw = is_nchw_layout(shape)
    dtype = spec.dtype
    height, width = spatial_size(shape, is_nchw)

    if path.is_file() and path.suffix.lower() == ".npy":
        array = np.load(path)
        if tuple(array.shape) != tuple(shape):
            raise ValueError(
                f"Input {path} has shape {array.shape}, model expects {shape}"
            )
        return [{spec.name: array.astype(dtype, copy=False)}]

    if path.is_dir():
        npy_files = sorted(path.glob("*.npy"))
        if npy_files and not any(
            is_image_path(p) for p in path.rglob("*") if p.is_file()
        ):
            npy_files = _shuffle_paths(npy_files, seed)
            if max_images is not None:
                npy_files = npy_files[:max_images]
            return [{spec.name: np.load(p)} for p in npy_files]

    image_paths = _shuffle_paths(collect_input_paths(path), seed)
    if max_images is not None:
        image_paths = image_paths[:max_images]

    batches = [preprocess_image(p, height, width, is_nchw, dtype) for p in image_paths]
    logger.info(f"Loaded {len(batches)} input(s) at {height}x{width} for {model_path}")
    return [{spec.name: batch} for batch in batches]


def _collect_npz_paths(path: Path) -> list[Path]:
    """Return one .npz path, or every .npz file under a directory."""
    if path.is_file():
        if path.suffix.lower() != ".npz":
            raise ValueError(
                f"Unsupported input file type: {path.suffix} (expected .npz)"
            )
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    files = sorted(p for p in path.rglob("*.npz") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No .npz files found under {path}")
    return files


def _load_npz_feed(path: Path, specs: list[ModelInputSpec]) -> dict[str, np.ndarray]:
    """Load one sample. Keys follow `specs`, and extra arrays are ignored."""
    with np.load(path) as data:
        feed: dict[str, np.ndarray] = {}
        for spec in specs:
            if spec.name not in data.files:
                raise ValueError(f"{path} is missing input '{spec.name}'")
            array = np.asarray(data[spec.name])
            if tuple(array.shape) != tuple(spec.shape):
                raise ValueError(
                    f"Input {path} field '{spec.name}' has shape {array.shape}, "
                    f"model expects {tuple(spec.shape)}"
                )
            # Copy out of the archive before it is closed.
            feed[spec.name] = np.array(array, dtype=spec.dtype, copy=True)
        return feed


def _load_npz_feeds(
    path: Path,
    specs: list[ModelInputSpec],
    max_images: int | None,
    seed: int | None,
    model_path: str,
) -> list[dict[str, np.ndarray]]:
    """Load .npz samples whose arrays match every model input."""
    npz_paths = _shuffle_paths(_collect_npz_paths(path), seed)
    if max_images is not None:
        npz_paths = npz_paths[:max_images]
    feeds = [_load_npz_feed(npz_path, specs) for npz_path in npz_paths]
    logger.info(
        "Loaded %d input(s) with %d tensor(s) each for %s",
        len(feeds),
        len(specs),
        model_path,
    )
    return feeds


def load_inputs_for_model(
    model_path: str,
    input_path: str | Path,
    max_images: int | None = None,
    seed: int | None = None,
) -> list[dict[str, np.ndarray]]:
    """Load one named feed per sample.

    A model with a single 4D float input still accepts images and `.npy`
    tensors. Every other model reads `.npz` files that contain each input
    by name. `max_images` limits how many samples are returned.
    """
    path = Path(input_path)
    specs = get_model_input_specs(model_path)
    if _is_single_image_input(specs):
        return _load_image_feeds(path, specs[0], max_images, seed, model_path)
    return _load_npz_feeds(path, specs, max_images, seed, model_path)


def export_model_input(
    model_path: str,
    input_path: str,
    output_npy: str,
    seed: int | None = None,
) -> None:
    """Write one preprocessed sample for later comparison.

    A single-input model is saved as `.npy`. A multi-input model is saved as
    `.npz`, and `output_npy` must already end in `.npz`.
    """
    feeds = load_inputs_for_model(model_path, input_path, max_images=1, seed=seed)
    feed = feeds[0]
    os.makedirs(os.path.dirname(os.path.abspath(output_npy)) or ".", exist_ok=True)
    if len(feed) == 1:
        array = next(iter(feed.values()))
        np.save(output_npy, array)
        logger.info(f"Exported input tensor of shape {array.shape} to {output_npy}")
        return
    if Path(output_npy).suffix.lower() != ".npz":
        raise ValueError(
            "Multi-input models require --export-input to end with .npz, "
            f"got {output_npy}"
        )
    np.savez(output_npy, **feed)
    logger.info(
        "Exported %d input tensor(s) to %s",
        len(feed),
        output_npy,
    )


def _onnx_node_maps(
    onnx_model: onnx.ModelProto,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return (node_name -> outputs, node_name -> op_type) for named nodes."""
    node_name_to_outputs: dict[str, list[str]] = {}
    node_name_to_op_type: dict[str, str] = {}
    for node in onnx_model.graph.node:
        if not node.name:
            continue
        node_name_to_outputs[node.name] = list(node.output)
        node_name_to_op_type[node.name] = node.op_type
    return node_name_to_outputs, node_name_to_op_type


def _sanitize_op_type_label(op_type: str) -> str:
    """Make an ONNX op type safe for filenames and CSV method labels."""
    return op_type.replace("/", "_").replace(":", "_")


def _non_constant_onnx_op_types(onnx_model: onnx.ModelProto) -> set[str]:
    return {
        node.op_type for node in onnx_model.graph.node if node.op_type != "Constant"
    }


def _list_mixed_op_types(input_model: str) -> int:
    """Print mixed float ONNX op types (one per line) and return an exit code."""
    onnx_model, mlir_module = _load_onnx_and_mlir(input_model)
    if onnx_model is None or mlir_module is None:
        return 1
    try:
        vishap_module = parse_vishap_module(_module_bytecode(mlir_module))
    except ImportError:
        logger.error(
            "Unable to import Vishap Python bindings. Make sure to build Vishap with bindings enabled."
        )
        return 1
    except Exception as e:
        logger.error("Failed to parse MLIR module for op-type listing")
        logger.error(e)
        return 1

    node_name_to_outputs, node_name_to_op_type = _onnx_node_maps(onnx_model)
    for op_type in mixed_candidate_op_types(
        vishap_module, node_name_to_outputs, node_name_to_op_type
    ):
        print(op_type)
    return 0


def _get_onnx_node_names(onnx_model: onnx.ModelProto) -> list[str]:
    node_names = []

    # Iterate through all nodes in the graph
    for i, node in enumerate(onnx_model.graph.node):
        if node.name:
            node_names.append(node.name)
        else:
            # ONNX nodes aren't strictly required to have a 'name' attribute.
            # If it's missing, we can identify it by its operator type and primary output.
            output_name = node.output[0] if node.output else "No_Output"
            fallback_name = f"Unnamed_{node.op_type}_idx{i}_(Out:{output_name})"
            node_names.append(fallback_name)

    return node_names


# Slice distributions for one result: one list per axis. An empty list means
# that axis is uniform (every slice matches the per-tensor summary).
ResultAxes = list[list[Stats]]


def _parse_stats_attr(dist) -> Stats:
    """Read one `[min, max, mean, variance]` distribution attribute."""
    vals = [np.float64(val) for val in dist]
    if len(vals) != 4:
        raise ValueError(f"expected 4 distribution values, got {len(vals)}")
    return (vals[0], vals[1], vals[2], vals[3])


def _parse_result_axes(result_axes) -> ResultAxes:
    """Read one result's per-axis slice distributions."""
    return [
        [_parse_stats_attr(slice_dist) for slice_dist in axis] for axis in result_axes
    ]


def _lambda_clip(lambd: int, stats: Stats) -> tuple[float, float]:
    """Clip `stats` to `mean ± λ√variance`, intersected with `[min, max]`."""
    min_est, max_est, mean_est, var_est = stats
    if var_est < 0 or not np.isfinite(var_est):
        return np.nan, np.nan
    std = np.sqrt(var_est)
    lowest = max(min_est, mean_est - lambd * std)
    highest = min(max_est, mean_est + lambd * std)
    return float(lowest), float(highest)


def _most_informative_axis(axes: ResultAxes) -> int | None:
    """Axis with the largest variance of slice means, matching the analysis.

    Non-finite scores are ignored. Uniform (empty) axes are skipped.
    """
    best: int | None = None
    best_score = -1.0
    for axis, slices in enumerate(axes):
        if not slices:
            continue
        means = np.array([s[2] for s in slices], dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            score = float(means.var())
        if not np.isfinite(score):
            continue
        if best is None or score > best_score:
            best = axis
            best_score = score
    return best


def _finite_float32_bounds(
    low: float, high: float
) -> tuple[np.float32, np.float32] | None:
    """Clamp an ordered interval into the finite float32 range.

    Non-finite or inverted bounds are unusable. ORT stores activation ranges
    as float32 and rejects a NaN scale.
    """
    if not np.isfinite(low) or not np.isfinite(high) or low > high:
        return None
    limit = float(np.finfo(np.float32).max)
    return np.float32(min(max(low, -limit), limit)), np.float32(
        min(max(high, -limit), limit)
    )


def _summary_quant_range(
    lambd: int, stats: Stats, *, use_real_minmax: bool
) -> tuple[float, float]:
    min_est, max_est, _, _ = stats
    if use_real_minmax:
        return float(min_est), float(max_est)
    return _lambda_clip(lambd, stats)


def _percentile_slice_range(
    axes: ResultAxes, percentile: float
) -> tuple[float, float] | None:
    """Percentile of the raw per-slice min and max on the most informative axis.

    The same percentile is taken of the slice mins and of the slice maxes, so
    50 is the median of each. Slices with non-finite or inverted bounds are
    dropped. Returns `None` when there is no usable axis or no slice survives.
    """
    axis = _most_informative_axis(axes)
    if axis is None:
        return None

    mins: list[float] = []
    maxs: list[float] = []
    for slice_stats in axes[axis]:
        low, high = float(slice_stats[0]), float(slice_stats[1])
        if not np.isfinite(low) or not np.isfinite(high) or low > high:
            continue
        mins.append(low)
        maxs.append(high)
    if not mins:
        return None
    # Each kept slice has min <= max, so the same percentile stays ordered.
    return (
        float(np.percentile(mins, percentile)),
        float(np.percentile(maxs, percentile)),
    )


def _range_from_axes(
    lambd: int,
    stats: Stats,
    axes: ResultAxes | None,
    *,
    use_real_minmax: bool,
    range_mode: RangeMode = RangeMode.lambda_clip,
    percentile: float = 50,
) -> tuple[float, float, bool]:
    """Return `(rmin, rmax, used_per_axis)`.

    Measured ranges keep the summary min/max. Estimated ranges come from the
    per-slice estimates on the most informative axis: the union of per-slice
    lambda clips (`lambda` mode) or one percentile of the raw per-slice min
    and max (`percentile` mode; 50 is the median). Both fall back to the
    summary lambda clip when that axis is missing or the slice range is
    unusable.
    """
    summary_low, summary_high = _summary_quant_range(
        lambd, stats, use_real_minmax=use_real_minmax
    )
    if use_real_minmax or not axes:
        return summary_low, summary_high, False

    if range_mode is RangeMode.percentile:
        slice_range = _percentile_slice_range(axes, percentile)
        if slice_range is None:
            return summary_low, summary_high, False
        slice_low, slice_high = slice_range
        # Dead slices pull a low percentile onto a near-empty range.
        # QLinearGlobalAveragePool then rejects
        # x_scale / (y_scale * spatial) >= 256. A slice range more than
        # 256x narrower than the summary lambda clip is that dead-slice
        # inlier, so keep the summary clip instead.
        summary_width = summary_high - summary_low
        slice_width = slice_high - slice_low
        if (
            not np.isfinite(summary_width)
            or summary_width < 0
            or slice_width < summary_width / 256.0
        ):
            return summary_low, summary_high, False
        return slice_low, slice_high, True

    axis = _most_informative_axis(axes)
    if axis is None:
        return summary_low, summary_high, False

    lows: list[float] = []
    highs: list[float] = []
    for slice_stats in axes[axis]:
        low, high = _lambda_clip(lambd, slice_stats)
        if not np.isfinite(low) or not np.isfinite(high) or low > high:
            continue
        lows.append(low)
        highs.append(high)
    if not lows:
        return summary_low, summary_high, False

    lowest = min(lows)
    highest = max(highs)
    if not np.isfinite(lowest) or not np.isfinite(highest) or lowest > highest:
        return summary_low, summary_high, False
    return lowest, highest, True


def _vishap_dist_to_tensor_quant(
    lambd,
    stats: Stats,
    *,
    use_real_minmax: bool = False,
    axes: ResultAxes | None = None,
    range_mode: RangeMode = RangeMode.lambda_clip,
    percentile: float = 50,
) -> tuple[list[dict[str, np.float32]], bool]:
    """Map one distribution to an ORT per-tensor `rmin`/`rmax` override.

    The bool is true when the range came from per-axis slice estimates.
    """
    lowest, highest, used_per_axis = _range_from_axes(
        lambd,
        stats,
        axes,
        use_real_minmax=use_real_minmax,
        range_mode=range_mode,
        percentile=percentile,
    )
    bounds = _finite_float32_bounds(lowest, highest)
    if bounds is None:
        bounds = _finite_float32_bounds(float(stats[0]), float(stats[1]))
        used_per_axis = False
    if bounds is None:
        bounds = (np.float32(0), np.float32(0))
        used_per_axis = False
    return [{"rmin": bounds[0], "rmax": bounds[1]}], used_per_axis


def _ns_to_ms(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) / 1_000_000


def _save_perf_stages(
    out_file: str,
    network: str,
    method: str,
    stages: list[tuple[str, float]],
) -> None:
    """Append one CSV row per (stage, time_ms) for a quantization run."""
    is_empty = not os.path.isfile(out_file) or os.path.getsize(out_file) == 0
    with open(out_file, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if is_empty:
            writer.writerow(["network", "time_ms", "method", "stage"])
        for stage, time_ms in stages:
            writer.writerow([network, time_ms, method, stage])


def _collect_vishap_dists(
    mlir_module, valid_onnx_names: set[str]
) -> tuple[dict[str, list[Stats]], dict[str, list[ResultAxes]], list[Stats], set[str]]:
    """Walk an annotated MLIR module and return node stats, axes, outputs, and constants.

    Per-result summaries come from `vishap.distribution`. When
    `vishap.axis_distribution` is present and has one entry per summary, it
    is returned in parallel; otherwise that node falls back to summaries only.

    The constant-name set contains ONNX loc names whose MLIR op is in
    `CONSTANT_OP_NAMES` (`stablehlo.constant` / `arith.constant`).
    """

    onnx_name_to_dists: dict[str, list[Stats]] = {}
    onnx_name_to_axes: dict[str, list[ResultAxes]] = {}
    func_output_stats: list[Stats] = []
    constant_onnx_names: set[str] = set()

    # Assume entry function is the first operation in the module's body
    entry_func = mlir_module.body.operations[0]
    for op in entry_func.regions[0].blocks[0]:
        if VISHAP_DIST_ATTR not in op.attributes:
            continue

        onnx_name = parse_onnx_loc_name(str(op.location))
        if onnx_name not in valid_onnx_names:
            continue

        summaries = [
            _parse_stats_attr(dist)
            for dist in op.operation.attributes[VISHAP_DIST_ATTR]
        ]
        onnx_name_to_dists[onnx_name] = summaries
        # A later op with the same loc replaces the summaries, so drop any
        # axes collected from the previous one unless this op has its own.
        onnx_name_to_axes.pop(onnx_name, None)
        if VISHAP_AXIS_DIST_ATTR in op.attributes:
            try:
                parsed_axes = [
                    _parse_result_axes(result_axes)
                    for result_axes in op.operation.attributes[VISHAP_AXIS_DIST_ATTR]
                ]
            except (TypeError, ValueError) as exc:
                logger.warning("Ignoring axis distributions for %s: %s", onnx_name, exc)
            else:
                if len(parsed_axes) != len(summaries):
                    logger.warning(
                        "Ignoring axis distributions for %s: %d result(s) vs %d summary(s)",
                        onnx_name,
                        len(parsed_axes),
                        len(summaries),
                    )
                else:
                    onnx_name_to_axes[onnx_name] = parsed_axes
        if op.name in CONSTANT_OP_NAMES:
            constant_onnx_names.add(onnx_name)

    if VISHAP_DIST_ATTR in entry_func.attributes:
        func_output_stats = [
            _parse_stats_attr(dist)
            for dist in entry_func.operation.attributes[VISHAP_DIST_ATTR]
        ]

    return (
        onnx_name_to_dists,
        onnx_name_to_axes,
        func_output_stats,
        constant_onnx_names,
    )


def _annotate_module_in_memory(mlir_module, input_stats: list[Stats]):
    """Run annotate-distributions on a MLIR module via the Vishap bindings.

    Raises:
        ImportError: If the Vishap bindings are not built/importable.

    Returns:
        The annotated module, owned by the Vishap bindings.
    """
    buf = io.BytesIO()
    mlir_module.operation.write_bytecode(file=buf)
    try:
        return annotate_distributions(buf.getvalue(), input_stats, annotate_axes=True)
    except ImportError:
        logger.error(
            "Unable to import Vishap Python bindings. Make sure to build Vishap with bindings enabled."
        )
        raise
    except Exception:
        raise


def _producer_result_axes(
    producer: str | None,
    output_name: str,
    node_name_to_outputs: dict[str, list[str]],
    onnx_name_to_axes: dict[str, list[ResultAxes]],
) -> ResultAxes | None:
    """Axis distributions for `output_name` on its producing ONNX node."""
    if producer is None:
        return None
    result_axes = onnx_name_to_axes.get(producer)
    outputs = node_name_to_outputs.get(producer, [])
    if result_axes is None or output_name not in outputs:
        return None
    out_idx = outputs.index(output_name)
    if out_idx >= len(result_axes):
        return None
    return result_axes[out_idx]


def _build_tensor_quant_overrides(
    onnx_model: onnx.ModelProto,
    onnx_name_to_dists: dict[str, list[Stats]],
    onnx_name_to_axes: dict[str, list[ResultAxes]],
    func_output_stats: list[Stats],
    input_stats: list[Stats],
    lambd: int,
    constant_onnx_names: set[str] | None = None,
    calibrated_onnx_names: set[str] | None = None,
    range_mode: RangeMode = RangeMode.lambda_clip,
    percentile: float = 50,
    locked_tensors: set[str] | None = None,
) -> dict[str, list[dict[str, np.float32]]]:
    """Map Vishap distribution estimates to ORT TensorQuantOverrides keys.

    `calibrated_onnx_names` is the set of ONNX nodes preloaded from ORT
    calibration (mixed mode). Those nodes, and graph outputs they produce,
    use measured min/max. Passing the set (including an empty one) also marks
    each floating-point graph input as measured. Integer inputs are left out
    of the overrides; their distributions are only for Vishap. `None` is the
    Vishap-only path, which keeps the lambda clip for every non-constant and
    for each floating-point graph input.

    Estimated ranges use the per-slice estimates on the most informative axis
    when `onnx_name_to_axes` has a non-uniform axis for that result. The
    function's own distribution attribute is summary-only, so graph outputs
    reuse the producing node's axes.

    `locked_tensors` are left unset so mixed quantization can keep the
    calibrator range for those names.
    """
    # Map ONNX node names (from MLIR locs) to actual output tensor names.
    # TensorQuantOverrides keys must be tensor names, not node names / ":0" ports.
    node_name_to_outputs = {
        node.name: list(node.output) for node in onnx_model.graph.node if node.name
    }
    output_to_producer = {
        output_name: node.name
        for node in onnx_model.graph.node
        if node.name
        for output_name in node.output
    }
    constant_names = constant_onnx_names or set()
    real_minmax_names = constant_names
    if calibrated_onnx_names is not None:
        real_minmax_names = constant_names | calibrated_onnx_names

    injected_stats: dict[str, list[dict[str, np.float32]]] = {}
    # Final source per tensor name, so graph-output overwrites are counted once.
    used_per_axis: dict[str, bool] = {}
    locked = locked_tensors or set()

    def _store(name: str, quant: list[dict[str, np.float32]], from_axes: bool) -> None:
        if name in locked:
            return
        injected_stats[name] = quant
        used_per_axis[name] = from_axes

    for node_name, out_stats in onnx_name_to_dists.items():
        outputs = node_name_to_outputs.get(node_name, [])
        use_real_minmax = node_name in real_minmax_names
        node_axes = onnx_name_to_axes.get(node_name)
        for idx, stats in enumerate(out_stats):
            if idx >= len(outputs):
                logger.warning(
                    "Skipping dist for %s output %d: node has only %d output(s)",
                    node_name,
                    idx,
                    len(outputs),
                )
                continue
            axes = (
                node_axes[idx]
                if node_axes is not None and idx < len(node_axes)
                else None
            )
            quant, from_axes = _vishap_dist_to_tensor_quant(
                lambd,
                stats,
                use_real_minmax=use_real_minmax,
                axes=axes,
                range_mode=range_mode,
                percentile=percentile,
            )
            _store(outputs[idx], quant, from_axes)

    output_names = [out.name for out in onnx_model.graph.output]
    initializer_names = {init.name for init in onnx_model.graph.initializer}
    graph_inputs = [
        graph_input
        for graph_input in onnx_model.graph.input
        if graph_input.name not in initializer_names
    ]
    if len(graph_inputs) != len(input_stats):
        raise ValueError(
            f"Expected {len(graph_inputs)} input distribution(s), "
            f"got {len(input_stats)}"
        )
    float_types = {
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.FLOAT16,
    }
    for graph_input, stats in zip(graph_inputs, input_stats):
        if not graph_input.type.HasField("tensor_type"):
            continue
        if graph_input.type.tensor_type.elem_type not in float_types:
            continue
        input_quant, input_from_axes = _vishap_dist_to_tensor_quant(
            lambd,
            stats,
            use_real_minmax=calibrated_onnx_names is not None,
            range_mode=range_mode,
            percentile=percentile,
        )
        _store(graph_input.name, input_quant, input_from_axes)
    for output_name, stats in zip(output_names, func_output_stats):
        producer = output_to_producer.get(output_name)
        use_real_minmax = producer is not None and producer in real_minmax_names
        axes = _producer_result_axes(
            producer, output_name, node_name_to_outputs, onnx_name_to_axes
        )
        quant, from_axes = _vishap_dist_to_tensor_quant(
            lambd,
            stats,
            use_real_minmax=use_real_minmax,
            axes=axes,
            range_mode=range_mode,
            percentile=percentile,
        )
        _store(output_name, quant, from_axes)

    per_axis_count = sum(used_per_axis.values())
    logger.info(
        "Quant ranges: %d from per-axis slice estimates, %d from per-tensor summary",
        per_axis_count,
        len(used_per_axis) - per_axis_count,
    )
    return injected_stats


def _default_op_types_to_quantize() -> list[str]:
    """Match quantize_static's default: QLinear + QDQ registries."""
    return list(set(QLinearOpsRegistry.keys()) | set(QDQRegistry.keys()))


def _float_activation_tensor_names(model: onnx.ModelProto) -> set[str]:
    """Collect names for float activations."""
    value_infos = {vi.name: vi for vi in model.graph.value_info}
    value_infos.update({out.name: out for out in model.graph.output})
    value_infos.update({inp.name: inp for inp in model.graph.input})
    initializer = {init.name for init in model.graph.initializer}
    tensor_types = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16}
    op_types = set(_default_op_types_to_quantize())

    tensors = set()
    for node in model.graph.node:
        if node.op_type not in op_types:
            continue
        for tensor_name in list(node.input) + list(node.output):
            vi = value_infos.get(tensor_name)
            if vi is None:
                continue
            if (
                vi.type.HasField("tensor_type")
                and vi.type.tensor_type.elem_type in tensor_types
                and tensor_name not in initializer
            ):
                tensors.add(tensor_name)
    return tensors


def _input_stats_from_batches(
    batches: list[dict[str, np.ndarray]],
) -> list[Stats]:
    """One distribution per feed key, in key order.

    Several samples are concatenated along the batch axis before the stats.
    Image models have one key, so this is still a one-element list.
    """
    if not batches:
        raise ValueError("No inputs to summarize")
    names = list(batches[0].keys())
    stats: list[Stats] = []
    for name in names:
        arrays = [batch[name] for batch in batches]
        combined = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
        stats.append(get_array_stats(combined))
    return stats


def _load_method_inputs(
    input_model: str,
    input_file: str | None,
    input_dir: str | None,
    num_inputs: int,
    seed: int | None,
    method_name: str,
) -> list[dict[str, np.ndarray]] | None:
    """Load up to `num_inputs` feeds for Vishap or mixed quantization."""
    if input_file:
        if num_inputs > 1:
            logger.error("--num-inputs > 1 requires --input-dir, not --input-file")
            return None
        input_source = input_file
    elif input_dir:
        input_source = input_dir
    else:
        logger.error(
            "%s quantization requires --input-file or --input-dir", method_name
        )
        return None

    batches = load_inputs_for_model(
        input_model, input_source, max_images=num_inputs, seed=seed
    )
    if not batches:
        logger.error(f"No inputs loaded from {input_source}")
        return None
    if len(batches) < num_inputs:
        logger.warning(
            "Requested %d input(s) but only loaded %d from %s",
            num_inputs,
            len(batches),
            input_source,
        )
    return batches


def _collect_ort_tensor_stats(
    onnx_model: onnx.ModelProto,
    batches: list[np.ndarray | dict[str, np.ndarray]],
    tensors_to_expose: set[str] | None = None,
    enable_fusion: bool = False,
) -> dict[str, Stats]:
    """Run ORT inference(s) exposing float activations and return merged stats.

    We fetch raw intermediate tensors and compute full (min, max, mean,
    variance) tuples, as required by Vishap. By default every float activation
    is exposed and graph optimizations are disabled, so fusions cannot remove
    an observed tensor. ``tensors_to_expose`` limits the extra outputs to that
    set, and ``enable_fusion`` runs at ``ORT_ENABLE_EXTENDED`` so nodes whose
    tensors were left out can fuse. Multiple batches are merged as if each
    activation were concatenated across inputs.
    """
    if not batches:
        return {}

    augmented = onnx.ModelProto()
    augmented.CopyFrom(onnx_model)
    graph = augmented.graph

    value_infos = {vi.name: vi for vi in graph.value_info}
    initializers = {init.name for init in graph.initializer}
    exposed = {out.name for out in graph.output}
    float_types = {
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.FLOAT16,
    }

    for node in graph.node:
        for tensor_name in node.output:
            if tensors_to_expose is not None and tensor_name not in tensors_to_expose:
                continue
            if tensor_name in exposed or tensor_name in initializers:
                continue
            vi = value_infos.get(tensor_name)
            if (
                vi is not None
                and vi.type.HasField("tensor_type")
                and vi.type.tensor_type.elem_type in float_types
            ):
                # Expose as graph output so we can observe it
                graph.output.append(vi)
                exposed.add(tensor_name)

    sess_options = ort.SessionOptions()
    if enable_fusion:
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
    else:
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
    session = ort.InferenceSession(
        augmented.SerializeToString(),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]
    if tensors_to_expose is not None:
        missing = sorted(tensors_to_expose - set(output_names))
        if missing:
            logger.warning(
                "Stats probe missed %d tensor(s): %s",
                len(missing),
                ", ".join(missing),
            )

    if len(batches) == 1:
        results = session.run(None, _as_session_feed(batches[0], input_name))
        return {
            name: get_array_stats(np.asarray(arr).astype(np.float32))
            for name, arr in zip(output_names, results)
        }

    counts: dict[str, int] = {}
    mins: dict[str, np.float64] = {}
    maxs: dict[str, np.float64] = {}
    sums: dict[str, np.float64] = {}
    sumsqs: dict[str, np.float64] = {}
    for batch in batches:
        results = session.run(None, _as_session_feed(batch, input_name))
        for name, arr in zip(output_names, results):
            a = np.asarray(arr).astype(np.float32)
            mn, mx, mean, var = get_array_stats(a)
            n = int(a.size)
            s = mean * n
            ss = (var + mean * mean) * n
            if name not in counts:
                counts[name] = n
                mins[name] = mn
                maxs[name] = mx
                sums[name] = s
                sumsqs[name] = ss
            else:
                counts[name] += n
                mins[name] = min(mins[name], mn)
                maxs[name] = max(maxs[name], mx)
                sums[name] += s
                sumsqs[name] += ss

    merged: dict[str, Stats] = {}
    for name, n in counts.items():
        mean = sums[name] / n
        var = sumsqs[name] / n - mean * mean
        merged[name] = (
            mins[name],
            maxs[name],
            np.float64(mean),
            np.float64(max(var, 0.0)),
        )
    return merged


def _tensors_data_from_overrides(
    model: onnx.ModelProto,
    injected_stats: dict[str, list[dict[str, np.float32]]],
) -> TensorsData:
    """Build MinMax TensorsData from overrides, with (0, 0) fallback for the rest.

    Fill unannotated tensors with degenerate (0, 0) ranges so QDQ still sees
    every activation.
    """
    zero = np.float32(0)
    data = {
        name: TensorData(lowest=zero, highest=zero)
        for name in _float_activation_tensor_names(model)
    }
    for name, ranges in injected_stats.items():
        data[name] = TensorData(
            lowest=np.float32(ranges[0]["rmin"]),
            highest=np.float32(ranges[0]["rmax"]),
        )
    return TensorsData(CalibrationMethod.MinMax, data)


def _qdq_quantize(
    model: onnx.ModelProto,
    tensors_range: TensorsData,
    output_model: str,
) -> None:
    """QDQ-quantize with the same arguments as ``_calibration_quantization``."""
    quantizer = QDQQuantizer(
        model,
        per_channel=False,
        reduce_range=False,
        weight_qType=QuantType.QInt8,
        activation_qType=QuantType.QUInt8,
        tensors_range=tensors_range,
        nodes_to_quantize=[],
        nodes_to_exclude=[],
        op_types_to_quantize=_default_op_types_to_quantize(),
        extra_options={},
    )
    quantizer.quantize_model()
    quantizer.model.save_model_to_file(output_model)


def _quantize_from_overrides(
    input_model: str,
    output_model: str,
    injected_stats: dict[str, list[dict[str, np.float32]]],
):
    """QDQ-quantize using supplied rmin/rmax, without a calibration inference pass."""
    model = load_model_with_shape_infer(Path(input_model))
    model = update_opset_version(model, QuantType.QInt8)
    tensors_range = _tensors_data_from_overrides(model, injected_stats)
    _qdq_quantize(model, tensors_range, output_model)


class _SelectiveMinMaxCalibrater(MinMaxCalibrater):
    """MinMax calibration that can leave activations out and fuse the rest.

    ``tensors_to_skip`` are not given ``ReduceMin``/``ReduceMax`` consumers.
    The session runs at ``ORT_ENABLE_EXTENDED``, so a node whose outputs are
    all in that set can fuse. Activations that are still measured keep their
    reductions, which pins them against fusion.
    """

    def __init__(
        self,
        model_path: str | Path,
        op_types_to_calibrate: list[str],
        augmented_model_path: str,
        tensors_to_skip: set[str],
    ):
        self._tensors_to_skip = set(tensors_to_skip)
        super().__init__(
            model_path,
            op_types_to_calibrate=op_types_to_calibrate,
            augmented_model_path=augmented_model_path,
            use_external_data_format=False,
            symmetric=False,
            moving_average=False,
            averaging_constant=0.01,
            max_intermediate_outputs=None,
            per_channel=False,
        )

    def select_tensors_to_calibrate(self, model: onnx.ModelProto):
        tensors, value_infos = super().select_tensors_to_calibrate(model)
        return set(tensors) - self._tensors_to_skip, value_infos

    def create_inference_session(self):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        self.infer_session = ort.InferenceSession(
            self.augmented_model_path,
            sess_options=sess_options,
            providers=self.execution_providers,
        )


def _prepare_calibration_model(model_path: str) -> tuple[onnx.ModelProto, bool]:
    """Shape-infer and opset-update the model ``quantize_static`` would calibrate.

    The bool is true when opset conversion replaced the model, so the
    calibrator must read the updated copy instead of ``model_path``.
    """
    model = load_model_with_shape_infer(Path(model_path))
    updated_model = update_opset_version(model, QuantType.QInt8)
    if updated_model is not model:
        return updated_model, True
    return model, False


def _collect_minmax_ranges(
    model_path: str,
    batches: list[np.ndarray | dict[str, np.ndarray]],
    tensors_to_skip: set[str] | None = None,
) -> tuple[onnx.ModelProto, TensorsData]:
    """Return the model and MinMax ranges ``quantize_static`` would use.

    Follows ``quantize_static`` through ``calibrator.compute_data()``. The
    ``TensorData`` values are the calibrator's own objects; do not rebuild
    them from ``get_array_stats``.

    ``tensors_to_skip`` drops those activations from the run and enables
    extended graph fusions. An empty set keeps the stock calibrator, including
    ``ORT_DISABLE_ALL``.
    """
    op_types = _default_op_types_to_quantize()
    model, is_model_updated = _prepare_calibration_model(model_path)
    skip = tensors_to_skip or set()

    # Named feeds are passed through. A bare array is wrapped with the first
    # input name, which quantize_static also reads from the original path
    # rather than the opset-updated copy.
    reader = RealDataCalibrationReader(model_path, batches)
    with tempfile.TemporaryDirectory(prefix="ort.quant.") as quant_tmp_dir:
        calibrator_input: str | onnx.ModelProto = model_path
        if is_model_updated:
            calibrator_input = copy.deepcopy(model)
        if isinstance(calibrator_input, onnx.ModelProto):
            saved_path = Path(quant_tmp_dir).joinpath("model_input.onnx").as_posix()
            onnx.save_model(
                calibrator_input, saved_path, save_as_external_data=True
            )
            calibrator_input = saved_path

        augmented_model_path = (
            Path(quant_tmp_dir).joinpath("augmented_model.onnx").as_posix()
        )
        if skip:
            calibrator = _SelectiveMinMaxCalibrater(
                Path(calibrator_input),
                op_types,
                augmented_model_path,
                skip,
            )
            calibrator.augment_graph()
            calibrator.create_inference_session()
        else:
            calibrator = create_calibrator(
                Path(calibrator_input),
                op_types,
                augmented_model_path=augmented_model_path,
                calibrate_method=CalibrationMethod.MinMax,
                use_external_data_format=False,
                providers=None,
                extra_options={},
            )
        calibrator.collect_data(reader)
        tensors_range = calibrator.compute_data()
    if not isinstance(tensors_range, TensorsData):
        raise TypeError(
            f"Unexpected type {type(tensors_range)} for calibration ranges"
        )
    return model, tensors_range


def _merge_activation_ranges(
    calibrated: TensorsData,
    vishap_overrides: dict[str, list[dict[str, np.float32]]],
    keep: set[str],
) -> TensorsData:
    """Keep calibrator ranges for ``keep``; replace the rest from Vishap.

    A finite Vishap range is written even when the calibrator did not return
    that name, so a statically analyzed activation left out of the calibration
    run still gets a quant range. Tensors with no finite Vishap range stay on
    the calibrator value when one exists, and are not inserted otherwise, so a
    hole never becomes ``[0, 0]``. A ``keep`` set that covers every calibrator
    tensor leaves each of those ``TensorData`` objects in place.
    """
    merged: dict[str, TensorData] = dict(calibrated.data)
    for name, ranges in vishap_overrides.items():
        if name in keep or not ranges:
            continue
        low = np.float32(ranges[0]["rmin"])
        high = np.float32(ranges[0]["rmax"])
        if not np.isfinite(low) or not np.isfinite(high):
            continue
        merged[name] = TensorData(lowest=low, highest=high)
    return TensorsData(calibrated.calibration_method, merged)


@dataclass(frozen=True)
class PreloadSelection:
    """Which calibrator tensors keep their measured range."""

    selected_outputs: set[str]
    keep: set[str]
    unselected_count: int
    selected_count: int
    pool_count: int


def _longest_path_depths(predecessors: dict[str, list[str]]) -> dict[str, int]:
    """Longest-path depth of each node. A node with no producers has depth 1.

    ``predecessors[name]`` lists the producer names. Raises ValueError on a cycle.
    """
    memo: dict[str, int] = {}
    visiting: set[str] = set()

    def depth(name: str) -> int:
        cached = memo.get(name)
        if cached is not None:
            return cached
        if name in visiting:
            raise ValueError(f"Cycle in mixed-candidate graph at {name}")
        visiting.add(name)
        producers = predecessors.get(name, [])
        if not producers:
            value = 1
        else:
            value = 1 + max(depth(producer) for producer in producers)
        visiting.remove(name)
        memo[name] = value
        return value

    for name in predecessors:
        depth(name)
    return {name: memo[name] for name in predecessors}


def _names_at_static_depth(depths: dict[str, int], depth: int) -> set[str]:
    """Names whose longest-path depth lands on the calibration stride.

    Depth ``D`` calibrates nodes at depths ``D, 2D+1, 3D+2, ...``, so the first
    calibrated node is ``D`` hops from the input and ``D`` statically analyzed
    nodes sit between later calibrated nodes on a chain.
    """
    period = depth + 1
    return {name for name, node_depth in depths.items() if node_depth % period == depth}


def _select_preload(
    model: onnx.ModelProto,
    calibrated_names: set[str],
    ratio: float | None,
    seed: int | None,
    vishap_op_type: str | None,
    vishap_names: set[str] | None = None,
    analyzable_names: set[str] | None = None,
) -> PreloadSelection:
    """Sample non-constant producers of calibrator tensors.

    The pool is every non-constant node that produces at least one name in
    ``calibrated_names``. ``keep`` is the outputs of the selected nodes plus
    every calibrator tensor that is not produced by the pool (graph inputs
    and constant outputs). At ``ratio`` 1 the unselected set is empty and
    ``keep`` covers ``calibrated_names``.

    Exactly one of ``ratio``, ``vishap_op_type``, and ``vishap_names`` is set.
    ``vishap_names`` are pool nodes left for Vishap; every other pool node is
    calibrated. ``analyzable_names``, when set, restricts the pool to ONNX
    nodes Vishap can annotate. Outputs of the rest stay measured, because
    leaving them out of calibration would leave a quant range unset.
    """
    provided = sum(
        value is not None for value in (ratio, vishap_op_type, vishap_names)
    )
    if provided != 1:
        raise ValueError(
            "exactly one of ratio, vishap_op_type, or vishap_names is required"
        )

    # (node name, op type, outputs), in graph order.
    pool: list[tuple[str, str, list[str]]] = []
    for node in model.graph.node:
        if node.op_type == "Constant":
            continue
        if analyzable_names is not None and node.name not in analyzable_names:
            continue
        if not any(name in calibrated_names for name in node.output):
            continue
        pool.append((node.name, node.op_type, list(node.output)))

    if vishap_names is not None:
        chosen = {i for i, (name, _, _) in enumerate(pool) if name not in vishap_names}
        logger.info(
            "Selecting %d/%d calibration ops for measured ranges "
            "(static-depth, %d left for Vishap)",
            len(chosen),
            len(pool),
            len(pool) - len(chosen),
        )
    elif vishap_op_type is not None:
        chosen = {
            i for i, (_, op_type, _) in enumerate(pool) if op_type != vishap_op_type
        }
        logger.info(
            "Selecting %d/%d calibration ops for measured ranges "
            "(vishap-op-type=%s, %d left for Vishap)",
            len(chosen),
            len(pool),
            vishap_op_type,
            len(pool) - len(chosen),
        )
    else:
        assert ratio is not None
        n = min(len(pool), max(0, int(round(ratio * len(pool)))))
        logger.info(
            "Selecting %d/%d calibration ops for measured ranges (ratio=%s)",
            n,
            len(pool),
            ratio,
        )
        if n == 0:
            chosen = set()
        else:
            rng = np.random.default_rng(seed)
            chosen = {int(i) for i in rng.choice(len(pool), size=n, replace=False)}

    selected_outputs: set[str] = set()
    pool_outputs: set[str] = set()
    unselected_count = 0
    for index, (_, _, outputs) in enumerate(pool):
        pool_outputs.update(outputs)
        if index in chosen:
            selected_outputs.update(outputs)
        else:
            unselected_count += 1

    keep = (selected_outputs & calibrated_names) | (calibrated_names - pool_outputs)
    return PreloadSelection(
        selected_outputs=selected_outputs,
        keep=keep,
        unselected_count=unselected_count,
        selected_count=len(chosen),
        pool_count=len(pool),
    )


def _mlir_attach_names(
    selected_outputs: set[str], node_name_to_outputs: dict[str, list[str]]
) -> set[str]:
    """ONNX node names on the MLIR model that produce a selected tensor."""
    return {
        name
        for name, outputs in node_name_to_outputs.items()
        if any(output in selected_outputs for output in outputs)
    }


def _load_onnx_and_mlir(input_model: str):
    try:
        onnx_model = load_onnx_model(input_model)
        mlir_module = convert_onnx_to_mlir(onnx_model)
        return onnx_model, mlir_module
    except Exception:
        logger.error(f"Failed to load ONNX model {input_model}")
        return None, None


def _vishap_quant_overrides(
    onnx_model: onnx.ModelProto,
    annotated_module,
    input_stats: list[Stats],
    lambd: int,
    calibrated_onnx_names: set[str] | None = None,
    range_mode: RangeMode = RangeMode.lambda_clip,
    percentile: float = 50,
    locked_tensors: set[str] | None = None,
) -> tuple[dict[str, list[dict[str, np.float32]]], list[tuple[str, float]]]:
    """Distribution overrides and the collect/build stage timings."""
    t0 = time.perf_counter_ns()
    valid_onnx_names = set(_get_onnx_node_names(onnx_model))
    (
        onnx_name_to_dists,
        onnx_name_to_axes,
        func_output_stats,
        constant_onnx_names,
    ) = _collect_vishap_dists(annotated_module, valid_onnx_names)
    t1 = time.perf_counter_ns()
    injected_stats = _build_tensor_quant_overrides(
        onnx_model,
        onnx_name_to_dists,
        onnx_name_to_axes,
        func_output_stats,
        input_stats,
        lambd,
        constant_onnx_names,
        calibrated_onnx_names,
        range_mode,
        percentile,
        locked_tensors,
    )
    t2 = time.perf_counter_ns()
    return injected_stats, [
        ("collect_dists", _ns_to_ms(t0, t1)),
        ("build_overrides", _ns_to_ms(t1, t2)),
    ]


def _quantize_with_vishap_dists(
    input_model: str,
    output_model: str,
    onnx_model: onnx.ModelProto,
    annotated_module,
    input_stats: list[Stats],
    lambd: int,
    calibrated_onnx_names: set[str] | None = None,
    range_mode: RangeMode = RangeMode.lambda_clip,
    percentile: float = 50,
) -> list[tuple[str, float]]:
    injected_stats, stages = _vishap_quant_overrides(
        onnx_model,
        annotated_module,
        input_stats,
        lambd,
        calibrated_onnx_names,
        range_mode,
        percentile,
    )
    t2 = time.perf_counter_ns()
    _quantize_from_overrides(input_model, output_model, injected_stats)
    t3 = time.perf_counter_ns()
    return [*stages, ("quantize", _ns_to_ms(t2, t3))]


def _vishap_quantization(
    input_model: str,
    output_model: str,
    input_file: str | None = None,
    input_dir: str | None = None,
    lambd: int = 5,
    num_inputs: int = 1,
    seed: int | None = None,
    perf_out: str = "perf_info.csv",
    range_mode: RangeMode = RangeMode.lambda_clip,
    percentile: float = 50,
) -> LogicalResult:
    t0 = time.perf_counter_ns()
    onnx_model, mlir_module = _load_onnx_and_mlir(input_model)
    t1 = time.perf_counter_ns()
    if onnx_model is None or mlir_module is None:
        logger.error("Could not load ONNX model")
        return LogicalResult.FAILURE

    batches = _load_method_inputs(
        input_model, input_file, input_dir, num_inputs, seed, "Vishap"
    )
    if batches is None:
        return LogicalResult.FAILURE

    logger.info(
        "Using stats from %d concatenated input(s) for Vishap analysis", len(batches)
    )
    input_stats = _input_stats_from_batches(batches)

    try:
        t2 = time.perf_counter_ns()
        annotated_module = _annotate_module_in_memory(mlir_module, input_stats)
        t3 = time.perf_counter_ns()
    except Exception as e:
        logger.error("Vishap quantization failed")
        logger.error(e)
        return LogicalResult.FAILURE

    quant_stages = _quantize_with_vishap_dists(
        input_model,
        output_model,
        onnx_model,
        annotated_module,
        input_stats,
        lambd,
        range_mode=range_mode,
        percentile=percentile,
    )

    perf_method = "vishap_agg" if num_inputs > 1 else "vishap"
    if range_mode is RangeMode.percentile:
        perf_method += f"_p{percentile}"
    _save_perf_stages(
        perf_out,
        Path(input_model).stem,
        perf_method,
        [
            ("onnx2mlir", _ns_to_ms(t0, t1)),
            ("collect_inputs", _ns_to_ms(t1, t2)),
            ("static_analysis", _ns_to_ms(t2, t3)),
            *quant_stages,
        ],
    )

    return LogicalResult.SUCCESS


def _module_bytecode(mlir_module) -> bytes:
    buf = io.BytesIO()
    mlir_module.operation.write_bytecode(file=buf)
    return buf.getvalue()


def _mixed_quantization(
    input_model: str,
    output_model: str,
    input_file: str | None = None,
    input_dir: str | None = None,
    lambd: int = 5,
    num_inputs: int = 1,
    preload_ratio: float | None = None,
    vishap_op_type: str | None = None,
    static_depth: int | None = None,
    seed: int | None = None,
    perf_out: str = "perf_info.csv",
    range_mode: RangeMode = RangeMode.lambda_clip,
    percentile: float = 50,
):
    from mlir_runtime import attach_tensor_stats_to_ops, collect_mixed_float_ops

    t0 = time.perf_counter_ns()
    onnx_model, mlir_module = _load_onnx_and_mlir(input_model)
    t1 = time.perf_counter_ns()
    if onnx_model is None:
        logger.error("Could not load ONNX model")
        return LogicalResult.FAILURE

    batches = _load_method_inputs(
        input_model, input_file, input_dir, num_inputs, seed, "Mixed"
    )
    if batches is None:
        return LogicalResult.FAILURE

    logger.info(
        "Using stats from %d concatenated input(s) for mixed analysis", len(batches)
    )
    input_stats = _input_stats_from_batches(batches)

    try:
        t2 = time.perf_counter_ns()
        node_name_to_outputs, node_name_to_op_type = _onnx_node_maps(onnx_model)
        mixed_module = parse_vishap_module(_module_bytecode(mlir_module))
        if vishap_op_type is not None:
            candidate_types = set(
                mixed_candidate_op_types(
                    mixed_module, node_name_to_outputs, node_name_to_op_type
                )
            )
            if vishap_op_type not in _non_constant_onnx_op_types(onnx_model):
                logger.error(
                    "Unknown ONNX op type %s; not present among non-constant nodes",
                    vishap_op_type,
                )
                return
            if vishap_op_type not in candidate_types:
                logger.error(
                    "ONNX op type %s has no mixed float candidates for Vishap analysis",
                    vishap_op_type,
                )
                return
        vishap_names = None
        if static_depth is not None:
            candidate_depths = _longest_path_depths(
                mixed_candidate_predecessors(mixed_module, node_name_to_outputs)
            )
            calibrated_names = _names_at_static_depth(
                candidate_depths, static_depth
            )
            vishap_names = set(candidate_depths) - calibrated_names
            calibrated_depths = ", ".join(
                str(depth)
                for depth in sorted(
                    {candidate_depths[name] for name in calibrated_names}
                )
            )
            logger.info(
                "Static depth %d: calibrating %d/%d candidate ops at depths %s",
                static_depth,
                len(calibrated_names),
                len(candidate_depths),
                calibrated_depths or "(none)",
            )
            if not calibrated_names:
                logger.warning(
                    "No mixed candidate reaches static depth %d; "
                    "every candidate is left for Vishap",
                    static_depth,
                )
        # The pool is the producers the calibrator would measure. Choosing it
        # before the run lets statically analyzed activations skip ReduceMin
        # and ReduceMax, which is what allows those nodes to fuse.
        prepared, _ = _prepare_calibration_model(input_model)
        candidate_tensors = _float_activation_tensor_names(prepared)
        # Ops dropped by ONNX-to-MLIR (identity Concat, Pad folded into a pool)
        # have no Vishap distribution. They stay in the measured set.
        analyzable_names = set(
            collect_mixed_float_ops(mixed_module, node_name_to_outputs)
        )
        selection = _select_preload(
            prepared,
            candidate_tensors,
            preload_ratio,
            seed,
            vishap_op_type,
            vishap_names,
            analyzable_names,
        )
        tensors_to_skip = candidate_tensors - selection.keep
        logger.info(
            "Running ORT MinMax calibration on %d input(s)",
            len(batches),
        )
        if tensors_to_skip:
            logger.info(
                "Leaving %d activation(s) out of calibration so statically "
                "analyzed nodes can fuse",
                len(tensors_to_skip),
            )
        calib_model, calibrated = _collect_minmax_ranges(
            input_model, batches, tensors_to_skip
        )
        # Mean and variance still seed the selected MLIR ops. With static
        # nodes omitted, only the calibrated outputs are exposed and the
        # session may fuse the rest. Quant ranges come from `calibrated`.
        if tensors_to_skip:
            tensor_stats = _collect_ort_tensor_stats(
                calib_model,
                batches,
                tensors_to_expose=selection.selected_outputs,
                enable_fusion=True,
            )
        else:
            tensor_stats = _collect_ort_tensor_stats(onnx_model, batches)
        attach_names = _mlir_attach_names(
            selection.selected_outputs, node_name_to_outputs
        )
        preloaded_names = attach_tensor_stats_to_ops(
            mixed_module, tensor_stats, node_name_to_outputs, attach_names
        )
        t3 = time.perf_counter_ns()
        logger.info(
            "Selected %d/%d calibration ops; seeded %d MLIR ops",
            selection.selected_count,
            selection.pool_count,
            len(preloaded_names),
        )

        t4 = time.perf_counter_ns()
        annotation_failed = False
        try:
            annotate_distributions_on_module(
                mixed_module, input_stats, annotate_axes=True
            )
        except Exception as exc:
            # Nothing is left for Vishap, so the calibrator ranges are enough.
            if selection.unselected_count:
                raise
            annotation_failed = True
            logger.warning(
                "Distribution annotation failed with every calibrator tensor "
                "selected; quantizing from calibration ranges (%s)",
                exc,
            )
        t5 = time.perf_counter_ns()

        if annotation_failed or selection.unselected_count == 0:
            overrides: dict[str, list[dict[str, np.float32]]] = {}
            override_stages = [("collect_dists", 0.0), ("build_overrides", 0.0)]
        else:
            overrides, override_stages = _vishap_quant_overrides(
                onnx_model,
                mixed_module,
                input_stats,
                lambd,
                preloaded_names,
                range_mode,
                percentile,
                selection.keep,
            )
        tq = time.perf_counter_ns()
        merged = _merge_activation_ranges(calibrated, overrides, selection.keep)
        missing_ranges = sorted(tensors_to_skip - set(merged.data))
        if missing_ranges:
            logger.warning(
                "No measured or finite Vishap range for %d activation(s): %s",
                len(missing_ranges),
                ", ".join(missing_ranges),
            )
        _qdq_quantize(calib_model, merged, output_model)
        quant_stages = [
            *override_stages,
            ("quantize", _ns_to_ms(tq, time.perf_counter_ns())),
        ]
    except ImportError:
        logger.error(
            "Unable to import Vishap Python bindings. Make sure to build Vishap with bindings enabled."
        )
        return LogicalResult.FAILURE
    except Exception as e:
        logger.error("Mixed quantization failed")
        logger.error(e)
        return LogicalResult.FAILURE

    if vishap_op_type is not None:
        op_label = _sanitize_op_type_label(vishap_op_type)
        perf_method = (
            f"mixed_agg_op_{op_label}" if num_inputs > 1 else f"mixed_op_{op_label}"
        )
    elif static_depth is not None:
        perf_method = (
            f"mixed_agg_depth_{static_depth}"
            if num_inputs > 1
            else f"mixed_depth_{static_depth}"
        )
    else:
        assert preload_ratio is not None
        perf_method = (
            f"mixed_agg_{preload_ratio:.2f}"
            if num_inputs > 1
            else f"mixed_{preload_ratio:.2f}"
        )
    if range_mode is RangeMode.percentile:
        perf_method += f"_p{percentile}"
    _save_perf_stages(
        perf_out,
        Path(input_model).stem,
        perf_method,
        [
            ("onnx2mlir", _ns_to_ms(t0, t1)),
            ("collect_inputs", _ns_to_ms(t1, t2)),
            ("probe", _ns_to_ms(t2, t3)),
            ("static_analysis", _ns_to_ms(t4, t5)),
            *quant_stages,
        ],
    )

    return LogicalResult.SUCCESS


def _calibration_quantization(
    input_model: str,
    output_model: str,
    input_dir: str,
    seed: int | None = None,
    perf_out: str = "perf_info.csv",
) -> LogicalResult:
    t0 = time.perf_counter_ns()
    quantize_static(
        model_input=input_model,
        model_output=output_model,
        calibration_data_reader=RealDataCalibrationReader(
            input_model, load_inputs_for_model(input_model, input_dir, seed=seed)
        ),
        quant_format=QuantFormat.QDQ,
        calibrate_method=CalibrationMethod.MinMax,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    t1 = time.perf_counter_ns()
    _save_perf_stages(
        perf_out,
        Path(input_model).stem,
        "calibration",
        [("quantization", _ns_to_ms(t0, t1))],
    )

    return LogicalResult.SUCCESS


def _random_quantization(
    input_model: str,
    output_model: str,
    seed: int | None = None,
    perf_out: str = "perf_info.csv",
) -> LogicalResult:
    t0 = time.perf_counter_ns()
    onnx_model = onnx.load(input_model)
    graph = onnx_model.graph

    input_names = [tensor.name for tensor in graph.input]
    tensor_names = [output for node in graph.node for output in node.output]

    rng = np.random.default_rng(seed)
    injected_stats = {}
    for tensor in input_names + tensor_names:
        bound1 = np.float32(rng.random())
        bound2 = np.float32(rng.random())
        injected_stats[tensor] = [
            {"rmin": min(bound1, bound2), "rmax": max(bound1, bound2)}
        ]
    t1 = time.perf_counter_ns()

    _quantize_from_overrides(input_model, output_model, injected_stats)
    t2 = time.perf_counter_ns()
    _save_perf_stages(
        perf_out,
        Path(input_model).stem,
        "random",
        [
            ("build_overrides", _ns_to_ms(t0, t1)),
            ("quantize", _ns_to_ms(t1, t2)),
        ],
    )

    return LogicalResult.SUCCESS


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize an ONNX network, using different approaches."
    )
    parser.add_argument(
        "input_model", type=str, help="input ONNX model", metavar="model_path"
    )
    parser.add_argument(
        "--method",
        type=QuantMethod,
        help="quantization method to use (default: %(default)s)",
        default=QuantMethod.vishap,
        choices=list(QuantMethod),
    )
    parser.add_argument(
        "--output-model",
        "-o",
        dest="output_model",
        type=str,
        # required=True,
        help="path to the output quantized ONNX model",
    )
    parser.add_argument(
        "--input-dir",
        dest="input_dir",
        type=str,
        help="path to the directory containing input data (images, .npy, or .npz)",
    )
    parser.add_argument(
        "--input-file",
        dest="input_file",
        type=str,
        help="path to a single input (image, .npy, or .npz)",
    )
    parser.add_argument(
        "--export-input",
        dest="export_input",
        type=str,
        help=(
            "write one preprocessed input to this path and exit "
            "(.npy for one tensor, .npz for several)"
        ),
    )
    parser.add_argument(
        "--lambda",
        dest="lambd",
        type=int,
        default=5,
        help="lambda value for Vishap quantization (default: %(default)s)",
    )
    parser.add_argument(
        "--range-mode",
        dest="range_mode",
        type=RangeMode,
        default=RangeMode.lambda_clip,
        choices=list(RangeMode),
        help=(
            "how per-axis slice estimates become a quant range: 'lambda' is "
            "the union of per-slice lambda clips, 'percentile' is one "
            "percentile of the raw per-slice min and max "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--range-percentile",
        dest="range_percentile",
        type=float,
        default=50,
        help=(
            "percentile in [0, 100] of the per-slice min and of the per-slice "
            "max when --range-mode is percentile; 50 is the median "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--num-inputs",
        dest="num_inputs",
        type=int,
        default=1,
        help=(
            "number of inputs to accumulate for Vishap or mixed distribution "
            "estimation (default: %(default)s; requires --input-dir when > 1)"
        ),
    )
    parser.add_argument(
        "--preload-ratio",
        dest="preload_ratio",
        type=float,
        default=None,
        help=(
            "fraction of non-constant ops that produce a calibrated activation "
            "to preload with ORT MinMax ranges for mixed quantization "
            "(in [0, 1]; mutually exclusive with --vishap-op-type and "
            "--static-depth)"
        ),
    )
    parser.add_argument(
        "--vishap-op-type",
        dest="vishap_op_type",
        type=str,
        default=None,
        help=(
            "ONNX op type to analyze with Vishap during mixed quantization; all "
            "other non-constant producers of calibrated activations are "
            "preloaded from ORT MinMax (mutually exclusive with --preload-ratio "
            "and --static-depth)"
        ),
    )
    parser.add_argument(
        "--static-depth",
        dest="static_depth",
        type=int,
        default=None,
        help=(
            "fixed hop distance for calibrated nodes in mixed quantization. "
            "The first calibrated node is this many candidate hops from the input, "
            "and later calibrated nodes have this many statically analyzed "
            "candidates between them (integer >= 1; mutually exclusive with "
            "--preload-ratio and --vishap-op-type)"
        ),
    )
    parser.add_argument(
        "--list-op-types",
        dest="list_op_types",
        action="store_true",
        help=(
            "print mixed float ONNX op types in the input graph (one per line) "
            "and exit"
        ),
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=0,
        help=(
            "RNG seed for input sampling, mixed op selection, and random quantization "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--perf-out",
        dest="perf_out",
        type=str,
        default="perf_info.csv",
        help="output file with quantization performance information",
    )

    return parser.parse_args()


def _main() -> LogicalResult:
    args = _parse_args()

    np.random.seed(args.seed)

    if args.list_op_types:
        return _list_mixed_op_types(args.input_model)

    if args.export_input:
        source = args.input_file or args.input_dir
        if not source:
            logger.error("--export-input requires --input-file or --input-dir")
            return LogicalResult.FAILURE
        export_model_input(args.input_model, source, args.export_input, seed=args.seed)
        return LogicalResult.SUCCESS

    if not args.output_model:
        logger.error(
            "--output-model is required unless using --export-input or --list-op-types"
        )
        return LogicalResult.FAILURE

    if args.num_inputs < 1:
        logger.error("--num-inputs must be >= 1")
        return LogicalResult.FAILURE

    if not 0 <= args.range_percentile <= 100:
        logger.error("--range-percentile must be in [0, 100]")
        return LogicalResult.FAILURE

    # TODO: use a Quantizer abstract class?
    match args.method:
        case QuantMethod.vishap:
            return _vishap_quantization(
                args.input_model,
                args.output_model,
                args.input_file,
                args.input_dir,
                args.lambd,
                args.num_inputs,
                args.seed,
                args.perf_out,
                args.range_mode,
                args.range_percentile,
            )
        case QuantMethod.mixed:
            selectors = [
                args.preload_ratio is not None,
                args.vishap_op_type is not None,
                args.static_depth is not None,
            ]
            if sum(selectors) != 1:
                logger.error(
                    "Mixed quantization requires exactly one of "
                    "--preload-ratio, --vishap-op-type, or --static-depth"
                )
                return LogicalResult.FAILURE
            if args.preload_ratio is not None and not 0.0 <= args.preload_ratio <= 1.0:
                logger.error("--preload-ratio must be in [0, 1]")
                return LogicalResult.FAILURE
            if args.static_depth is not None and args.static_depth < 1:
                logger.error("--static-depth must be >= 1")
                return LogicalResult.FAILURE
            return _mixed_quantization(
                args.input_model,
                args.output_model,
                args.input_file,
                args.input_dir,
                args.lambd,
                args.num_inputs,
                args.preload_ratio,
                args.vishap_op_type,
                args.static_depth,
                args.seed,
                args.perf_out,
                args.range_mode,
                args.range_percentile,
            )
        case QuantMethod.calibration:
            if not args.input_dir:
                logger.error("Calibration quantization requires --input-dir")
                return LogicalResult.FAILURE
            return _calibration_quantization(
                args.input_model,
                args.output_model,
                args.input_dir,
                args.seed,
                args.perf_out,
            )
        case QuantMethod.random:
            return _random_quantization(
                args.input_model, args.output_model, args.seed, args.perf_out
            )
        case _:
            logger.error(f"Invalid quantization method: {args.method}")
            return LogicalResult.FAILURE


if __name__ == "__main__":
    sys.exit(_main())
