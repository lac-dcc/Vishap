import argparse
import csv
import io
import logging
import numpy as np
import onnx
import onnxruntime as ort
import os
import sys
import time

from enum import Enum
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.calibrate import (
    CalibrationMethod,
    CalibrationDataReader,
    TensorData,
    TensorsData,
)
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
from onnxruntime.quantization.quant_utils import (
    load_model_with_shape_infer,
    update_opset_version,
)
from onnxruntime.quantization.registry import QDQRegistry, QLinearOpsRegistry
from pathlib import Path

from onnx2mlir import convert_onnx_to_mlir, load_onnx_model
from mlir_runtime import CONSTANT_OP_NAMES
from utils import (
    Stats,
    TMP_DIR,
    VISHAP_DIST_ATTR,
    annotate_distributions,
    annotate_distributions_on_module,
    get_array_stats,
    is_image_path,
    is_nchw_layout,
    parse_vishap_module,
    preprocess_image,
    spatial_size,
    collect_input_paths,
)

DEFAULT_SPATIAL_SIZE = 224

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

QUANT_TMP_DIR = TMP_DIR / "quantization"


class RealDataCalibrationReader(CalibrationDataReader):
    """
    A data reader that feeds real batches of data to the calibrator.
    """

    def __init__(self, model_path: str, data_batches: list[np.ndarray]):
        self.data_batches = data_batches
        self.iterator = iter(self.data_batches)

        # We need the exact input name the model expects
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = session.get_inputs()[0].name

    def get_next(self) -> dict:
        """
        ORT calls this method repeatedly until it returns None.
        Each dictionary returned represents one batch of calibration data.
        """
        try:
            # Get the next batch of data from our iterator
            batch = next(self.iterator)
            return {self.input_name: batch}
        except StopIteration:
            # Returning None tells the calibrator that we are out of data
            return None


class QuantMethod(Enum):
    calibration = "calibration"
    vishap = "vishap"
    mixed = "mixed"
    random = "random"

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


def get_model_input_spec(model_path: str) -> tuple[str, list[int], bool, np.dtype]:
    """Return (name, concrete_shape, is_nchw, dtype) for the model's first input."""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    raw_shape = list(inp.shape)

    shape: list[int] = []
    for i, dim in enumerate(raw_shape):
        if isinstance(dim, str) or dim is None:
            # Batch -> 1; spatial dims -> DEFAULT_SPATIAL_SIZE
            shape.append(1 if i == 0 else DEFAULT_SPATIAL_SIZE)
        else:
            shape.append(int(dim))

    if len(shape) != 4:
        raise ValueError(
            f"Expected a 4D image input for {model_path}, got shape {raw_shape}"
        )

    is_nchw = is_nchw_layout(shape)
    dtype = _ort_dtype_to_numpy(inp.type)
    logger.info(
        f"Model input '{inp.name}': shape={shape}, "
        f"layout={'NCHW' if is_nchw else 'NHWC'}, dtype={dtype}"
    )
    return inp.name, shape, is_nchw, dtype


def _shuffle_paths(paths: list, seed: int | None) -> list:
    """Return paths shuffled with seed, or unchanged when seed is None."""
    if seed is None or len(paths) <= 1:
        return paths
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    return [paths[i] for i in order]


def load_inputs_for_model(
    model_path: str,
    input_path: str | Path,
    max_images: int | None = None,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Load .npy tensors or images resized to the model's input shape."""
    path = Path(input_path)
    _, shape, is_nchw, dtype = get_model_input_spec(model_path)
    height, width = spatial_size(shape, is_nchw)

    if path.is_file() and path.suffix.lower() == ".npy":
        array = np.load(path)
        if tuple(array.shape) != tuple(shape):
            raise ValueError(
                f"Input {path} has shape {array.shape}, model expects {shape}"
            )
        return [array.astype(dtype, copy=False)]

    if path.is_dir():
        npy_files = sorted(path.glob("*.npy"))
        if npy_files and not any(
            is_image_path(p) for p in path.rglob("*") if p.is_file()
        ):
            npy_files = _shuffle_paths(npy_files, seed)
            if max_images is not None:
                npy_files = npy_files[:max_images]
            return [np.load(p) for p in npy_files]

    image_paths = _shuffle_paths(collect_input_paths(path), seed)
    if max_images is not None:
        image_paths = image_paths[:max_images]

    batches = [preprocess_image(p, height, width, is_nchw, dtype) for p in image_paths]
    logger.info(f"Loaded {len(batches)} input(s) at {height}x{width} for {model_path}")
    return batches


def export_model_input(
    model_path: str,
    input_path: str,
    output_npy: str,
    seed: int | None = None,
) -> None:
    """Write one preprocessed input tensor as .npy for compare_models."""
    batches = load_inputs_for_model(model_path, input_path, max_images=1, seed=seed)
    os.makedirs(os.path.dirname(os.path.abspath(output_npy)) or ".", exist_ok=True)
    np.save(output_npy, batches[0])
    logger.info(f"Exported input tensor of shape {batches[0].shape} to {output_npy}")


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


def _vishap_dist_to_tensor_quant(
    lambd, stats: Stats, *, use_real_minmax: bool = False
) -> list[dict[str, np.float32]]:
    min_est, max_est, mean_est, var_est = stats
    if use_real_minmax:
        lowest, highest = min_est, max_est
    else:
        lowest = max(min_est, mean_est - lambd * np.sqrt(var_est))
        highest = min(max_est, mean_est + lambd * np.sqrt(var_est))
    return [{"rmin": np.float32(lowest), "rmax": np.float32(highest)}]


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


def _parse_onnx_loc_name(location: str) -> str:
    """Strip loc("...") wrapper from an MLIR location string, if present."""
    if location.startswith('loc("') and location.endswith('")'):
        return location[5:-2]
    return location


def _collect_vishap_dists(
    mlir_module, valid_onnx_names: set[str]
) -> tuple[dict[str, list[Stats]], list[Stats], set[str]]:
    """Walk an annotated MLIR module and return node stats, entry output stats, and constant names.

    The constant-name set contains ONNX loc names whose MLIR op is in
    ``CONSTANT_OP_NAMES`` (``stablehlo.constant`` / ``arith.constant``).
    """

    onnx_name_to_dists: dict[str, list[Stats]] = {}
    func_output_stats: list[Stats] = []
    constant_onnx_names: set[str] = set()

    # Assume entry function is the first operation in the module's body
    entry_func = mlir_module.body.operations[0]
    for op in entry_func.regions[0].blocks[0]:
        if VISHAP_DIST_ATTR not in op.attributes:
            continue

        onnx_name = _parse_onnx_loc_name(str(op.location))
        if onnx_name not in valid_onnx_names:
            continue

        onnx_name_to_dists[onnx_name] = [
            tuple(np.float64(val) for val in dist)
            for dist in op.operation.attributes[VISHAP_DIST_ATTR]
        ]
        if op.name in CONSTANT_OP_NAMES:
            constant_onnx_names.add(onnx_name)

    if VISHAP_DIST_ATTR in entry_func.attributes:
        func_output_stats = [
            tuple(np.float64(val) for val in dist)
            for dist in entry_func.operation.attributes[VISHAP_DIST_ATTR]
        ]

    return onnx_name_to_dists, func_output_stats, constant_onnx_names


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
        return annotate_distributions(buf.getvalue(), input_stats)
    except ImportError:
        logger.error(
            "Unable to import Vishap Python bindings. Make sure to build Vishap with bindings enabled."
        )
        raise
    except Exception:
        raise


def _build_tensor_quant_overrides(
    onnx_model: onnx.ModelProto,
    onnx_name_to_dists: dict[str, list[Stats]],
    func_output_stats: list[Stats],
    input_stats: Stats,
    lambd: int,
    constant_onnx_names: set[str] | None = None,
) -> dict[str, list[dict[str, np.float32]]]:
    """Map Vishap distribution estimates to ORT TensorQuantOverrides keys."""
    # Map ONNX node names (from MLIR locs) to actual output tensor names.
    # TensorQuantOverrides keys must be tensor names, not node names / ":0" ports.
    node_name_to_outputs = {
        node.name: list(node.output) for node in onnx_model.graph.node if node.name
    }
    constant_names = constant_onnx_names or set()

    injected_stats: dict[str, list[dict[str, np.float32]]] = {}
    for node_name, out_stats in onnx_name_to_dists.items():
        outputs = node_name_to_outputs.get(node_name, [])
        use_real_minmax = node_name in constant_names
        for idx, stats in enumerate(out_stats):
            if idx >= len(outputs):
                logger.warning(
                    "Skipping dist for %s output %d: node has only %d output(s)",
                    node_name,
                    idx,
                    len(outputs),
                )
                continue
            injected_stats[outputs[idx]] = _vishap_dist_to_tensor_quant(
                lambd, stats, use_real_minmax=use_real_minmax
            )

    input_name = onnx_model.graph.input[0].name
    output_names = [out.name for out in onnx_model.graph.output]
    injected_stats[input_name] = _vishap_dist_to_tensor_quant(lambd, input_stats)
    for output_name, stats in zip(output_names, func_output_stats):
        injected_stats[output_name] = _vishap_dist_to_tensor_quant(lambd, stats)

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


def _quantize_from_overrides(
    input_model: str,
    output_model: str,
    injected_stats: dict[str, list[dict[str, np.float32]]],
):
    """QDQ-quantize using supplied rmin/rmax, without a calibration inference pass."""
    model = load_model_with_shape_infer(Path(input_model))
    model = update_opset_version(model, QuantType.QInt8)
    tensors_range = _tensors_data_from_overrides(model, injected_stats)
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


def _load_onnx_and_mlir(input_model: str):
    try:
        onnx_model = load_onnx_model(input_model)
        mlir_module = convert_onnx_to_mlir(onnx_model)
        return onnx_model, mlir_module
    except Exception:
        logger.error(f"Failed to load ONNX model {input_model}")
        return None, None


def _quantize_with_vishap_dists(
    input_model: str,
    output_model: str,
    onnx_model: onnx.ModelProto,
    annotated_module,
    input_stats: Stats,
    lambd: int,
) -> list[tuple[str, float]]:
    t0 = time.perf_counter_ns()
    valid_onnx_names = set(_get_onnx_node_names(onnx_model))
    onnx_name_to_dists, func_output_stats, constant_onnx_names = _collect_vishap_dists(
        annotated_module, valid_onnx_names
    )
    t1 = time.perf_counter_ns()
    injected_stats = _build_tensor_quant_overrides(
        onnx_model,
        onnx_name_to_dists,
        func_output_stats,
        input_stats,
        lambd,
        constant_onnx_names,
    )
    t2 = time.perf_counter_ns()
    _quantize_from_overrides(input_model, output_model, injected_stats)
    t3 = time.perf_counter_ns()
    return [
        ("collect_dists", _ns_to_ms(t0, t1)),
        ("build_overrides", _ns_to_ms(t1, t2)),
        ("quantize", _ns_to_ms(t2, t3)),
    ]


def _vishap_quantization(
    input_model: str,
    output_model: str,
    input_file: str | None = None,
    input_dir: str | None = None,
    lambd: int = 5,
    num_inputs: int = 1,
    seed: int | None = None,
    perf_out: str = "perf_info.csv",
):
    t0 = time.perf_counter_ns()
    onnx_model, mlir_module = _load_onnx_and_mlir(input_model)
    t1 = time.perf_counter_ns()
    if onnx_model is None or mlir_module is None:
        return

    if input_file:
        if num_inputs > 1:
            logger.error("--num-inputs > 1 requires --input-dir, not --input-file")
            return
        input_source = input_file
    elif input_dir:
        input_source = input_dir
    else:
        logger.error("Vishap quantization requires --input-file or --input-dir")
        return

    batches = load_inputs_for_model(
        input_model, input_source, max_images=num_inputs, seed=seed
    )
    if not batches:
        logger.error(f"No inputs loaded from {input_source}")
        return
    if len(batches) < num_inputs:
        logger.warning(
            "Requested %d input(s) but only loaded %d from %s",
            num_inputs,
            len(batches),
            input_source,
        )

    logger.info(
        "Using stats from %d concatenated input(s) for Vishap analysis", len(batches)
    )
    input_stats = get_array_stats(
        batches[0] if len(batches) == 1 else np.concatenate(batches, axis=0)
    )

    try:
        # We assume one input tensor per model
        t2 = time.perf_counter_ns()
        annotated_module = _annotate_module_in_memory(mlir_module, [input_stats])
        t3 = time.perf_counter_ns()
    except Exception as e:
        logger.error("Vishap quantization failed")
        logger.error(e)
        return

    quant_stages = _quantize_with_vishap_dists(
        input_model, output_model, onnx_model, annotated_module, input_stats, lambd
    )

    perf_method = "vishap_agg" if num_inputs > 1 else "vishap"
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
    preload_ratio: float = 0.5,
    seed: int | None = None,
    perf_out: str = "perf_info.csv",
):
    from mlir_runtime import (
        attach_probe_dists_to_ops,
        collect_all_float_probe_targets,
        io_arrays_from_entry_func,
        parse_probe_report,
        run_probed_network,
    )

    t0 = time.perf_counter_ns()
    onnx_model, mlir_module = _load_onnx_and_mlir(input_model)
    t1 = time.perf_counter_ns()
    if onnx_model is None:
        return

    if input_file:
        input_source = input_file
    elif input_dir:
        input_source = input_dir
    else:
        logger.error("Mixed quantization requires --input-file or --input-dir")
        return None

    batches = load_inputs_for_model(input_model, input_source, max_images=1, seed=seed)
    if not batches or batches[0] is None:
        logger.error(f"No inputs loaded from {input_source}")
        return None

    input_stats = get_array_stats(batches[0])
    bytecode = _module_bytecode(mlir_module)

    try:
        t2 = time.perf_counter_ns()
        probe_module = parse_vishap_module(bytecode)
        targets = collect_all_float_probe_targets(probe_module)
        func = probe_module.body.operations[0]
        func_name = func.attributes["sym_name"].value
        input_arrays, output_arrays = io_arrays_from_entry_func(
            probe_module, [batches[0]]
        )
        logger.info(
            "Running probed network (%d float ops) to collect telemetry",
            len(targets),
        )
        run_probed_network(
            probe_module,
            input_arrays,
            output_arrays,
            func_name,
            all_float_ops=True,
        )
        probe_dists = parse_probe_report()

        mixed_module = parse_vishap_module(bytecode)
        n_preloaded = attach_probe_dists_to_ops(
            mixed_module, targets, probe_dists, preload_ratio, seed
        )
        t3 = time.perf_counter_ns()
        logger.info("Preloaded probe distributions on %d ops", n_preloaded)

        t4 = time.perf_counter_ns()
        annotate_distributions_on_module(mixed_module, [input_stats])
        t5 = time.perf_counter_ns()

        quant_stages = _quantize_with_vishap_dists(
            input_model,
            output_model,
            onnx_model,
            mixed_module,
            input_stats,
            lambd,
        )
    except ImportError:
        logger.error(
            "Unable to import Vishap Python bindings. Make sure to build Vishap with bindings enabled."
        )
        return
    except Exception as e:
        logger.error("Mixed quantization failed")
        logger.error(e)
        return

    _save_perf_stages(
        perf_out,
        Path(input_model).stem,
        f"mixed_{preload_ratio:.2f}",
        [
            ("onnx2mlir", _ns_to_ms(t0, t1)),
            ("collect_inputs", _ns_to_ms(t1, t2)),
            ("probe", _ns_to_ms(t2, t3)),
            ("static_analysis", _ns_to_ms(t4, t5)),
            *quant_stages,
        ],
    )


def _calibration_quantization(
    input_model: str,
    output_model: str,
    input_dir: str,
    seed: int | None = None,
    perf_out: str = "perf_info.csv",
):
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


def _random_quantization(
    input_model: str,
    output_model: str,
    seed: int | None = None,
    perf_out: str = "perf_info.csv",
):
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
        help="path to the directory containing input data (images or .npy)",
    )
    parser.add_argument(
        "--input-file",
        dest="input_file",
        type=str,
        help="path to a single input (.npy or image)",
    )
    parser.add_argument(
        "--export-input",
        dest="export_input",
        type=str,
        help="write one preprocessed input tensor to this .npy path and exit",
    )
    parser.add_argument(
        "--lambda",
        dest="lambd",
        type=int,
        default=5,
        help="lambda value for Vishap quantization (default: %(default)s)",
    )
    parser.add_argument(
        "--num-inputs",
        dest="num_inputs",
        type=int,
        default=1,
        help=(
            "number of inputs to concatenate for Vishap distribution estimation "
            "(default: %(default)s; requires --input-dir when > 1)"
        ),
    )
    parser.add_argument(
        "--preload-ratio",
        dest="preload_ratio",
        type=float,
        default=None,
        help=(
            "fraction of non-constant float ops to preload with probe telemetry "
            "for mixed quantization (required for --method mixed, in [0, 1])"
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


def _main():
    args = _parse_args()

    np.random.seed(args.seed)

    if args.export_input:
        source = args.input_file or args.input_dir
        if not source:
            logger.error("--export-input requires --input-file or --input-dir")
            return 1
        export_model_input(args.input_model, source, args.export_input, seed=args.seed)
        return 0

    if not args.output_model:
        logger.error("--output-model is required unless using --export-input")
        return 1

    if args.num_inputs < 1:
        logger.error("--num-inputs must be >= 1")
        return 1

    # TODO: use a Quantizer abstract class?
    match args.method:
        case QuantMethod.vishap:
            _vishap_quantization(
                args.input_model,
                args.output_model,
                args.input_file,
                args.input_dir,
                args.lambd,
                args.num_inputs,
                args.seed,
                args.perf_out,
            )
        case QuantMethod.mixed:
            if args.preload_ratio is None:
                logger.error("Mixed quantization requires --preload-ratio")
                return 1
            if not 0.0 <= args.preload_ratio <= 1.0:
                logger.error("--preload-ratio must be in [0, 1]")
                return 1
            if args.num_inputs > 1:
                logger.error(
                    "Mixed quantization uses a single input; do not set --num-inputs > 1"
                )
                return 1
            _mixed_quantization(
                args.input_model,
                args.output_model,
                args.input_file,
                args.input_dir,
                args.lambd,
                args.preload_ratio,
                args.seed,
                args.perf_out,
            )
        case QuantMethod.calibration:
            if not args.input_dir:
                logger.error("Calibration quantization requires --input-dir")
                return 1
            _calibration_quantization(
                args.input_model,
                args.output_model,
                args.input_dir,
                args.seed,
                args.perf_out,
            )
        case QuantMethod.random:
            _random_quantization(
                args.input_model, args.output_model, args.seed, args.perf_out
            )
        case _:
            logger.error(f"Invalid quantization method: {args.method}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(_main())
