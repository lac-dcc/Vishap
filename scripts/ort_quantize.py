import argparse
import csv
import logging
import numpy as np
import onnx
import onnxruntime as ort
import os
import sys
import time

from enum import Enum
from onnxruntime.quantization import QuantType, quantize_static
from onnxruntime.quantization.calibrate import (
    CalibrationMethod,
    CalibrationDataReader,
    # TODO: use ort-nightly to test CalibrationCache?
    # TensorsData,
    # TensorData,
    # save_tensors_data
)
from pathlib import Path

# The annotated file is in the stablehlo dialect, which is not part of
# the upstream MLIR Python bindings. Parse it with the torch-mlir
# bindings instead.
# FIXME: use our own Python bindings
from torch_mlir.ir import Module as TorchMlirModule

from onnx2mlir import convert_onnx_to_mlir, load_onnx_model
from utils import (
    Stats,
    TMP_DIR,
    VISHAP_DIST_ATTR,
    get_array_stats,
    is_image_path,
    is_nchw_layout,
    make_stablehlo_context,
    preprocess_image,
    run_vishap,
    spatial_size,
)

DEFAULT_SPATIAL_SIZE = 224

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

QUANT_TMP_DIR = TMP_DIR / "quantization"


class SmartDummyDataReader(CalibrationDataReader):
    """
    A dummy data reader that doesn't read any data
    """

    def __init__(self, model_path: str):
        self.yielded = False
        self.dummy_data = {}

        # 1. Briefly load the model to inspect its required inputs
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        for inp in session.get_inputs():
            # 2. Handle dynamic dimensions (e.g., 'batch_size', 'None', or 'unk__1')
            # We replace any unknown text/None dimensions with 1 so numpy can build the array.
            shape = [
                1 if (isinstance(dim, str) or dim is None) else dim for dim in inp.shape
            ]

            # 3. Figure out the numpy data type (defaulting to float32)
            if inp.type == "tensor(int64)":
                dtype = np.int64
            elif inp.type == "tensor(int32)":
                dtype = np.int32
            elif inp.type == "tensor(float16)":
                dtype = np.float16
            else:
                dtype = np.float32

            # 4. Create the dummy zero-tensor
            self.dummy_data[inp.name] = np.zeros(shape, dtype=dtype)
            logger.info(f"Created dummy input: '{inp.name}' of shape {shape} ({dtype})")

    def get_next(self) -> dict:
        if not self.yielded:
            self.yielded = True
            return self.dummy_data
        # Return None on the second call to end the calibration loop cleanly
        return None


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


def _collect_image_paths(path: Path) -> list[Path]:
    if path.is_file():
        if not is_image_path(path):
            raise ValueError(f"Not a supported image file: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    images = sorted(p for p in path.rglob("*") if p.is_file() and is_image_path(p))
    if not images:
        raise FileNotFoundError(f"No images found under {path}")
    return images


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

    image_paths = _shuffle_paths(_collect_image_paths(path), seed)
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


def _vishap_dist_to_tensor_quant(lambd, stats: Stats) -> list[dict[str, np.float32]]:
    min_est, max_est, mean_est, var_est = stats
    lowest = max(min_est, mean_est - lambd * np.sqrt(var_est))
    highest = min(max_est, mean_est + lambd * np.sqrt(var_est))
    return [{"rmin": np.float32(lowest), "rmax": np.float32(highest)}]


def _save_perf_info(out_file: str, time_ms: int, method: str):
    is_empty = not os.path.isfile(out_file) or os.path.getsize(out_file) == 0
    with open(out_file, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if is_empty:
            writer.writerow(["time_ms", "method"])
        writer.writerow([time_ms, method])


def _parse_onnx_loc_name(location: str) -> str:
    """Strip loc("...") wrapper from an MLIR location string, if present."""
    if location.startswith('loc("') and location.endswith('")'):
        return location[5:-2]
    return location


def _collect_vishap_dists(
    dist_file: str, valid_onnx_names: set[str]
) -> tuple[dict[str, list[Stats]], list[Stats]]:
    """Parse annotated MLIR and return (node_name -> output stats, entry output stats)."""
    onnx_name_to_dists: dict[str, list[Stats]] = {}
    entry_output_stats: list[Stats] = []

    with make_stablehlo_context() as ctx:
        module = TorchMlirModule.parseFile(dist_file, ctx)

        # Assume entry function is the first operation in the module's body
        entry_func = module.body.operations[0]
        for op in entry_func.body.blocks[0]:
            if VISHAP_DIST_ATTR not in op.attributes:
                continue

            onnx_name = _parse_onnx_loc_name(str(op.location))
            if onnx_name not in valid_onnx_names:
                continue

            onnx_name_to_dists[onnx_name] = [
                tuple(np.float64(val) for val in dist)
                for dist in op.operation.attributes[VISHAP_DIST_ATTR]
            ]

        if VISHAP_DIST_ATTR in entry_func.attributes:
            entry_output_stats = [
                tuple(np.float64(val) for val in dist)
                for dist in entry_func.operation.attributes[VISHAP_DIST_ATTR]
            ]

    return onnx_name_to_dists, entry_output_stats


def _build_tensor_quant_overrides(
    onnx_model: onnx.ModelProto,
    onnx_name_to_dists: dict[str, list[Stats]],
    entry_output_stats: list[Stats],
    input_stats: Stats,
    lambd: int,
) -> dict[str, list[dict[str, np.float32]]]:
    """Map Vishap distribution estimates to ORT TensorQuantOverrides keys."""
    # Map ONNX node names (from MLIR locs) to actual output tensor names.
    # TensorQuantOverrides keys must be tensor names, not node names / ":0" ports.
    node_name_to_outputs = {
        node.name: list(node.output) for node in onnx_model.graph.node if node.name
    }

    injected_stats: dict[str, list[dict[str, np.float32]]] = {}
    for node_name, out_stats in onnx_name_to_dists.items():
        outputs = node_name_to_outputs.get(node_name, [])
        for idx, stats in enumerate(out_stats):
            if idx >= len(outputs):
                logger.warning(
                    "Skipping dist for %s output %d: node has only %d output(s)",
                    node_name,
                    idx,
                    len(outputs),
                )
                continue
            injected_stats[outputs[idx]] = _vishap_dist_to_tensor_quant(lambd, stats)

    input_name = onnx_model.graph.input[0].name
    output_names = [out.name for out in onnx_model.graph.output]
    injected_stats[input_name] = _vishap_dist_to_tensor_quant(lambd, input_stats)
    for output_name, stats in zip(output_names, entry_output_stats):
        injected_stats[output_name] = _vishap_dist_to_tensor_quant(lambd, stats)

    return injected_stats


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
    output_mlir = QUANT_TMP_DIR / os.path.basename(input_model).replace(
        ".onnx", ".mlir"
    )
    os.makedirs(QUANT_TMP_DIR, exist_ok=True)

    start = time.perf_counter_ns()
    try:
        onnx_model = load_onnx_model(input_model)
        convert_onnx_to_mlir(onnx_model, output_file=output_mlir)
    except:
        # TODO: how to report errors on main?
        logger.error(f"Failed to load ONNX model {input_model}")
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
        # TODO: enable python bindings
        # We assume one input tensor per model
        dist_file = run_vishap(
            output_mlir,
            [input_stats],
            QUANT_TMP_DIR,
        )
    except Exception as e:
        logger.error("Vishap quantization failed")
        logger.error(e)
        return

    valid_onnx_names = set(_get_onnx_node_names(onnx_model))
    onnx_name_to_dists, entry_output_stats = _collect_vishap_dists(
        dist_file, valid_onnx_names
    )
    injected_stats = _build_tensor_quant_overrides(
        onnx_model,
        onnx_name_to_dists,
        entry_output_stats,
        input_stats,
        lambd,
    )

    extra_options = {"TensorQuantOverrides": injected_stats}
    quantize_static(
        model_input=input_model,
        model_output=output_model,
        calibration_data_reader=SmartDummyDataReader(input_model),
        calibrate_method=CalibrationMethod.MinMax,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        # Override quantization parameters with Vishap estimations
        extra_options=extra_options,
    )
    end = time.perf_counter_ns()
    perf_method = "vishap_agg" if num_inputs > 1 else "vishap"
    _save_perf_info(perf_out, (end - start) / 1_000_000, perf_method)


def _calibration_quantization(
    input_model: str,
    output_model: str,
    input_dir: str,
    seed: int | None = None,
    perf_out: str = "perf_info.csv",
):
    start = time.perf_counter_ns()
    quantize_static(
        model_input=input_model,
        model_output=output_model,
        calibration_data_reader=RealDataCalibrationReader(
            input_model, load_inputs_for_model(input_model, input_dir, seed=seed)
        ),
        calibrate_method=CalibrationMethod.MinMax,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    end = time.perf_counter_ns()
    _save_perf_info(perf_out, (end - start) / 1_000_000, "calibration")


def _random_quantization(
    input_model: str,
    output_model: str,
    seed: int | None = None,
    perf_out: str = "perf_info.csv",
):
    start = time.perf_counter_ns()
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

    extra_options = {"TensorQuantOverrides": injected_stats}
    quantize_static(
        model_input=input_model,
        model_output=output_model,
        calibration_data_reader=SmartDummyDataReader(input_model),
        calibrate_method=CalibrationMethod.MinMax,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        # Override quantization parameters with Vishap estimations
        extra_options=extra_options,
    )
    end = time.perf_counter_ns()
    _save_perf_info(perf_out, (end - start) / 1_000_000, "random")


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
        "--seed",
        dest="seed",
        type=int,
        default=0,
        help=(
            "RNG seed for input sampling and random quantization "
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
