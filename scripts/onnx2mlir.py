"""Convert ONNX model into an MLIR file, using mainly upstream dialects.

This script uses Vishap's embedded ONNX importer, which comes from torch-mlir.
The importer produces the torch dialect, which is then lowered to StableHLO.

This script can be called directly via command line, but you can also use it
as a module to import the `convert_onnx_to_mlir` function.

Usage:
    onnx2mlir.py [-h] [-o FILENAME]
                      [--emit-bytecode]
                      [--elide-constants-if-larger N]
                      model_path

Arguments:
    model_path
        input ONNX model path
    -o FILENAME
        output filename
    --emit-bytecode
        emit bytecode when generating output (default: False)
    --elide-constants-if-larger N
        elide constants in emitted IR if larger than specified value
    -h, --help
        show help message and exit
"""

import argparse
import logging
import numpy as np
import onnx
import sys

from utils import (
    ensure_vishap_bindings_on_path,
    make_vishap_context,
    run_vishap_pipeline,
)

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    ensure_vishap_bindings_on_path()
    from vishap.extras import onnx_importer
except Exception:
    logger.error(
        "Vishap bindings are not available. Build Vishap with MLIR_ENABLE_BINDINGS_PYTHON."
    )
    exit(1)

# torch-mlir ONNX converters are version-gated. MaxPool is registered at
# sinceVersion=12, BatchNormalization at 15 and GreaterOrEqual at 16, so older
# models fail with `failed to legalize operation 'torch.operator'`.
_TORCH_MLIR_TARGET_OPSET = 16


def _dump_module(module, output_filename, binary_format=False, max_constant=None):
    if binary_format:
        assert output_filename, "Output file must be specified when using binary format"

    mode = "wb" if binary_format else "wt"
    output_file = open(output_filename, mode) if output_filename else None

    if binary_format:
        module.write_bytecode(file=output_file)
    else:
        module.print(
            file=output_file,
            large_elements_limit=max_constant,
            assume_verified=True,
            enable_debug_info=True,
        )

    if output_file:
        output_file.close()


def _onnx_default_opset(model: onnx.ModelProto) -> int | None:
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return opset.version
    return None


def _upgrade_onnx_opset_for_torch_mlir(model: onnx.ModelProto) -> onnx.ModelProto:
    """Rewrite the model to an opset supported by torch-mlir's ONNX converters."""
    current = _onnx_default_opset(model)
    if current is None or current >= _TORCH_MLIR_TARGET_OPSET:
        return model

    logger.info(
        "Upgrading ONNX opset from %s to %s for torch-mlir conversion",
        current,
        _TORCH_MLIR_TARGET_OPSET,
    )
    return onnx.version_converter.convert_version(model, _TORCH_MLIR_TARGET_OPSET)


# Ops safe to evaluate eagerly when every input is a known constant. This is
# deliberately a small whitelist of scalar/shape arithmetic.
_FOLDABLE_OPS = frozenset(
    {
        "Abs",
        "Add",
        "Cast",
        "Ceil",
        "Concat",
        "Div",
        "Equal",
        "Floor",
        "Gather",
        "Greater",
        "Less",
        "Max",
        "Min",
        "Mod",
        "Mul",
        "Neg",
        "Reshape",
        "Round",
        "Slice",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Unsqueeze",
    }
)
# Constant to control the maximum acceptable size for constant folding
_MAX_FOLD_ELEMENTS = 1024


def _constant_node_value(node: onnx.NodeProto) -> np.ndarray | None:
    """Return the numpy value of an ONNX ``Constant`` node, or None."""

    for attr in node.attribute:
        if attr.name == "value":
            return onnx.numpy_helper.to_array(attr.t)
        if attr.name == "value_int":
            return np.array(attr.i, dtype=np.int64)
        if attr.name == "value_ints":
            return np.array(list(attr.ints), dtype=np.int64)
        if attr.name == "value_float":
            return np.array(attr.f, dtype=np.float32)
        if attr.name == "value_floats":
            return np.array(list(attr.floats), dtype=np.float32)
    return None


def _fold_constant_subgraphs(model: onnx.ModelProto) -> int:
    """Fold small constant-only subgraphs (e.g. computed reshape targets).

    ONNX shape inference does not evaluate chains like
    ``Ceil -> Mul -> Cast -> Div`` on constants, so the tensors they feed
    get symbolic dims. torch-mlir then emits ``torch.runtime.assert`` ops on
    those dims that cannot be folded, and the StableHLO conversion rejects
    them.

    Returns the number of nodes replaced by constants.
    """
    from onnx import numpy_helper

    graph = model.graph
    graph_inputs = {i.name for i in graph.input}
    values: dict[str, np.ndarray] = {}
    for init in graph.initializer:
        if init.name in graph_inputs:
            # May be overridden at runtime
            continue

        if int(np.prod(init.dims, dtype=np.int64)) <= _MAX_FOLD_ELEMENTS:
            values[init.name] = numpy_helper.to_array(init)

    opset = _onnx_default_opset(model)
    opsets = {"": opset} if opset is not None else None
    folded = 0
    for idx, node in enumerate(graph.node):
        # Record values for constants
        if node.op_type == "Constant":
            value = _constant_node_value(node)
            if value is not None and value.size <= _MAX_FOLD_ELEMENTS:
                values[node.output[0]] = value
            continue

        # Skip non-foldable ops
        if (
            node.domain not in ("", "ai.onnx")
            or node.op_type not in _FOLDABLE_OPS
            or len(node.output) != 1
            or not node.input
            or any(i == "" or i not in values for i in node.input)
        ):
            continue

        try:
            result = onnx.reference.ReferenceEvaluator(node, opsets=opsets).run(
                None, {i: values[i] for i in node.input}
            )[0]
        except Exception:
            continue

        result = np.asarray(result)
        if result.size > _MAX_FOLD_ELEMENTS:
            continue
        values[node.output[0]] = result
        graph.node[idx].CopyFrom(
            onnx.helper.make_node(
                "Constant",
                [],
                [node.output[0]],
                name=node.name,
                value=numpy_helper.from_array(result),
            )
        )
        folded += 1

    if folded:
        # Drop nodes (typically the now-dead constant producers) whose
        # outputs are never consumed.
        used = {o.name for o in graph.output}
        kept = []
        for node in reversed(graph.node):
            if any(o in used for o in node.output):
                kept.append(node)
                used.update(i for i in node.input if i)
        del graph.node[:]
        graph.node.extend(reversed(kept))
    return folded


def load_onnx_model(model: str) -> onnx.ModelProto:
    """Load, name unnamed nodes, fold constants, and run shape inference."""
    raw_model = onnx.load(model)
    valid_opset_model = _upgrade_onnx_opset_for_torch_mlir(raw_model)

    # ONNX nodes don't strictly require names. If they are unnamed,
    # MLIR location tracking drops them. We force a name for every node.
    for i, node in enumerate(valid_opset_model.graph.node):
        if not node.name:
            node.name = f"{node.op_type}_{i}"

    folded = _fold_constant_subgraphs(valid_opset_model)
    if folded:
        logger.info("Folded %d constant-only ONNX node(s)", folded)

    try:
        return onnx.shape_inference.infer_shapes(valid_opset_model)
    except onnx.onnx_cpp2py_export.shape_inference.InferenceError:
        logger.error("Failed to infer shapes for ONNX model")
        raise


def convert_onnx_to_mlir(
    model: str | onnx.ModelProto,
    output_file: str = "",
    emit_bytecode: bool = False,
    elide_constant_if_larger: int = None,
):
    """Convert ONNX model into an MLIR file, using mainly upstream dialects.

    Args:
        model: Path to an ONNX model file, or a pre-loaded ``onnx.ModelProto``.
        output_file: Path where the ouput MLIR should be dumped. If empty, the module is not dumped.
        emit_bytecode: If True, emits MLIR as bytecode instead of text.
        elide_constant_if_larger: Elide constants in emitted IR if larger than specified value.

    Raises:
        May re-raise exceptions from the ONNX importer and onnx utilities

    Returns:
        The imported MLIR module.
    """

    if isinstance(model, str):
        inferred_model = load_onnx_model(model)
    else:
        inferred_model = model

    with make_vishap_context() as ctx:
        model_info = onnx_importer.ModelInfo(inferred_model)
        module = model_info.create_module(context=ctx)
        module_op = module.operation

        passes = [
            "torch-onnx-to-torch-backend-pipeline",
            "torch-backend-to-stablehlo-backend-pipeline",
            "func.func(stablehlo-aggressive-simplification)",
            "canonicalize",
        ]
        try:
            imp = onnx_importer.NodeImporter.define_function(
                model_info.main_graph, module_op
            )
            imp.import_all()
            module_op.verify()
            pipeline = ",".join(passes)
            # Same as `vishap-opt -pass-pipeline=<pipeline> <filename>`
            run_vishap_pipeline(
                module,
                f"builtin.module({pipeline})",
                "Lowering Torch ONNX -> StableHLO Backend IR",
            )
        except Exception:
            logger.error("Failed to lower ONNX to MLIR")
            raise

        if output_file:
            _dump_module(
                module_op,
                output_file,
                binary_format=emit_bytecode,
                max_constant=elide_constant_if_larger,
            )

    return module


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Utility script for converting ONNX models into MLIR with upstream dialects."
    )
    parser.add_argument(
        "input_model", type=str, help="input ONNX model path", metavar="model_path"
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
        "--emit-bytecode",
        dest="emit_bytecode",
        help="emit bytecode when generating output (default: %(default)s)",
        action="store_true",
    )
    parser.add_argument(
        "--elide-constants-if-larger",
        dest="max_constant",
        type=int,
        default=None,
        help="elide constants in emitted IR if larger than specified value",
        metavar="N",
    )

    return parser.parse_args()


def _main():
    args = _parse_args()

    try:
        convert_onnx_to_mlir(
            args.input_model, args.output_file, args.emit_bytecode, args.max_constant
        )
        return 0
    except Exception as e:
        logger.error(e)
        return 1


if __name__ == "__main__":
    sys.exit(_main())
