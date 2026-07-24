"""Convert ONNX model into an MLIR file, using mainly upstream dialects.

This script uses torch-mlir to convert ONNX models into MLIR files. We avoid
using non-upstream dialects (e.g., `torch`), since the main goal is to use
these models in projects that.

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
import onnx
import sys

from torch_mlir import ir
from torch_mlir.extras import onnx_importer
from torch_mlir.dialects import torch as torch_d
from torch_mlir.compiler_utils import run_pipeline_with_repro_report
from typing import NoReturn

logging.basicConfig(
    format="%(levelname)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def _dump_module(module, output_filename, binary_format=False, max_constant=None):
    if binary_format:
        assert output_filename, "Output file must be specified when using binary format"

    mode = "wb" if binary_format else "wt"
    output_file = open(output_filename, mode) if output_filename else None

    if binary_format:
        module.write_bytecode(file=output_file)
    else:
        module.print(
            file=output_file, large_elements_limit=max_constant, assume_verified=True
        )

    if output_file:
        output_file.close()


def convert_onnx_to_mlir(
    model_path: str,
    output_file: str,
    emit_bytecode: bool = False,
    elide_constant_if_larger: int = None,
) -> NoReturn:
    """Convert ONNX model into an MLIR file, using mainly upstream dialects.

    Args:
        model_path: Path to the target ONNX model.
        output_file: Path where the ouput MLIR should be saved.
        emit_bytecode: If True, emits MLIR as bytecode instead of text.
        elide_constant_if_larger: Elide constants in emitted IR if larger than specified value.

    Raises:
        May re-raise exceptions from torch-mlir and onnx utilities
    """

    try:
        raw_model = onnx.load(model_path)
        inferred_model = onnx.shape_inference.infer_shapes(raw_model)
    except onnx.onnx_cpp2py_export.shape_inference.InferenceError:
        logger.error("Failed to load ONNX model")
        raise

    with ir.Context() as ctx:
        torch_d.register_dialect(ctx)

        model_info = onnx_importer.ModelInfo(inferred_model)
        m = model_info.create_module(context=ctx).operation

        passes = [
            "torch-onnx-to-torch-backend-pipeline",
            "torch-backend-to-stablehlo-backend-pipeline",
            "func.func(stablehlo-aggressive-simplification)",
            "canonicalize",
        ]
        try:
            imp = onnx_importer.NodeImporter.define_function(model_info.main_graph, m)
            imp.import_all()
            m.verify()
            pipeline = ",".join(passes)
            # Same as `torch-mlir-opt -pass-pipeline=<pipeline> <filename>`
            run_pipeline_with_repro_report(
                m,
                f"builtin.module({pipeline})",
                "Lowering Torch ONNX -> StableHLO Backend IR",
            )
        except Exception:
            logger.error("Failed to lower ONNX to MLIR")
            raise

        _dump_module(
            m,
            output_file,
            binary_format=emit_bytecode,
            max_constant=elide_constant_if_larger,
        )


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
