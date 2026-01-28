import ctypes
import os
import numpy as np
from PIL import Image

# This script uses MLIR Python bindings. To run it, you must have a LLVM build
# with MLIR_ENABLE_BINDINGS_PYTHON enabled. You also need to update your
# PYTHONPATH. More information here: https://mlir.llvm.org/docs/Bindings/Python/
from mlir.execution_engine import ExecutionEngine
from mlir.ir import Module, Context, UnitAttr
from mlir.passmanager import PassManager
from mlir.runtime import get_ranked_memref_descriptor, ranked_memref_to_numpy


def parse_network_to_llvm(ctx):
    TOSA_TO_LLVM_PIPELINE = """builtin.module(
        func.func(
            tosa-infer-shapes,
            tosa-layerwise-constant-fold,
            canonicalize,
            cse,

            tosa-to-linalg-named,
            tosa-to-linalg,
            tosa-to-arith,
            tosa-to-tensor,
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
        convert-arith-to-llvm,
        convert-index-to-llvm,
        convert-func-to-llvm,
        reconcile-unrealized-casts
    )"""

    module = Module.parseFile("resnet50.tosa.mlir", ctx)

    entry_func = module.body.operations[0]
    entry_func.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    pm = PassManager.parse(TOSA_TO_LLVM_PIPELINE)
    pm.run(module.operation)

    return module


def init_input():
    # Convert input image to numpy array. Assume NHWC layout
    img = Image.open("tests/inputs/coffee_pot.jpg").convert("RGB")
    img = img.resize((224, 224))
    img_arr = np.array(img, dtype=np.float32) / 255.0
    img_arr = np.expand_dims(
        img_arr,
        axis=0,
    )
    return img_arr


def init_output():
    out_arr = np.zeros((1, 1001), dtype=np.float32)
    return out_arr


def array_to_ranked_memref_ptr(array):
    return ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(array)))


def main():
    llvm_build_dir = os.getenv("LLVM_BUILD_DIR")
    assert llvm_build_dir, "'LLVM_BUILD_DIR' env variable must be set"
    assert os.path.exists(llvm_build_dir), f"{llvm_build_dir} is not a valid directory"

    MLIR_SHARED_LIBS = [
        os.path.join(llvm_build_dir, "lib/libmlir_runner_utils.so"),
        os.path.join(llvm_build_dir, "lib/libmlir_c_runner_utils.so"),
    ]

    with Context() as ctx:
        module = parse_network_to_llvm(ctx)
        # module.operation.print(large_elements_limit=10)
        engine = ExecutionEngine(module, opt_level=3, shared_libs=MLIR_SHARED_LIBS)

    input_arr = init_input()
    input_memref_ptr = array_to_ranked_memref_ptr(input_arr)
    output_arr = init_output()
    output_memref_ptr = array_to_ranked_memref_ptr(output_arr)
    engine.invoke(
        "main",
        output_memref_ptr,
        input_memref_ptr,
    )

    output = ranked_memref_to_numpy(output_memref_ptr[0])
    print("=" * 10, "Output summary", "=" * 10)
    print("Sum:", output.sum())
    print("Max:", output.max())
    print("Min:", output.min())
    print("Argmax:", output.argmax())

    with open("assets/text/imagenet1k_classes.txt") as f:
        prediction = output.argmax()
        predicted_class = f.readlines()[prediction].strip()
        print(f"Inferred class: {prediction} - {predicted_class}")


if __name__ == "__main__":
    main()
