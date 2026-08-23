import numpy as np
import os
import sys
from pathlib import Path
from PIL import Image

Stats = tuple[np.float64, np.float64, np.float64, np.float64]
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
BUILD_DIR = ROOT_DIR / "build"
TMP_DIR = ROOT_DIR / "tmp"
VISHAP_DIST_ATTR = "vishap.distribution"

DTYPE_MAP = {
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


def parse_io(io_desc: str) -> list[tuple[tuple[int, ...], np.dtype]]:
    """Parse a comma-separated list of tensor descriptors like ``1x2x3xf32``."""
    shapes_and_types = []
    for desc in io_desc.split(","):
        parts = desc.split("x")
        dtype_str = parts[-1]
        assert dtype_str in DTYPE_MAP, f"Unsupported data type: {dtype_str}"
        shape = tuple(int(dim) for dim in parts[:-1])
        shapes_and_types.append((shape, DTYPE_MAP[dtype_str]))
    return shapes_and_types


def get_llvm_build_dir() -> str:
    llvm_build_dir = os.getenv("LLVM_BUILD_DIR")
    assert llvm_build_dir, "'LLVM_BUILD_DIR' env variable must be set"
    assert os.path.exists(llvm_build_dir), f"{llvm_build_dir} is not a valid directory"

    return Path(llvm_build_dir)


def ensure_mlir_bindings_on_path():
    """
    Make the MLIR Python bindings importable. You must have a LLVM build
    with MLIR_ENABLE_BINDINGS_PYTHON enabled. More information here:
    https://mlir.llvm.org/docs/Bindings/Python/.
    """
    llvm_build_dir = get_llvm_build_dir()
    path = str(llvm_build_dir / "tools" / "mlir" / "python_packages" / "mlir_core")
    if path not in sys.path:
        sys.path.insert(0, path)


def ensure_vishap_bindings_on_path():
    """Make the Vishap Python bindings built under BUILD_DIR importable."""
    path = str(BUILD_DIR / "python_packages" / "vishap")
    if path not in sys.path:
        sys.path.insert(0, path)


def make_vishap_context():
    """Create an MLIR context with all the dialects needed by Vishap.

    Raises:
        ImportError: If the bindings are not built/importable.
    """
    ensure_vishap_bindings_on_path()
    from vishap import ir
    from vishap._mlir_libs import _vishap

    ctx = ir.Context()
    _vishap.register_dialects(ctx)
    return ctx


def annotate_distributions(module_asm: bytes | str, input_stats: list[Stats]):
    """Run the annotate-distributions pass in-process via Python bindings.

    Args:
        module_asm: The module to annotate, as in-memory MLIR bytecode
            (bytes) or textual assembly (str).
        input_stats: Distribution stats for each entry function argument.

    Raises:
        ImportError: If the Vishap bindings are not built/importable.
        Exception: May re-raise MLIR errors from parsing or the pass.

    Returns:
        The annotated MLIR module (a `vishap.ir.Module`, which keeps its
        context alive).
    """
    ensure_vishap_bindings_on_path()
    from vishap.ir import Module
    from vishap.passmanager import PassManager

    with make_vishap_context() as ctx:
        module = Module.parse(module_asm)
        pipeline = (
            "builtin.module(func.func(annotate-distributions{"
            + _stats_to_vishap_arg(input_stats)
            + "}))"
        )
        PassManager.parse(pipeline).run(module.operation)
    return module


def get_array_stats(array: np.ndarray) -> Stats:
    return (
        np.float64(array.min()),
        np.float64(array.max()),
        np.float64(array.mean()),
        np.float64(array.var()),
    )


def is_nchw_layout(shape: tuple[int, ...] | list[int]) -> bool:
    """Heuristic: NCHW when channel dim is at index 1 and looks like a channel count."""
    return len(shape) == 4 and shape[1] in (1, 3)


def spatial_size(shape: tuple[int, ...] | list[int], is_nchw: bool) -> tuple[int, int]:
    if is_nchw:
        return shape[2], shape[3]  # H, W
    return shape[1], shape[2]  # H, W


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png"}


def preprocess_image(
    image_path: Path, height: int, width: int, is_nchw: bool, dtype: np.dtype
) -> np.ndarray:
    """Load an image, resize to (height, width), return float tensor in [0, 1]."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((width, height), Image.BILINEAR)
        array = np.asarray(img, dtype=np.float32) / 255.0  # HWC

    array = array[np.newaxis, ...]  # NHWC
    if is_nchw:
        array = array.transpose(0, 3, 1, 2)  # NCHW
    return array.astype(dtype, copy=False)


def load_input_array(
    path: Path | str, shape: tuple[int, ...], dtype: np.dtype
) -> np.ndarray:
    """Load an image or .npy tensor matching the given shape and dtype.

    Images are resized and preprocessed to the target layout. .npy files must
    already have the exact shape.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    if is_image_path(path):
        is_nchw = is_nchw_layout(shape)
        height, width = spatial_size(shape, is_nchw)
        return preprocess_image(path, height, width, is_nchw, dtype)

    if path.suffix.lower() == ".npy":
        array = np.load(path)
        if tuple(array.shape) != tuple(shape):
            raise ValueError(f"Input {path} has shape {array.shape}, expected {shape}")
        return array.astype(dtype, copy=False)

    raise ValueError(
        f"Unsupported input file type: {path.suffix} (expected image or .npy)"
    )


def _stats_to_vishap_arg(stats: list[Stats]) -> str:
    arg = "input-distributions="
    input_dists = ";".join(
        [
            f"{min_val},{max_val},{mean_val},{var_val}"
            for min_val, max_val, mean_val, var_val in stats
        ]
    )

    return arg + input_dists
