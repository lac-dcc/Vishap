import csv
import ctypes
import logging
from pathlib import Path

import numpy as np

from utils import (
    BUILD_DIR,
    Stats,
    VISHAP_DIST_ATTR,
    ensure_vishap_bindings_on_path,
    get_array_stats,
    get_llvm_build_dir,
    parse_onnx_loc_name,
)

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    ensure_vishap_bindings_on_path()
    from vishap.execution_engine import ExecutionEngine
    from vishap.ir import (
        ArrayAttr,
        F64Type,
        FloatAttr,
        FloatType,
        IntegerType,
        Location,
        RankedTensorType,
        UnitAttr,
        WalkOrder,
        WalkResult,
    )
    from vishap.passmanager import PassManager
    from vishap.runtime import get_ranked_memref_descriptor, ranked_memref_to_numpy
except ImportError:
    logger.error(
        "Vishap bindings are not available. Ensure you have built Vishap with MLIR_ENABLE_BINDINGS_PYTHON."
    )
    raise

PROBE_REPORT_PATH = Path("vishap_probe_report.csv")
CONSTANT_OP_NAMES = frozenset({"stablehlo.constant", "arith.constant"})


def _op_parent_name(op) -> str | None:
    parent = getattr(op, "parent", None)
    if parent is None:
        return None
    if hasattr(parent, "name"):
        return parent.name
    owner = getattr(parent, "owner", None)
    if owner is not None and hasattr(owner, "name"):
        return owner.name
    return type(parent).__name__


def _is_nested_in_region(op) -> bool:
    parent_name = _op_parent_name(op)
    return parent_name is not None and parent_name not in (
        "func.func",
        "builtin.module",
    )


def is_float_ranked_tensor_type(type_) -> bool:
    """Identify float ranked tensors. Only these are observed by probe."""
    return isinstance(type_, RankedTensorType) and isinstance(
        type_.element_type, FloatType
    )


def walk_entry_ops(module) -> list:
    """Return ops in the same pre-order walk as `func.walk` in AddProbeCalls."""
    ops = []

    def visitor(op):
        ops.append(op)
        return WalkResult.ADVANCE

    module.body.operations[0].operation.walk(visitor, WalkOrder.PRE_ORDER)
    return ops


def collect_all_float_probe_targets(module) -> list[tuple[int, list[int]]]:
    """Walk indices and float result IDs matching `add-probe-calls{all-float-ops}`.

    Constants and ops nested in regions are skipped,
    matching ``add-probe-calls{all-float-ops}``. Each returned pair is
    ``(walk_index, [result_id, ...])`` for float ranked-tensor results.
    """
    targets: list[tuple[int, list[int]]] = []
    for walk_idx, op in enumerate(walk_entry_ops(module)):
        if op.name in CONSTANT_OP_NAMES or _is_nested_in_region(op):
            continue
        float_ids = [
            i
            for i, result in enumerate(op.results)
            if is_float_ranked_tensor_type(result.type)
        ]
        if float_ids:
            targets.append((walk_idx, float_ids))
    return targets


def _mlir_element_type_to_dtype(elem) -> np.dtype:
    if isinstance(elem, FloatType):
        return {16: np.float16, 32: np.float32, 64: np.float64}[elem.width]
    if isinstance(elem, IntegerType):
        width = elem.width
        if elem.is_unsigned:
            return {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}[width]
        return {1: np.bool_, 8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}[
            width
        ]
    raise TypeError(f"Unsupported MLIR element type: {elem}")


def _ranked_tensor_spec(ty) -> tuple[tuple[int, ...], np.dtype]:
    if not isinstance(ty, RankedTensorType):
        raise TypeError(f"Expected ranked tensor type, got {ty}")
    if any(dim < 0 for dim in ty.shape):
        raise ValueError(f"Dynamic tensor type is not supported: {ty}")
    return tuple(ty.shape), _mlir_element_type_to_dtype(ty.element_type)


def io_arrays_from_entry_func(
    module, provided_inputs: list[np.ndarray] | None = None
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Allocate numpy I/O arrays from the entry function type.

    ``provided_inputs[i]`` is used for argument i when present; remaining
    inputs and all outputs are zeros.
    """
    func = module.body.operations[0]
    func_type = func.type
    provided = provided_inputs or []

    input_arrays: list[np.ndarray] = []
    for i, ty in enumerate(func_type.inputs):
        shape, dtype = _ranked_tensor_spec(ty)
        if i < len(provided):
            arr = np.asarray(provided[i])
            if tuple(arr.shape) != shape:
                raise ValueError(
                    f"Function argument {i} has shape {shape}, got {arr.shape}"
                )
            input_arrays.append(arr.astype(dtype, copy=False))
        else:
            input_arrays.append(np.zeros(shape, dtype=dtype))

    output_arrays = []
    for ty in func_type.results:
        shape, dtype = _ranked_tensor_spec(ty)
        output_arrays.append(np.zeros(shape, dtype=dtype))
    return input_arrays, output_arrays


def stats_list_to_dist_attr(dists: list[Stats], context):
    """Build a `vishap.distribution` ArrayAttr (one 4-float array per result)."""
    f64 = F64Type.get(context=context)
    return ArrayAttr.get(
        [
            ArrayAttr.get([FloatAttr.get(f64, float(val)) for val in dist])
            for dist in dists
        ]
    )


def attach_probe_dists_to_ops(
    module,
    targets: list[tuple[int, list[int]]],
    probe_dists: list[Stats],
    ratio: float,
    seed: int | None,
) -> int:
    """Preload probe stats onto a random subset of non-constant float ops.

    ``targets`` must match ``collect_all_float_probe_targets`` / CSV row order.
    Returns the number of ops that received ``vishap.distribution``.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"preload ratio must be in [0, 1], got {ratio}")

    row_i = 0
    observations: list[tuple[int, list[int], list[Stats]]] = []
    for walk_idx, float_ids in targets:
        dists: list[Stats] = []
        for _ in float_ids:
            if row_i >= len(probe_dists):
                raise ValueError(
                    "Probe report has fewer rows than float results observed "
                    f"({len(probe_dists)} rows, needed at least {row_i + 1})"
                )
            dists.append(probe_dists[row_i])
            row_i += 1
        observations.append((walk_idx, float_ids, dists))

    if row_i != len(probe_dists):
        raise ValueError(
            "Probe report row count does not match observed float results "
            f"({len(probe_dists)} rows vs {row_i} expected)"
        )

    ops = walk_entry_ops(module)
    candidates: list[tuple[int, list[Stats]]] = []
    for walk_idx, float_ids, dists in observations:
        n_results = len(list(ops[walk_idx].results))
        if n_results == len(float_ids) and n_results > 0:
            candidates.append((walk_idx, dists))

    n = min(len(candidates), max(0, int(round(ratio * len(candidates)))))
    logger.info(
        "Preloading probe distributions onto %d/%d candidate ops (ratio=%s)",
        n,
        len(candidates),
        ratio,
    )
    if n == 0:
        return 0

    rng = np.random.default_rng(seed)
    chosen = {int(i) for i in rng.choice(len(candidates), size=n, replace=False)}

    with module.context, Location.unknown():
        for i, (walk_idx, dists) in enumerate(candidates):
            if i not in chosen:
                continue
            ops[walk_idx].attributes[VISHAP_DIST_ATTR] = stats_list_to_dist_attr(
                dists, module.context
            )
    return n


def collect_mixed_float_ops(module, node_name_to_outputs: dict[str, list[str]]) -> dict:
    """Map ONNX node name to the matching non-constant float MLIR op.

    An op is a candidate when its ONNX loc name maps (via
    ``node_name_to_outputs``) to a node whose output count matches the op's
    float result count. Constants and ops nested in regions are skipped.
    When several ops share a loc name (one ONNX node expanded into multiple
    MLIR ops), only the last one in walk order is kept, since it produces
    the value matching the ONNX node's output.
    """
    candidates_by_name = {}
    for op in walk_entry_ops(module):
        if op.name in CONSTANT_OP_NAMES or _is_nested_in_region(op):
            continue
        onnx_name = parse_onnx_loc_name(str(op.location))
        outputs = node_name_to_outputs.get(onnx_name)
        if not outputs:
            continue
        results = list(op.results)
        if len(results) != len(outputs) or not all(
            is_float_ranked_tensor_type(result.type) for result in results
        ):
            continue
        candidates_by_name[onnx_name] = op
    return candidates_by_name


def mixed_candidate_op_types(
    module,
    node_name_to_outputs: dict[str, list[str]],
    node_name_to_op_type: dict[str, str],
) -> list[str]:
    """Sorted unique ONNX op types among mixed float candidates, excluding Constant."""
    types = {
        node_name_to_op_type[name]
        for name in collect_mixed_float_ops(module, node_name_to_outputs)
        if name in node_name_to_op_type and node_name_to_op_type[name] != "Constant"
    }
    return sorted(types)


def attach_tensor_stats_to_ops(
    module,
    tensor_stats: dict[str, Stats],
    node_name_to_outputs: dict[str, list[str]],
    ratio: float | None = None,
    seed: int | None = None,
    *,
    node_name_to_op_type: dict[str, str] | None = None,
    vishap_op_type: str | None = None,
) -> int:
    """Preload ONNX tensor stats onto a subset of matching MLIR ops.

    Selection is either a random ``ratio`` of candidates, or every candidate
    whose ONNX op type is not ``vishap_op_type`` (those ops are left for
    Vishap static analysis). Exactly one of ``ratio`` or ``vishap_op_type``
    must be set.

    An op is a candidate when it is a mixed float op (see
    ``collect_mixed_float_ops``) and every output tensor has an entry in
    ``tensor_stats``. Returns the number of ops that received
    ``vishap.distribution``.
    """
    if (ratio is None) == (vishap_op_type is None):
        raise ValueError("exactly one of ratio or vishap_op_type must be set")
    if ratio is not None and not 0.0 <= ratio <= 1.0:
        raise ValueError(f"preload ratio must be in [0, 1], got {ratio}")
    if vishap_op_type is not None and node_name_to_op_type is None:
        raise ValueError("node_name_to_op_type is required when vishap_op_type is set")

    mixed_ops = collect_mixed_float_ops(module, node_name_to_outputs)
    candidates: list[tuple] = []
    for onnx_name, op in mixed_ops.items():
        outputs = node_name_to_outputs[onnx_name]
        if any(name not in tensor_stats for name in outputs):
            continue
        candidates.append(
            (op, [tensor_stats[name] for name in outputs], onnx_name)
        )

    if vishap_op_type is not None:
        chosen = {
            i
            for i, (_, _, name) in enumerate(candidates)
            if node_name_to_op_type.get(name) != vishap_op_type
        }
        n = len(chosen)
        logger.info(
            "Preloading tensor stats onto %d/%d candidate ops "
            "(vishap-op-type=%s, %d left for Vishap)",
            n,
            len(candidates),
            vishap_op_type,
            len(candidates) - n,
        )
    else:
        n = min(len(candidates), max(0, int(round(ratio * len(candidates)))))
        logger.info(
            "Preloading tensor stats onto %d/%d candidate ops (ratio=%s)",
            n,
            len(candidates),
            ratio,
        )
        if n == 0:
            return 0
        rng = np.random.default_rng(seed)
        chosen = {int(i) for i in rng.choice(len(candidates), size=n, replace=False)}

    if n == 0:
        return 0

    with module.context, Location.unknown():
        for i, (op, dists, _) in enumerate(candidates):
            if i not in chosen:
                continue
            op.attributes[VISHAP_DIST_ATTR] = stats_list_to_dist_attr(
                dists, module.context
            )
    return n


def parse_probe_report(path: Path | str = PROBE_REPORT_PATH) -> list[Stats]:
    """Read probe CSV rows as (min, max, mean, variance) tuples."""
    dists: list[Stats] = []
    with open(path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)
        for row in reader:
            dists.append(tuple(np.float64(val) for val in row[2:]))
    return dists


def lower_module_to_llvm(module, all_float_ops: bool = False):
    """Lower a StableHLO module to LLVM, inserting probe calls.

    Mutates ``module`` in place.
    """
    probe_pass = (
        "add-probe-calls{all-float-ops}" if all_float_ops else "add-probe-calls"
    )
    pipeline = f"""builtin.module(
        func.func(
            {probe_pass},
            canonicalize,
            cse
        ),
        stablehlo-convert-to-signless,
        func.func(stablehlo-legalize-to-linalg),
        empty-tensor-to-alloc-tensor,
        one-shot-bufferize{{bufferize-function-boundaries}},
        probe-lower-to-func-calls,
        convert-linalg-to-loops,
        convert-scf-to-cf,
        expand-strided-metadata,
        lower-affine,
        finalize-memref-to-llvm,
        convert-math-to-llvm,
        convert-math-to-libm,
        convert-arith-to-llvm,
        convert-index-to-llvm,
        convert-cf-to-llvm,
        func.func(
            llvm-request-c-wrappers
        ),
        convert-func-to-llvm,
        reconcile-unrealized-casts,
        func.func(
            canonicalize,
            cse
        )
    )"""

    PassManager.parse(pipeline).run(module.operation)
    entry_func = module.body.operations[0]
    entry_func.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
    return module


def _array_to_ranked_memref_ptr(array: np.ndarray):
    return ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(array)))


def run_probed_network(
    module,
    input_arrays: list[np.ndarray],
    output_arrays: list[np.ndarray],
    function: str,
    all_float_ops: bool = False,
) -> list[Stats]:
    """Lower ``module`` to LLVM, run it, and return output tensor stats.

    Mutates ``module``. Writes ``vishap_probe_report.csv`` via the runtime.
    """
    llvm_build_dir = get_llvm_build_dir()
    mlir_shared_libs = [
        str(llvm_build_dir / "lib/libmlir_runner_utils.so"),
        str(llvm_build_dir / "lib/libmlir_c_runner_utils.so"),
        str(BUILD_DIR / "lib/Runtime/libVishapRuntime.so"),
    ]

    with module.context:
        llvm_module = lower_module_to_llvm(module, all_float_ops=all_float_ops)
        engine = ExecutionEngine(llvm_module, opt_level=3, shared_libs=mlir_shared_libs)

    input_ptrs = [_array_to_ranked_memref_ptr(arr) for arr in input_arrays]
    output_ptrs = [_array_to_ranked_memref_ptr(arr) for arr in output_arrays]
    engine.invoke(function, *output_ptrs, *input_ptrs)

    return [
        get_array_stats(ranked_memref_to_numpy(ptr[0]).astype(np.float32))
        for ptr in output_ptrs
    ]
