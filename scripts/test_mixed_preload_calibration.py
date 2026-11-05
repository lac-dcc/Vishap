"""Checks for mixed preload ratio 1.0 against ONNX Runtime MinMax calibration.

Run from this directory:

    python -m unittest test_mixed_preload_calibration.py
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.calibrate import CalibrationMethod, TensorData, TensorsData

from ort_quantize import (
    RealDataCalibrationReader,
    _collect_minmax_ranges,
    _longest_path_depths,
    _merge_activation_ranges,
    _names_at_static_depth,
    _qdq_quantize,
    _select_preload,
)


def _range(low: float, high: float) -> TensorData:
    return TensorData(
        lowest=np.array(low, dtype=np.float32),
        highest=np.array(high, dtype=np.float32),
    )


def _conv_and_constant_model() -> onnx.ModelProto:
    """One Conv plus a Constant, so the preload pool has a single producer."""
    value = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "const_value")
    nodes = [
        helper.make_node("Conv", ["input", "weight"], ["output"], name="conv"),
        helper.make_node("Constant", [], ["const_out"], name="const", value=value),
    ]
    graph = helper.make_graph(
        nodes,
        "conv_const",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 1, 4, 4])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 1, 2, 2])],
        [
            numpy_helper.from_array(
                np.ones((1, 1, 3, 3), dtype=np.float32), "weight"
            )
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])


class MergeAndSelectionTest(unittest.TestCase):
    def test_full_keep_preserves_calibrator_objects(self):
        first = _range(0.0, 1.0)
        second = _range(-2.0, 2.0)
        calibrated = TensorsData(
            CalibrationMethod.MinMax, {"a": first, "b": second}
        )
        overrides = {
            "a": [{"rmin": np.float32(-5), "rmax": np.float32(5)}],
            "b": [{"rmin": np.float32(0), "rmax": np.float32(0)}],
            "missing": [{"rmin": np.float32(0), "rmax": np.float32(0)}],
        }
        merged = _merge_activation_ranges(calibrated, overrides, keep={"a", "b"})

        self.assertIs(merged.data["a"], first)
        self.assertIs(merged.data["b"], second)
        self.assertEqual(set(merged.data), {"a", "b"})

    def test_partial_keep_replaces_only_unselected(self):
        kept = _range(0.0, 1.0)
        replaced = _range(0.0, 1.0)
        untouched = _range(4.0, 5.0)
        calibrated = TensorsData(
            CalibrationMethod.MinMax,
            {"kept": kept, "replaced": replaced, "untouched": untouched},
        )
        overrides = {
            "kept": [{"rmin": np.float32(-1), "rmax": np.float32(1)}],
            "replaced": [{"rmin": np.float32(2), "rmax": np.float32(3)}],
            "untouched": [{"rmin": np.float32(np.nan), "rmax": np.float32(1)}],
            "extra": [{"rmin": np.float32(0), "rmax": np.float32(0)}],
        }
        merged = _merge_activation_ranges(calibrated, overrides, keep={"kept"})

        self.assertIs(merged.data["kept"], kept)
        self.assertIs(merged.data["untouched"], untouched)
        self.assertIsNot(merged.data["replaced"], replaced)
        self.assertEqual(set(merged.data), {"kept", "replaced", "untouched"})
        np.testing.assert_array_equal(merged.data["replaced"].lowest, np.float32(2))
        np.testing.assert_array_equal(merged.data["replaced"].highest, np.float32(3))

    def test_ratio_one_keeps_every_calibrator_tensor(self):
        model = _conv_and_constant_model()
        calibrated = {"input", "output", "const_out"}
        selection = _select_preload(model, calibrated, ratio=1.0, seed=0, vishap_op_type=None)

        self.assertEqual(selection.unselected_count, 0)
        self.assertEqual(selection.pool_count, 1)
        self.assertEqual(selection.selected_count, 1)
        self.assertTrue(calibrated <= selection.keep)
        self.assertIn("output", selection.selected_outputs)

    def test_ratio_zero_leaves_pool_outputs_unselected(self):
        model = _conv_and_constant_model()
        calibrated = {"input", "output", "const_out"}
        selection = _select_preload(model, calibrated, ratio=0.0, seed=0, vishap_op_type=None)

        self.assertEqual(selection.unselected_count, 1)
        self.assertEqual(selection.selected_count, 0)
        self.assertNotIn("output", selection.keep)
        self.assertIn("input", selection.keep)
        self.assertIn("const_out", selection.keep)


def _chain_predecessors(names: list[str]) -> dict[str, list[str]]:
    predecessors = {names[0]: []}
    for previous, name in zip(names, names[1:]):
        predecessors[name] = [previous]
    return predecessors


class StaticDepthTest(unittest.TestCase):
    def test_chain_depth_three_calibrates_every_fourth_hop(self):
        names = list("abcdefgh")
        depths = _longest_path_depths(_chain_predecessors(names))

        self.assertEqual(
            depths, {name: index for index, name in enumerate(names, start=1)}
        )
        self.assertEqual(_names_at_static_depth(depths, 3), {"c", "g"})

    def test_parallel_roots_share_depth(self):
        depths = _longest_path_depths({"a": [], "b": [], "c": ["a", "b"]})

        self.assertEqual(depths, {"a": 1, "b": 1, "c": 2})
        self.assertEqual(_names_at_static_depth(depths, 1), {"a", "b"})

    def test_skip_into_later_calibrated_node_keeps_longest_path(self):
        predecessors = _chain_predecessors([f"n{i}" for i in range(1, 8)])
        predecessors["n7"] = ["n6", "n3"]
        depths = _longest_path_depths(predecessors)

        self.assertEqual(depths["n3"], 3)
        self.assertEqual(depths["n7"], 7)
        self.assertEqual(_names_at_static_depth(depths, 3), {"n3", "n7"})

    def test_cycle_is_rejected(self):
        with self.assertRaises(ValueError):
            _longest_path_depths({"a": ["b"], "b": ["a"]})


def _relu_chain_model() -> onnx.ModelProto:
    """Four Relus plus a Constant, so the preload pool is the Relu chain."""
    value = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "const_value")
    nodes = [
        helper.make_node("Relu", ["input"], ["r0"], name="relu0"),
        helper.make_node("Relu", ["r0"], ["r1"], name="relu1"),
        helper.make_node("Relu", ["r1"], ["r2"], name="relu2"),
        helper.make_node("Relu", ["r2"], ["r3"], name="relu3"),
        helper.make_node("Constant", [], ["const_out"], name="const", value=value),
    ]
    graph = helper.make_graph(
        nodes,
        "relu_chain",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("r3", onnx.TensorProto.FLOAT, [1, 4])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])


class StaticDepthSelectionTest(unittest.TestCase):
    def test_vishap_names_calibrate_the_complement(self):
        model = _relu_chain_model()
        calibrated = {"input", "r0", "r1", "r2", "r3", "const_out"}
        selection = _select_preload(
            model,
            calibrated,
            ratio=None,
            seed=None,
            vishap_op_type=None,
            vishap_names={"relu2"},
        )

        self.assertEqual(selection.pool_count, 4)
        self.assertEqual(selection.unselected_count, 1)
        self.assertEqual(selection.selected_count, 3)
        self.assertNotIn("r2", selection.keep)
        self.assertNotIn("r2", selection.selected_outputs)
        for name in ("r0", "r1", "r3", "input", "const_out"):
            self.assertIn(name, selection.keep)

    def test_depth_stride_keeps_stride_and_leaves_the_rest(self):
        model = _relu_chain_model()
        names = ["relu0", "relu1", "relu2", "relu3"]
        depths = _longest_path_depths(_chain_predecessors(names))
        calibrated_nodes = _names_at_static_depth(depths, 3)
        vishap_names = set(depths) - calibrated_nodes
        calibrated = {"input", "r0", "r1", "r2", "r3", "const_out"}
        selection = _select_preload(
            model,
            calibrated,
            ratio=None,
            seed=None,
            vishap_op_type=None,
            vishap_names=vishap_names,
        )

        self.assertEqual(calibrated_nodes, {"relu2"})
        self.assertEqual(vishap_names, {"relu0", "relu1", "relu3"})
        self.assertEqual(selection.pool_count, 4)
        self.assertEqual(selection.selected_count, 1)
        self.assertEqual(selection.unselected_count, 3)
        self.assertIn("r2", selection.keep)
        self.assertIn("r2", selection.selected_outputs)
        for name in ("r0", "r1", "r3"):
            self.assertNotIn(name, selection.keep)
        for name in ("input", "const_out"):
            self.assertIn(name, selection.keep)


def _initializers(path: Path) -> dict[str, np.ndarray]:
    model = onnx.load(path)
    return {
        init.name: numpy_helper.to_array(init) for init in model.graph.initializer
    }


class CalibrationIdentityTest(unittest.TestCase):
    def test_full_keep_matches_quantize_static_scales(self):
        rng = np.random.default_rng(0)
        batches = [
            rng.standard_normal((1, 1, 8, 8)).astype(np.float32),
            rng.standard_normal((1, 1, 8, 8)).astype(np.float32),
        ]
        weight = rng.standard_normal((2, 1, 3, 3)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "conv.onnx"
            static_path = root / "static.onnx"
            mixed_path = root / "mixed.onnx"
            graph = helper.make_graph(
                [helper.make_node("Conv", ["input", "weight"], ["output"], name="conv")],
                "conv",
                [
                    helper.make_tensor_value_info(
                        "input", onnx.TensorProto.FLOAT, [1, 1, 8, 8]
                    )
                ],
                [
                    helper.make_tensor_value_info(
                        "output", onnx.TensorProto.FLOAT, [1, 2, 6, 6]
                    )
                ],
                [numpy_helper.from_array(weight, "weight")],
            )
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 15)]
            )
            # The installed ONNX Runtime accepts IR version <= 11.
            model.ir_version = 8
            onnx.checker.check_model(model)
            onnx.save(model, model_path)

            quantize_static(
                model_input=str(model_path),
                model_output=str(static_path),
                calibration_data_reader=RealDataCalibrationReader(
                    str(model_path), batches
                ),
                quant_format=QuantFormat.QDQ,
                calibrate_method=CalibrationMethod.MinMax,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
            )
            quantized_model, ranges = _collect_minmax_ranges(str(model_path), batches)
            merged = _merge_activation_ranges(ranges, {}, set(ranges.data))
            for name, tensor_data in ranges.data.items():
                self.assertIs(merged.data[name], tensor_data)
            _qdq_quantize(quantized_model, merged, str(mixed_path))

            static_inits = _initializers(static_path)
            mixed_inits = _initializers(mixed_path)
            self.assertEqual(set(static_inits), set(mixed_inits))
            for name, value in static_inits.items():
                np.testing.assert_array_equal(value, mixed_inits[name], err_msg=name)


if __name__ == "__main__":
    unittest.main()
