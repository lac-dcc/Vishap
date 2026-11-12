// RUN: vishap-opt --annotate-distributions="per-axis=false" %s | FileCheck %s --check-prefixes=CHECK,TENSOR
// RUN: vishap-opt --annotate-distributions="per-axis=true annotate-axes=true" %s | FileCheck %s --check-prefixes=CHECK,SLICE

func.func @max_reduce_window() -> tensor<1x2x1x1xf32> {
  %input = stablehlo.constant dense<[[[[1.0, 3.0], [2.0, 4.0]], [[5.0, 7.0], [6.0, 8.0]]]]> : tensor<1x2x2x2xf32>
  %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %res = "stablehlo.reduce_window"(%input, %init) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %max = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %max : tensor<f32>
  }) {
    window_dimensions = array<i64: 1, 1, 2, 2>,
    window_strides = array<i64: 1, 1, 1, 1>
  } : (tensor<1x2x2x2xf32>, tensor<f32>) -> tensor<1x2x1x1xf32>
  return %res : tensor<1x2x1x1xf32>
}
// CHECK-LABEL: func.func @max_reduce_window
// CHECK:         stablehlo.reduce_window
// SLICE:           vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          [],
// SLICE-SAME:          {{\[\[}}1.000000e+00, 4.000000e+00, 3.750000e+00, 1.250000e+00], [5.000000e+00, 8.000000e+00, 7.750000e+00, 1.250000e+00{{\]\]}},
// SLICE-SAME:          [],
// SLICE-SAME:          []
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// SLICE:           vishap.distribution = [
// SLICE-SAME:        [1.000000e+00, 8.000000e+00, 5.750000e+00, 5.250000e+00]
// SLICE-SAME:      ]
// TENSOR:          vishap.distribution = [
// TENSOR-SAME:       [1.000000e+00, 8.000000e+00, 9.750000e+00, 5.250000e+00]
// TENSOR-SAME:     ]
// CHECK:         return

func.func @sum_reduce_window() -> tensor<1x2x1x1xf32> {
  %input = stablehlo.constant dense<[[[[1.0, 3.0], [2.0, 4.0]], [[5.0, 7.0], [6.0, 8.0]]]]> : tensor<1x2x2x2xf32>
  %init = stablehlo.constant dense<0.0> : tensor<f32>
  %res = "stablehlo.reduce_window"(%input, %init) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %sum = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %sum : tensor<f32>
  }) {
    window_dimensions = array<i64: 1, 1, 2, 2>,
    window_strides = array<i64: 1, 1, 1, 1>
  } : (tensor<1x2x2x2xf32>, tensor<f32>) -> tensor<1x2x1x1xf32>
  return %res : tensor<1x2x1x1xf32>
}
// CHECK-LABEL: func.func @sum_reduce_window
// CHECK:         stablehlo.reduce_window
// SLICE:           vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          [],
// SLICE-SAME:          {{\[\[}}4.000000e+00, 1.600000e+01, 1.000000e+01, 5.000000e+00], [2.000000e+01, 3.200000e+01, 2.600000e+01, 5.000000e+00{{\]\]}},
// SLICE-SAME:          [],
// SLICE-SAME:          []
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// SLICE:           vishap.distribution = [
// SLICE-SAME:        [4.000000e+00, 3.200000e+01, 1.800000e+01, 6.900000e+01]
// SLICE-SAME:      ]
// TENSOR:          vishap.distribution = [
// TENSOR-SAME:       [4.000000e+00, 3.200000e+01, 1.800000e+01, 2.100000e+01]
// TENSOR-SAME:     ]
// CHECK:         return

func.func @max_reduce() -> tensor<1x2xf32> {
  %input = stablehlo.constant dense<[[[[1.0, 3.0], [2.0, 4.0]], [[5.0, 7.0], [6.0, 8.0]]]]> : tensor<1x2x2x2xf32>
  %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %res = stablehlo.reduce(%input init: %init) applies stablehlo.maximum across dimensions = [2, 3] : (tensor<1x2x2x2xf32>, tensor<f32>) -> tensor<1x2xf32>
  return %res : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @max_reduce
// CHECK:         stablehlo.reduce
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          [],
// SLICE-SAME:          {{\[\[}}1.000000e+00, 4.000000e+00, 3.750000e+00, 1.250000e+00], [5.000000e+00, 8.000000e+00, 7.750000e+00, 1.250000e+00{{\]\]}}
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// SLICE:           vishap.distribution = [
// SLICE-SAME:        [3.000000e+00, 8.000000e+00, 9.750000e+00, 4.250000e+00]
// SLICE-SAME:      ]
// TENSOR:          vishap.distribution = [
// TENSOR-SAME:       [1.000000e+00, 8.000000e+00, 9.750000e+00, 5.250000e+00]
// TENSOR-SAME:     ]
// CHECK:         return

func.func @sum_reduce() -> tensor<1x2xf32> {
  %input = stablehlo.constant dense<[[[[1.0, 3.0], [2.0, 4.0]], [[5.0, 7.0], [6.0, 8.0]]]]> : tensor<1x2x2x2xf32>
  %init = stablehlo.constant dense<0.0> : tensor<f32>
  %res = stablehlo.reduce(%input init: %init) applies stablehlo.add across dimensions = [2, 3] : (tensor<1x2x2x2xf32>, tensor<f32>) -> tensor<1x2xf32>
  return %res : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @sum_reduce
// CHECK:         stablehlo.reduce
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          [],
// SLICE-SAME:          {{\[\[}}4.000000e+00, 1.600000e+01, 1.000000e+01, 5.000000e+00], [2.000000e+01, 3.200000e+01, 2.600000e+01, 5.000000e+00{{\]\]}}
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// SLICE:           vishap.distribution = [
// SLICE-SAME:        [8.000000e+00, 2.800000e+01, 1.800000e+01, 1.700000e+01]
// SLICE-SAME:      ]
// TENSOR:          vishap.distribution = [
// TENSOR-SAME:       [4.000000e+00, 3.200000e+01, 1.800000e+01, 2.100000e+01]
// TENSOR-SAME:     ]
// CHECK:         return
