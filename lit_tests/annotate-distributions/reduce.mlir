// RUN: vishap-opt --annotate-distributions="input-distributions=-1.5,3.,1.,0.5" %s | FileCheck %s

func.func @max_reduce_window(%input: tensor<1x48x32x32xf32>) -> tensor<1x48x1x1xf32> {
  %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %res = "stablehlo.reduce_window"(%input, %init) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %max = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %max : tensor<f32>
  }) {
    window_dimensions = array<i64: 1, 1, 32, 32>,
    window_strides = array<i64: 1, 1, 1, 1>
  } : (tensor<1x48x32x32xf32>, tensor<f32>) -> tensor<1x48x1x1xf32>
  return %res : tensor<1x48x1x1xf32>
}
// CHECK-LABEL: func.func @max_reduce_window
// CHECK:         stablehlo.reduce_window
// CHECK:         vishap.distribution = [
// CHECK-SAME:        [-1.500000e+00, 3.000000e+00, 1.500000e+00, 5.000000e-01]
// CHECK-SAME:      ]
// CHECK:         return

func.func @sum_reduce_window(%input: tensor<1x48x32x32xf32>) -> tensor<1x48x1x1xf32> {
  %init = stablehlo.constant dense<0.0> : tensor<f32>
  %res = "stablehlo.reduce_window"(%input, %init) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %sum = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %sum : tensor<f32>
  }) {
    window_dimensions = array<i64: 1, 1, 32, 32>,
    window_strides = array<i64: 1, 1, 1, 1>
  } : (tensor<1x48x32x32xf32>, tensor<f32>) -> tensor<1x48x1x1xf32>
  return %res : tensor<1x48x1x1xf32>
}
// CHECK-LABEL: func.func @sum_reduce_window
// CHECK:         stablehlo.reduce_window
// CHECK:         vishap.distribution = [
// CHECK-SAME:        [-1.536000e+03, 3.072000e+03, 1.024000e+03, 5.120000e+02]
// CHECK-SAME:      ]
// CHECK:         return

func.func @max_reduce(%input: tensor<1x48x32x32xf32>) -> tensor<1x48xf32> {
  %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %res = stablehlo.reduce(%input init: %init) applies stablehlo.maximum across dimensions = [2, 3] : (tensor<1x48x32x32xf32>, tensor<f32>) -> tensor<1x48xf32>
  return %res : tensor<1x48xf32>
}
// CHECK-LABEL: func.func @max_reduce
// CHECK:         stablehlo.reduce
// CHECK:         vishap.distribution = [
// CHECK-SAME:        [-1.500000e+00, 3.000000e+00, 1.500000e+00, 5.000000e-01]
// CHECK-SAME:      ]
// CHECK:         return

func.func @sum_reduce(%input: tensor<1x48x32x32xf32>) -> tensor<1x48xf32> {
  %init = stablehlo.constant dense<0.0> : tensor<f32>
  %res = stablehlo.reduce(%input init: %init) applies stablehlo.add across dimensions = [2, 3] : (tensor<1x48x32x32xf32>, tensor<f32>) -> tensor<1x48xf32>
  return %res : tensor<1x48xf32>
}
// CHECK-LABEL: func.func @sum_reduce
// CHECK:         stablehlo.reduce
// CHECK:         vishap.distribution = [
// CHECK-SAME:        [-1.536000e+03, 3.072000e+03, 1.024000e+03, 5.120000e+02]
// CHECK-SAME:      ]
// CHECK:         return
