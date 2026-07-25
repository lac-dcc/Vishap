// RUN: vishap-opt --annotate-distributions %s | FileCheck %s

func.func @pad_zero(%arg0: tensor<1x32x10x3xf32>) -> tensor<1x32x32x32xf32>
            attributes {vishap.args_distribution = [[-1., 12., 3., 4.]]} {
  %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %padded = stablehlo.pad %arg0, %cst_0, low = [0, 0, 10, 11], high = [0, 0, 12, 18], interior = [0, 0, 0, 0]
              : (tensor<1x32x10x3xf32>, tensor<f32>) -> tensor<1x32x32x32xf32>
  return %padded : tensor<1x32x32x32xf32>
}
// CHECK-LABEL: func.func @pad_zero
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.pad
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-1.000000e+00, 1.200000e+01, 0.087890625, 0.37313461303710938]
// CHECK-SAME:      ]

func.func @pad_one(%arg0: tensor<1x16x10x3xf32>) -> tensor<1x16x16x16xf32>
            attributes {vishap.args_distribution = [[-1., 1., 0.55, 0.05]]} {
  %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %padded = stablehlo.pad %arg0, %cst_1, low = [0, 0, 3, 5], high = [0, 0, 3, 8], interior = [0, 0, 0, 0]
              : (tensor<1x16x10x3xf32>, tensor<f32>) -> tensor<1x16x16x16xf32>
  return %padded : tensor<1x16x16x16xf32>
}
// CHECK-LABEL: func.func @pad_one
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.pad
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-1.000000e+00, 1.000000e+00, 0.947265625, 0.026808929443359369]
// CHECK-SAME:      ]

func.func @pad_dynamic_shape(%arg0: tensor<?x2x3x4xf64>) -> tensor<?x4x4x4xf64>
            attributes {vishap.args_distribution = [[0.75, 1.5, 1., 0.25]]} {
  %cst = stablehlo.constant dense<0.5> : tensor<f64>
  %padded = stablehlo.pad %arg0, %cst, low = [0, 1, 1, 0], high = [0, 1, 0, 0], interior = [0, 0, 0, 0]
              : (tensor<?x2x3x4xf64>, tensor<f64>) -> tensor<?x4x4x4xf64>
  return %padded : tensor<?x4x4x4xf64>
}
// CHECK-LABEL: func.func @pad_dynamic_shape
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.pad
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [5.000000e-01, 1.500000e+00, 6.875000e-01, 0.15234375]
// CHECK-SAME:      ]

func.func @pad_nan(%arg0: tensor<1x14x10x10xf32>) -> tensor<1x16x16x16xf32>
            attributes {vishap.args_distribution = [[-1., 1., 0.55, 0.05]]} {
  %cst_nan = stablehlo.constant dense<0x7FC00000> : tensor<f32> // NaN
  %padded = stablehlo.pad %arg0, %cst_nan, low = [0, 2, 3, 5], high = [0, 0, 3, 1], interior = [0, 0, 0, 0]
              : (tensor<1x14x10x10xf32>, tensor<f32>) -> tensor<1x16x16x16xf32>
  return %padded : tensor<1x16x16x16xf32>
}
// CHECK-LABEL: func.func @pad_nan
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.pad
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-1.000000e+00, 1.000000e+00, 5.500000e-01, 5.000000e-02]
// CHECK-SAME:      ]

func.func @pad_inf(%arg0: tensor<1x14x13x11xf32>) -> tensor<1x16x16x16xf32>
            attributes {vishap.args_distribution = [[-1., 1., 0.55, 0.05]]} {
  %cst_inf = stablehlo.constant dense<0x7F800000> : tensor<f32> // Infinity
  %padded = stablehlo.pad %arg0, %cst_inf, low = [0, 0, 0, 5], high = [0, 2, 3, 0], interior = [0, 0, 0, 0]
              : (tensor<1x14x13x11xf32>, tensor<f32>) -> tensor<1x16x16x16xf32>
  return %padded : tensor<1x16x16x16xf32>
}
// CHECK-LABEL: func.func @pad_inf
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.pad
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-1.000000e+00, 1.000000e+00, 5.500000e-01, 5.000000e-02]
// CHECK-SAME:      ]

func.func @pad_converted_one(%arg0: tensor<1x32x10x3xf32>) -> tensor<1x32x32x32xf32>
            attributes {vishap.args_distribution = [[-1., 12., 3., 4.]]} {
  %cst = arith.constant dense<1> : tensor<1xi64>
  %0 = stablehlo.convert %cst : (tensor<1xi64>) -> tensor<1xf32>
  %1 = stablehlo.reshape %0 : (tensor<1xf32>) -> tensor<f32>
  %padded = stablehlo.pad %arg0, %1, low = [0, 0, 10, 11], high = [0, 0, 12, 18], interior = [0, 0, 0, 0]
              : (tensor<1x32x10x3xf32>, tensor<f32>) -> tensor<1x32x32x32xf32>
  return %padded : tensor<1x32x32x32xf32>
}
// CHECK-LABEL: func.func @pad_converted_one
// CHECK-NEXT:    arith.constant
// CHECK-NEXT:    stablehlo.convert
// CHECK-NEXT:    stablehlo.reshape
// CHECK-NEXT:    stablehlo.pad
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-1.000000e+00, 1.200000e+01, 1.05859375, 0.2309417724609375]
// CHECK-SAME:      ]
