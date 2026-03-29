// RUN: vishap-opt --annotate-distributions %s | FileCheck %s

func.func @broadcast() -> tensor<1x64x3x3xf32> {
  %cst = arith.constant dense<[[0.0, 2.0, 4.0], [6.0, 8.0, -10.0], [0.0, 2.0, 0.0]]> : tensor<3x3xf32>
  %out = tensor.empty() : tensor<1x64x3x3xf32>
  %broadcasted = linalg.broadcast ins(%cst : tensor<3x3xf32>) outs(%out : tensor<1x64x3x3xf32>) dimensions = [0, 1]
  return %broadcasted : tensor<1x64x3x3xf32>
}
// CHECK-LABEL: func.func @broadcast
// CHECK-NEXT:    arith.constant
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-1.000000e+01, 8.000000e+00, 1.3333333333333333, 23.111111111111114]
// CHECK-SAME:      ]
// CHECK-NEXT:    tensor.empty
// CHECK-NEXT:    linalg.broadcast
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-1.000000e+01, 8.000000e+00, 1.3333333333333333, 23.111111111111114]
// CHECK-SAME:      ]

func.func @transpose() -> tensor<1x3x2xf32> {
  %cst = arith.constant dense<[[[1.0, 0.0, -2.0], [3.0, 0.1, -0.1]]]> : tensor<1x2x3xf32>
  %out = tensor.empty() : tensor<1x3x2xf32>
  %tranposed = linalg.transpose ins(%cst : tensor<1x2x3xf32>) outs(%out : tensor<1x3x2xf32>) permutation = [0, 2, 1]
  return %tranposed : tensor<1x3x2xf32>
}
// CHECK-LABEL: func.func @transpose
// CHECK-NEXT:    arith.constant
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-2.000000e+00, 3.000000e+00, 0.33333333333333331, 2.2255555556548967]
// CHECK-SAME:      ]
// CHECK-NEXT:    tensor.empty
// CHECK-NEXT:    linalg.transpose
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-2.000000e+00, 3.000000e+00, 0.33333333333333331, 2.2255555556548967]
// CHECK-SAME:      ]

func.func @collapse_shape() -> tensor<2x3x4xf64> {
  %cst = arith.constant dense<[[[[1.2, 1.5, 0.3, 1.4],[0.5, 1.7, 1.3, 2.],[0.1, 0.9, 0.6, 0.7]],[[2.3, 1.9, 1.1, 1.8],[0.2, 1.6, 2.1, 0.8],[1., 0.4, 2.2, 0.]]]]> : tensor<1x2x3x4xf64>
  %collapsed = tensor.collapse_shape %cst [[0, 1], [2], [3]] : tensor<1x2x3x4xf64> into tensor <2x3x4xf64>
  return %collapsed : tensor<2x3x4xf64>
}
// CHECK-LABEL: func.func @collapse_shape
// CHECK-NEXT:    arith.constant
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [0.000000e+00, 2.300000e+00, 1.1500000000000001, 0.47916666666666674]
// CHECK-SAME:      ]
// CHECK-NEXT:    tensor.collapse_shape
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [0.000000e+00, 2.300000e+00, 1.1500000000000001, 0.47916666666666674]
// CHECK-SAME:      ]
