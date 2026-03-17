// RUN: vishap-opt --annotate-distributions %s | FileCheck %s

func.func @broadcast() -> tensor<1x64x3x3xf32> {
  %cst = arith.constant dense<[[0.0, 2.0, 4.0], [6.0, 8.0, -10.0], [0.0, 2.0, 0.0]]> : tensor<3x3xf32>
  %out = tensor.empty() : tensor<1x64x3x3xf32>
  %broadcasted = linalg.broadcast ins(%cst : tensor<3x3xf32>) outs(%out : tensor<1x64x3x3xf32>) dimensions = [0, 1]
  return %broadcasted : tensor<1x64x3x3xf32>
}
// CHECK-LABEL: func.func @broadcast
// CHECK-NEXT:    arith.constant
// CHECK-NEXT:    tensor.empty
// CHECK-NEXT:    linalg.broadcast
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-1.000000e+01, 8.000000e+00, 1.3333333333333333, 23.111111111111114]
// CHECK-SAME:      ]
