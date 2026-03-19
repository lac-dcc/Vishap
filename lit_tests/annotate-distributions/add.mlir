// RUN: vishap-opt --annotate-distributions %s | FileCheck %s

func.func @add() -> tensor<2x2xf32> {
  %0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %1 = arith.constant dense<[[4.0, 3.0], [2.0, 1.0]]> : tensor<2x2xf32>
  %out = tensor.empty() : tensor<2x2xf32>
  %res = linalg.add ins(%0, %1 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%out: tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @add
// CHECK-NEXT:    arith.constant
// CHECK-NEXT:    arith.constant
// CHECK-NEXT:    tensor.empty
// CHECK-NEXT:    linalg.add
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [2.000000e+00, 8.000000e+00, 5.000000e+00, 2.500000e+00]
// CHECK-SAME:      ]
// CHECK-NEXT:    return

func.func @same_values() -> tensor<2x2xf64> {
  %0 = arith.constant dense<[[1.0, 1.0], [1.0, 1.0]]> : tensor<2x2xf64>
  %1 = arith.constant dense<[[1.0, 1.0], [1.0, 1.0]]> : tensor<2x2xf64>
  %out = tensor.empty() : tensor<2x2xf64>
  %res = linalg.add ins(%0, %1 : tensor<2x2xf64>, tensor<2x2xf64>) outs(%out: tensor<2x2xf64>) -> tensor<2x2xf64>
  return %res : tensor<2x2xf64>
}
// CHECK-LABEL: func.func @same_values
// CHECK-NEXT:    arith.constant
// CHECK-NEXT:    arith.constant
// CHECK-NEXT:    tensor.empty
// CHECK-NEXT:    linalg.add
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [2.000000e+00, 2.000000e+00, 2.000000e+00, 0.000000e+00]
// CHECK-SAME:      ]
// CHECK-NEXT:    return
