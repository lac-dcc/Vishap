// RUN: vishap-opt --annotate-distributions %s | FileCheck %s

func.func @square_matrices() -> tensor<2x2xf32> {
  %0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %1 = arith.constant dense<[[4.0, 3.0], [2.0, 1.0]]> : tensor<2x2xf32>
  %out = tensor.empty() : tensor<2x2xf32>
  %res = linalg.matmul ins(%0, %1 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%out: tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @square_matrices
// CHECK-NEXT:    arith.constant
// CHECK-NEXT:    arith.constant
// CHECK-NEXT:    tensor.empty
// CHECK-NEXT:    linalg.matmul
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-22.678118198675726, 47.678118198675726, 1.250000e+01, 3.437500e+01]
// CHECK-SAME:      ]
// CHECK-NEXT:    return

func.func @different_shapes() -> tensor<2x2xf32> {
  %0 = arith.constant dense<[[4.0, 3.0, 2.0], [1.0, 0.0, -1.0]]> : tensor<2x3xf32>
  %1 = arith.constant dense<[[1.0, 0.0], [-0.5, 3.0], [2.0, 4.0]]> : tensor<3x2xf32>
  %out = tensor.empty() : tensor<2x2xf32>
  %res = linalg.matmul ins(%0, %1 : tensor<2x3xf32>, tensor<3x2xf32>) outs(%out: tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @different_shapes
// CHECK-NEXT:    arith.constant
// CHECK-NEXT:    arith.constant
// CHECK-NEXT:    tensor.empty
// CHECK-NEXT:    linalg.matmul
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-39.822444019882482, 54.072444019882482, 7.125000e+00, 61.223958333333321]
// CHECK-SAME:      ]
// CHECK-NEXT:    return
