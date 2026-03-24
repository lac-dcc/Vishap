// RUN: vishap-opt --annotate-distributions %s | FileCheck %s

func.func @fill() -> tensor<1x64x3x3xf32> {
  %cst = arith.constant 1.0 : f32
  %out = tensor.empty() : tensor<1x64x3x3xf32>
  %filled = linalg.fill ins(%cst : f32) outs(%out : tensor<1x64x3x3xf32>) -> tensor<1x64x3x3xf32>
  return %filled : tensor<1x64x3x3xf32>
}
// CHECK-LABEL: func.func @fill
// CHECK-NEXT:    arith.constant
// CHECK-NEXT:    tensor.empty
// CHECK-NEXT:    linalg.fill
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00]
// CHECK-SAME:      ]
