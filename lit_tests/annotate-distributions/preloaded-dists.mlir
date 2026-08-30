// RUN: vishap-opt --annotate-distributions %s | FileCheck %s

func.func @preloaded() -> tensor<2x2xf32> {
  %0 = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %1 = stablehlo.constant dense<[[4.0, 3.0], [2.0, 1.0]]> : tensor<2x2xf32>
  %2 = stablehlo.add %0, %1 {vishap.distribution = [[-1.0, 1.0, 1.0, 0.0]]} : tensor<2x2xf32>
  %res = stablehlo.add %2, %2 : tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @preloaded
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.add
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00]
// CHECK-SAME:      ]
// CHECK-NEXT:    stablehlo.add
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-2.000000e+00, 2.000000e+00, 2.000000e+00, 0.000000e+00]
// CHECK-SAME:      ]
// CHECK-NEXT:    return
