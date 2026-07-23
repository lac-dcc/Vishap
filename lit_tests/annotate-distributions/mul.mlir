// RUN: vishap-opt --annotate-distributions %s | FileCheck %s

func.func @multiply() -> tensor<2x2xf32> {
  %0 = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %1 = stablehlo.constant dense<[[4.0, 3.0], [2.0, 1.0]]> : tensor<2x2xf32>
  %res = stablehlo.multiply %0, %1 : tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @multiply
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.multiply
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [1.000000e+00, 1.600000e+01, 6.250000e+00, 1.718750e+01]
// CHECK-SAME:      ]
// CHECK-NEXT:    return

func.func @identity() -> tensor<2x2xf64> {
  %0 = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
  %1 = stablehlo.constant dense<[[1.0, 1.0], [1.0, 1.0]]> : tensor<2x2xf64>
  %res = stablehlo.multiply %0, %1 : tensor<2x2xf64>
  return %res : tensor<2x2xf64>
}
// CHECK-LABEL: func.func @identity
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.multiply
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [1.000000e+00, 4.000000e+00, 2.500000e+00, 1.250000e+00]
// CHECK-SAME:      ]
// CHECK-NEXT:    return


func.func @zero() -> tensor<2x2xf64> {
  %0 = stablehlo.constant dense<[[0.5, 0.45], [0.01, 0.2]]> : tensor<2x2xf64>
  %1 = stablehlo.constant dense<[[0.0, 0.0], [0.0, 0.0]]> : tensor<2x2xf64>
  %res = stablehlo.multiply %0, %1 : tensor<2x2xf64>
  return %res : tensor<2x2xf64>
}
// CHECK-LABEL: func.func @zero
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.multiply
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
// CHECK-SAME:      ]
// CHECK-NEXT:    return
