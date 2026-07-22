// RUN: vishap-opt --annotate-distributions %s | FileCheck %s

func.func @subtract() -> tensor<2x2xf32> {
  %0 = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %1 = stablehlo.constant dense<[[4.0, 3.0], [2.0, 1.0]]> : tensor<2x2xf32>
  %res = stablehlo.subtract %0, %1 : tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @subtract
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.subtract
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-3.000000e+00, 3.000000e+00, 0.000000e+00, 2.500000e+00]
// CHECK-SAME:      ]
// CHECK-NEXT:    return

func.func @same_values() -> tensor<2x2xf64> {
  %0 = stablehlo.constant dense<[[1.0, 1.0], [1.0, 1.0]]> : tensor<2x2xf64>
  %1 = stablehlo.constant dense<[[1.0, 1.0], [1.0, 1.0]]> : tensor<2x2xf64>
  %res = stablehlo.subtract %0, %1 : tensor<2x2xf64>
  return %res : tensor<2x2xf64>
}
// CHECK-LABEL: func.func @same_values
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.subtract
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
// CHECK-SAME:      ]
// CHECK-NEXT:    return
