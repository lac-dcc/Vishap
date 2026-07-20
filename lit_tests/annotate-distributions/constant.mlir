// RUN: vishap-opt --annotate-distributions %s | FileCheck %s

func.func @same_values() -> tensor<2x2xf64> {
  %cst = stablehlo.constant dense<[[1.0, 1.0], [1.0, 1.0]]> : tensor<2x2xf64>
  return %cst : tensor<2x2xf64>
}
// CHECK-LABEL: func.func @same_values
// CHECK-NEXT:    stablehlo.constant
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00]
// CHECK-SAME:      ]
// CHECK-NEXT:    return

func.func @different_values() -> tensor<2x2xf32> {
  %cst = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  return %cst : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @different_values
// CHECK-NEXT:    stablehlo.constant
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [1.000000e+00, 4.000000e+00, 2.500000e+00, 1.250000e+00]
// CHECK-SAME:      ]
// CHECK-NEXT:    return

func.func @negative_values() -> tensor<2x2xf32> {
  %cst = stablehlo.constant dense<[[-1.0, -2.0], [-3.0, -4.0]]> : tensor<2x2xf32>
  return %cst : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @negative_values
// CHECK-NEXT:    stablehlo.constant
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-4.000000e+00, -1.000000e+00, -2.500000e+00, 1.250000e+00]
// CHECK-SAME:      ]
// CHECK-NEXT:    return

// Non-tensor constants (e.g. from the arith dialect) are ignored.
func.func @scalar_constant() -> f64 {
  %cst = arith.constant 42.0 : f64
  return %cst : f64
}
// CHECK-LABEL: func.func @scalar_constant
// CHECK-NEXT:    arith.constant
// CHECK-NOT:       vishap.distribution
// CHECK-NEXT:    return
