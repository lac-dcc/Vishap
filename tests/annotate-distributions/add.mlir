// RUN: vishap-opt --annotate-distributions="per-axis=false" %s | FileCheck --allow-unused-prefixes %s --check-prefixes=CHECK,TENSOR
// RUN: vishap-opt --annotate-distributions="per-axis=true annotate-axes=true" %s | FileCheck --allow-unused-prefixes %s --check-prefixes=CHECK,SLICE

func.func @add() -> tensor<2x2xf32> {
  %0 = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %1 = stablehlo.constant dense<[[4.0, 3.0], [2.0, 1.0]]> : tensor<2x2xf32>
  %res = stablehlo.add %0, %1 : tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @add
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.add
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          {{\[\[}}4.000000e+00, 6.000000e+00, 5.000000e+00, 5.000000e-01], [4.000000e+00, 6.000000e+00, 5.000000e+00, 5.000000e-01{{\]\]}},
// SLICE-SAME:          {{\[\[}}3.000000e+00, 7.000000e+00, 5.000000e+00, 2.000000e+00], [3.000000e+00, 7.000000e+00, 5.000000e+00, 2.000000e+00{{\]\]}}
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// SLICE-SAME:      vishap.distribution = [
// SLICE-SAME:        [4.000000e+00, 6.000000e+00, 5.000000e+00, 5.000000e-01]
// SLICE-SAME:      ]
// TENSOR-SAME:     vishap.distribution = [
// TENSOR-SAME:       [2.000000e+00, 8.000000e+00, 5.000000e+00, 2.500000e+00]
// TENSOR-SAME:     ]
// CHECK-NEXT:    return

func.func @same_values() -> tensor<2x2xf64> {
  %0 = stablehlo.constant dense<[[1.0, 1.0], [1.0, 1.0]]> : tensor<2x2xf64>
  %1 = stablehlo.constant dense<[[1.0, 1.0], [1.0, 1.0]]> : tensor<2x2xf64>
  %res = stablehlo.add %0, %1 : tensor<2x2xf64>
  return %res : tensor<2x2xf64>
}
// CHECK-LABEL: func.func @same_values
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.add
// Equal values are a splat, so both axes stay uniform.
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          [],
// SLICE-SAME:          []
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [2.000000e+00, 2.000000e+00, 2.000000e+00, 0.000000e+00]
// CHECK-SAME:      ]
// CHECK-NEXT:    return
