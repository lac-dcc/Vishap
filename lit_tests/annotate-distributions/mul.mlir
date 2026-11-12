// RUN: vishap-opt --annotate-distributions="per-axis=false" %s | FileCheck --allow-unused-prefixes %s --check-prefixes=CHECK,TENSOR
// RUN: vishap-opt --annotate-distributions="per-axis=true annotate-axes=true" %s | FileCheck --allow-unused-prefixes %s --check-prefixes=CHECK,SLICE

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
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          {{\[\[}}3.000000e+00, 8.000000e+00, 5.250000e+00, 3.687500e+00], [3.000000e+00, 8.000000e+00, 5.250000e+00, 3.687500e+00{{\]\]}},
// SLICE-SAME:          {{\[\[}}2.000000e+00, 1.200000e+01, 6.000000e+00, 1.400000e+01], [2.000000e+00, 1.200000e+01, 6.000000e+00, 1.400000e+01{{\]\]}}
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// SLICE-SAME:      vishap.distribution = [
// SLICE-SAME:        [3.000000e+00, 8.000000e+00, 5.250000e+00, 3.687500e+00]
// SLICE-SAME:      ]
// TENSOR-SAME:     vishap.distribution = [
// TENSOR-SAME:       [1.000000e+00, 1.600000e+01, 6.250000e+00, 1.718750e+01]
// TENSOR-SAME:     ]
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
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          {{\[\[}}1.000000e+00, 2.000000e+00, 1.500000e+00, 2.500000e-01], [3.000000e+00, 4.000000e+00, 3.500000e+00, 2.500000e-01{{\]\]}},
// SLICE-SAME:          {{\[\[}}1.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00], [2.000000e+00, 4.000000e+00, 3.000000e+00, 1.000000e+00{{\]\]}}
// SLICE-SAME:        ]
// SLICE-SAME:      ]
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
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          {{\[\[}}0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00{{\]\]}},
// SLICE-SAME:          {{\[\[}}0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00{{\]\]}}
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
// CHECK-SAME:      ]
// CHECK-NEXT:    return
