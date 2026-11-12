// RUN: vishap-opt --annotate-distributions="per-axis=false" %s | FileCheck --allow-unused-prefixes %s --check-prefixes=CHECK,TENSOR
// RUN: vishap-opt --annotate-distributions="per-axis=true annotate-axes=true" %s | FileCheck --allow-unused-prefixes %s --check-prefixes=CHECK,SLICE

func.func @broadcast() -> tensor<1x64x3x3xf32> {
  %cst = stablehlo.constant dense<[[0.0, 2.0, 4.0], [6.0, 8.0, -10.0], [0.0, 2.0, 0.0]]> : tensor<3x3xf32>
  %broadcasted = stablehlo.broadcast_in_dim %cst, dims = [2, 3] : (tensor<3x3xf32>) -> tensor<1x64x3x3xf32>
  return %broadcasted : tensor<1x64x3x3xf32>
}
// CHECK-LABEL: func.func @broadcast
// CHECK-NEXT:    stablehlo.constant
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-1.000000e+01, 8.000000e+00, 1.3333333333333333, 23.111111111111114]
// CHECK-SAME:      ]
// CHECK-NEXT:    stablehlo.broadcast_in_dim
// Input axes land on output axes 2 and 3. Leading axes are new and uniform.
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          [],
// SLICE-SAME:          [],
// SLICE-SAME:          {{\[\[}}0.000000e+00, 4.000000e+00, 2.000000e+00, 2.6666666666666665], [-1.000000e+01, 8.000000e+00, 1.3333333333333333, 64.8888888888889], [0.000000e+00, 2.000000e+00, 0.66666666666666663, 0.88888888888888895{{\]\]}},
// SLICE-SAME:          {{\[\[}}0.000000e+00, 6.000000e+00, 2.000000e+00, 8.000000e+00], [2.000000e+00, 8.000000e+00, 4.000000e+00, 8.000000e+00], [-1.000000e+01, 4.000000e+00, -2.000000e+00, 34.666666666666664{{\]\]}}
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-1.000000e+01, 8.000000e+00, 1.3333333333333333, 23.111111111111114]
// CHECK-SAME:      ]

func.func @transpose() -> tensor<1x3x2xf32> {
  %cst = stablehlo.constant dense<[[[1.0, 0.0, -2.0], [3.0, 0.1, -0.1]]]> : tensor<1x2x3xf32>
  %tranposed = stablehlo.transpose %cst, dims = [0, 2, 1] : (tensor<1x2x3xf32>) -> tensor<1x3x2xf32>
  return %tranposed : tensor<1x3x2xf32>
}
// CHECK-LABEL: func.func @transpose
// CHECK-NEXT:    stablehlo.constant
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-2.000000e+00, 3.000000e+00, 0.33333333333333331, 2.2255555556548967]
// CHECK-SAME:      ]
// CHECK-NEXT:    stablehlo.transpose
// Permutation [0, 2, 1] swaps the last two axes.
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          [],
// SLICE-SAME:          {{\[\[}}1.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00], [0.000000e+00, 0.10000000149011612, 0.05000000074505806, 0.0025000000745058065], [-2.000000e+00, -0.10000000149011612, -1.0500000007450581, 0.90249999858438967{{\]\]}},
// SLICE-SAME:          {{\[\[}}-2.000000e+00, 1.000000e+00, -0.33333333333333331, 1.5555555555555556], [-0.10000000149011612, 3.000000e+00, 1.000000e+00, 2.0066666668653483{{\]\]}}
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-2.000000e+00, 3.000000e+00, 0.33333333333333331, 2.2255555556548967]
// CHECK-SAME:      ]

func.func @reshape() -> tensor<2x3x4xf64> {
  %cst = stablehlo.constant dense<[[[[1.2, 1.5, 0.3, 1.4],[0.5, 1.7, 1.3, 2.],[0.1, 0.9, 0.6, 0.7]],[[2.3, 1.9, 1.1, 1.8],[0.2, 1.6, 2.1, 0.8],[1., 0.4, 2.2, 0.]]]]> : tensor<1x2x3x4xf64>
  %reshaped = stablehlo.reshape %cst : (tensor<1x2x3x4xf64>) -> tensor<2x3x4xf64>
  return %reshaped : tensor<2x3x4xf64>
}
// CHECK-LABEL: func.func @reshape
// CHECK-NEXT:    stablehlo.constant
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [0.000000e+00, 2.300000e+00, 1.1500000000000001, 0.47916666666666674]
// CHECK-SAME:      ]
// CHECK-NEXT:    stablehlo.reshape
// The leading size-1 axis is dropped; the other three axes are relabeled.
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          {{\[\[}}1.000000e-01, 2.000000e+00, 1.0166666666666666, 0.31972222222222219], [0.000000e+00, 2.300000e+00, 1.2833333333333332, 0.60305555555555557{{\]\]}},
// SLICE-SAME:          {{\[\[}}3.000000e-01, 2.300000e+00, 1.437500e+00, 0.31984374999999993], [2.000000e-01, 2.100000e+00, 1.2750000000000001, 0.43437499999999996], [0.000000e+00, 2.200000e+00, 7.375000e-01, 0.41484375000000007{{\]\]}},
// SLICE-SAME:          {{\[\[}}1.000000e-01, 2.300000e+00, 0.8833333333333333, 0.55805555555555553], [4.000000e-01, 1.900000e+00, 1.3333333333333333, 0.2688888888888889], [3.000000e-01, 2.200000e+00, 1.2666666666666668, 0.49555555555555558], [0.000000e+00, 2.000000e+00, 1.1166666666666665, 0.47472222222222227{{\]\]}}
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [0.000000e+00, 2.300000e+00, 1.1500000000000001, 0.47916666666666674]
// CHECK-SAME:      ]

func.func @slice() -> tensor<2x2xf32> {
  %cst = stablehlo.constant dense<[[1., 0.9, 0.41, 0.05], [-0.66, 0.03, 0.5, 0.98], [0., 0., 0., 0.]]> : tensor<3x4xf32>
  %sliced = stablehlo.slice %cst [1:3, 2:4] : (tensor<3x4xf32>) -> tensor<2x2xf32>
  return %sliced : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @slice
// CHECK-NEXT:    stablehlo.constant
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-0.6600000262260437, 1.000000e+00, 0.26749999712531763, 0.23073542038702108]
// CHECK-SAME:      ]
// CHECK-NEXT:    stablehlo.slice
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          {{\[\[}}-0.6600000262260437, 0.98000001907348633, 0.21249999804422259, 0.36656876882175005], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00{{\]\]}},
// SLICE-SAME:          {{\[\[}}0.000000e+00, 5.000000e-01, 0.30333333214124042, 0.047355555301242407], [0.000000e+00, 0.98000001907348633, 0.34333333993951481, 0.203088896838824{{\]\]}}
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// SLICE-SAME:      vishap.distribution = [
// SLICE-SAME:        [0.000000e+00, 0.98000001907348633, 0.1062499990221113, 0.19457344670307369]
// SLICE-SAME:      ]
// TENSOR-SAME:     vishap.distribution = [
// TENSOR-SAME:       [-0.6600000262260437, 1.000000e+00, 0.26749999712531763, 0.23073542038702108]
// TENSOR-SAME:     ]

func.func @gather() -> tensor<1x2xf32> {
  %operand = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]> : tensor<3x3xf32>
  %indices = stablehlo.constant dense<[1]> : tensor<1xi64>
  %result = "stablehlo.gather"(%operand, %indices) <{
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1
    >,
    slice_sizes = array<i64: 1, 2>,
    indices_are_sorted = false
  }> : (tensor<3x3xf32>, tensor<1xi64>) -> tensor<1x2xf32>
  return %result : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @gather
// CHECK-NEXT:    stablehlo.constant
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [1.000000e+00, 9.000000e+00, 5.000000e+00, 6.666666666666667]
// CHECK-SAME:      ]
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.gather
// Gather does not track per-axis information, so the result is uniform.
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          [],
// SLICE-SAME:          []
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [1.000000e+00, 9.000000e+00, 5.000000e+00, 6.666666666666667]
// CHECK-SAME:      ]
