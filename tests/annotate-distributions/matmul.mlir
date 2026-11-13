// RUN: vishap-opt --annotate-distributions="per-axis=false" %s | FileCheck --allow-unused-prefixes %s --check-prefixes=CHECK,TENSOR
// RUN: vishap-opt --annotate-distributions="per-axis=true annotate-axes=true" %s | FileCheck --allow-unused-prefixes %s --check-prefixes=CHECK,SLICE

func.func @square_matrices() -> tensor<2x2xf32> {
  %0 = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %1 = stablehlo.constant dense<[[4.0, 3.0], [2.0, 1.0]]> : tensor<2x2xf32>
  %res = stablehlo.dot_general %0, %1, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @square_matrices
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.dot_general
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          {{\[\[}}-6.7476412994917716, 19.747641299491772, 6.500000e+00, 4.875000e+00], [-2.3547076349648037, 35.354707634964804, 1.650000e+01, 9.875000e+00{{\]\]}},
// SLICE-SAME:          {{\[\[}}-20.467375879228172, 48.467375879228172, 1.400000e+01, 3.300000e+01], [-19.774989139876315, 37.774989139876311, 9.000000e+00, 2.300000e+01{{\]\]}}
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// SLICE-SAME:      vishap.distribution = [
// SLICE-SAME:        [-6.7476412994917716, 35.354707634964804, 1.150000e+01, 3.237500e+01]
// SLICE-SAME:      ]
// TENSOR-SAME:     vishap.distribution = [
// TENSOR-SAME:       [-22.678118198675726, 47.678118198675726, 1.250000e+01, 3.437500e+01]
// TENSOR-SAME:     ]
// CHECK-NEXT:    return

func.func @different_shapes() -> tensor<2x2xf32> {
  %0 = stablehlo.constant dense<[[4.0, 3.0, 2.0], [1.0, 0.0, -1.0]]> : tensor<2x3xf32>
  %1 = stablehlo.constant dense<[[1.0, 0.0], [-0.5, 3.0], [2.0, 4.0]]> : tensor<3x2xf32>
  %res = stablehlo.dot_general %0, %1, contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @different_shapes
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.dot_general
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          {{\[\[}}-30.205333391596348, 53.705333391596348, 1.175000e+01, 48.895833333333329], [-21.55255888325765, 16.55255888325765, -2.500000e+00, 10.083333333333334{{\]\]}},
// SLICE-SAME:          {{\[\[}}-26.277259217001607, 28.777259217001607, 1.250000e+00, 21.048611111111111], [-46.175640282326157, 62.175640282326157, 8.000000e+00, 81.527777777777785{{\]\]}}
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// SLICE-SAME:      vishap.distribution = [
// SLICE-SAME:        [-30.205333391596348, 53.705333391596348, 4.625000e+00, 80.255208333333328]
// SLICE-SAME:      ]
// TENSOR-SAME:     vishap.distribution = [
// TENSOR-SAME:       [-39.822444019882482, 54.072444019882482, 7.125000e+00, 61.223958333333321]
// TENSOR-SAME:     ]
// CHECK-NEXT:    return

// Batched matmul: the batch dimension must not affect the contraction size.
// Both operands are splats, so the result stays uniform along every axis.
func.func @batched() -> tensor<4x2x2xf32> {
  %0 = stablehlo.constant dense<1.0> : tensor<4x2x2xf32>
  %1 = stablehlo.constant dense<1.0> : tensor<4x2x2xf32>
  %res = stablehlo.dot_general %0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x2x2xf32>, tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
  return %res : tensor<4x2x2xf32>
}
// CHECK-LABEL: func.func @batched
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.dot_general
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          [],
// SLICE-SAME:          [],
// SLICE-SAME:          []
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [2.000000e+00, 2.000000e+00, 2.000000e+00, 0.000000e+00]
// CHECK-SAME:      ]
// CHECK-NEXT:    return
