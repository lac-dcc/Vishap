// RUN: vishap-opt --annotate-distributions="per-axis=false" %s | FileCheck %s
// RUN: vishap-opt --annotate-distributions="per-axis=true annotate-axes=true" %s | FileCheck %s --check-prefixes=CHECK,SLICE

func.func @i64_to_f32_splat() -> tensor<1xf32> {
  %cst = arith.constant dense<49> : tensor<1xi64>
  %0 = stablehlo.convert %cst : (tensor<1xi64>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}
// CHECK-LABEL: func.func @i64_to_f32_splat
// CHECK-NEXT:    arith.constant
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          []
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [4.900000e+01, 4.900000e+01, 4.900000e+01, 0.000000e+00]
// CHECK-SAME:      ]
// CHECK-NEXT:    stablehlo.convert
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          []
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [4.900000e+01, 4.900000e+01, 4.900000e+01, 0.000000e+00]
// CHECK-SAME:      ]
// CHECK-NEXT:    return

func.func @i32_to_f32() -> tensor<2x2xf32> {
  %cst = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %0 = stablehlo.convert %cst : (tensor<2x2xi32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @i32_to_f32
// CHECK-NEXT:    stablehlo.constant
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          {{\[\[}}1.000000e+00, 2.000000e+00, 1.500000e+00, 2.500000e-01], [3.000000e+00, 4.000000e+00, 3.500000e+00, 2.500000e-01{{\]\]}},
// SLICE-SAME:          {{\[\[}}1.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00], [2.000000e+00, 4.000000e+00, 3.000000e+00, 1.000000e+00{{\]\]}}
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [1.000000e+00, 4.000000e+00, 2.500000e+00, 1.250000e+00]
// CHECK-SAME:      ]
// CHECK-NEXT:    stablehlo.convert
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

func.func @f64_to_f32() -> tensor<2x2xf32> {
  %cst = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
  %0 = stablehlo.convert %cst : (tensor<2x2xf64>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @f64_to_f32
// CHECK-NEXT:    stablehlo.constant
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          {{\[\[}}1.000000e+00, 2.000000e+00, 1.500000e+00, 2.500000e-01], [3.000000e+00, 4.000000e+00, 3.500000e+00, 2.500000e-01{{\]\]}},
// SLICE-SAME:          {{\[\[}}1.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00], [2.000000e+00, 4.000000e+00, 3.000000e+00, 1.000000e+00{{\]\]}}
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [1.000000e+00, 4.000000e+00, 2.500000e+00, 1.250000e+00]
// CHECK-SAME:      ]
// CHECK-NEXT:    stablehlo.convert
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
