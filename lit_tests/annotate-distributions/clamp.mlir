// RUN: vishap-opt --annotate-distributions %s | FileCheck %s

func.func @relu() -> tensor<3x2xf32> {
  %cst_0 = stablehlo.constant dense<0.0> : tensor<3x2xf32>
  %0 = stablehlo.constant dense<[[1.0, 2.0], [-3.0, -4.0], [5.0, -6.0]]> : tensor<3x2xf32>
  %res = stablehlo.maximum %0, %cst_0 : tensor<3x2xf32>
  return %res : tensor<3x2xf32>
}
// CHECK-LABEL: func.func @relu
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.maximum
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [0.000000e+00, 5.000000e+00, 1.1372720262294531, 3.7403129901231531]
// CHECK-SAME:      ]

// Same as @relu, but with the constant as the LHS operand.
func.func @relu_constant_lhs() -> tensor<3x2xf32> {
  %cst_0 = stablehlo.constant dense<0.0> : tensor<3x2xf32>
  %0 = stablehlo.constant dense<[[1.0, 2.0], [-3.0, -4.0], [5.0, -6.0]]> : tensor<3x2xf32>
  %res = stablehlo.maximum %cst_0, %0 : tensor<3x2xf32>
  return %res : tensor<3x2xf32>
}
// CHECK-LABEL: func.func @relu_constant_lhs
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.maximum
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [0.000000e+00, 5.000000e+00, 1.1372720262294531, 3.7403129901231531]
// CHECK-SAME:      ]

// Clamping with a non-zero constant.
func.func @clamp_nonzero(%arg0: tensor<3x2xf32>) -> tensor<3x2xf32>
            attributes {vishap.args_distribution = [[-6., 5., -0.83333333333333337, 13.472222222222221]]} {
  %cst_1 = stablehlo.constant dense<1.0> : tensor<3x2xf32>
  %res = stablehlo.maximum %arg0, %cst_1 : tensor<3x2xf32>
  return %res : tensor<3x2xf32>
}
// CHECK-LABEL: func.func @clamp_nonzero
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.maximum
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [1.000000e+00, 5.000000e+00, 1.7265871083335411, 2.2991275693075979]
// CHECK-SAME:      ]
