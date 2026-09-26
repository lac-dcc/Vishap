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
// CHECK-SAME:        [0.000000e+00, 5.000000e+00, 1.1372720262294533, 3.7403129901231535]
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
// CHECK-SAME:        [0.000000e+00, 5.000000e+00, 1.1372720262294533, 3.7403129901231535]
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

// ReLU6-style two-sided clamp with bounds fed through broadcast_in_dim of
// scalar constants, as produced for MobileNetV2. Input N(1, 4) with range
// [-4, 10] clamped to [0, 6]: min 0, max 6, and winsorized moments
// mean ~= 1.3916, variance ~= 2.1720.
func.func @relu6(%arg0: tensor<3x2xf32>) -> tensor<3x2xf32>
            attributes {vishap.args_distribution = [[-4., 10., 1., 4.]]} {
  %cst_lo = stablehlo.constant dense<0.0> : tensor<f32>
  %cst_hi = stablehlo.constant dense<6.0> : tensor<f32>
  %lo = stablehlo.broadcast_in_dim %cst_lo, dims = [] : (tensor<f32>) -> tensor<3x2xf32>
  %hi = stablehlo.broadcast_in_dim %cst_hi, dims = [] : (tensor<f32>) -> tensor<3x2xf32>
  %res = stablehlo.clamp %lo, %arg0, %hi : tensor<3x2xf32>
  return %res : tensor<3x2xf32>
}
// CHECK-LABEL: func.func @relu6
// CHECK:         stablehlo.clamp
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [0.000000e+00, 6.000000e+00, 1.3915848404443558, 2.1720380099332228]
// CHECK-SAME:      ]

// General two-sided clamp with nonzero bounds and scalar bound operands.
// N(-0.8333, 13.47) with range [-6, 5] clamped to [-2, 3].
func.func @clamp_two_sided(%arg0: tensor<3x2xf32>) -> tensor<3x2xf32>
            attributes {vishap.args_distribution = [[-6., 5., -0.83333333333333337, 13.472222222222221]]} {
  %cst_lo = stablehlo.constant dense<-2.0> : tensor<f32>
  %cst_hi = stablehlo.constant dense<3.0> : tensor<f32>
  %res = stablehlo.clamp %cst_lo, %arg0, %cst_hi : (tensor<f32>, tensor<3x2xf32>, tensor<f32>) -> tensor<3x2xf32>
  return %res : tensor<3x2xf32>
}
// CHECK-LABEL: func.func @clamp_two_sided
// CHECK:         stablehlo.clamp
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-2.000000e+00, 3.000000e+00, -0.15983989613215421, 3.7766765405381482]
// CHECK-SAME:      ]
