// RUN: vishap-opt --annotate-distributions %s | FileCheck %s

// Decomposed batch normalization, as produced by ONNX -> StableHLO:
//   y = gamma * (x - running_mean) * rsqrt(running_var + eps) + beta
// The rsqrt over the constant running variance is constant-folded, and the
// normalize multiply propagates the running statistics stored in the graph
// (mean 0, variance v_c / (v_c + eps)) instead of the abstract product
// transfer function.
//
// Here x has range [0, 4]; running_mean = {1, 3}, running_var = {4, 1},
// eps = 0, so rsqrt = {0.5, 1.0} and the normalized output has
// min = (0 - 3) * 1 = -3, max = (4 - 1) * 0.5 = 1.5, mean 0, variance 1.
func.func @batchnorm() -> tensor<1x2x2x2xf32> {
  %x = stablehlo.constant dense<[[[[0.0, 4.0], [2.0, 2.0]], [[1.0, 3.0], [2.0, 2.0]]]]> : tensor<1x2x2x2xf32>
  %gamma = stablehlo.constant dense<[2.0, 0.5]> : tensor<2xf32>
  %beta = stablehlo.constant dense<[0.5, -0.5]> : tensor<2xf32>
  %mean = stablehlo.constant dense<[1.0, 3.0]> : tensor<2xf32>
  %var = stablehlo.constant dense<[4.0, 1.0]> : tensor<2xf32>
  %eps = stablehlo.constant dense<0.0> : tensor<1xf64>
  %0 = stablehlo.reshape %mean : (tensor<2xf32>) -> tensor<1x2x1x1xf32>
  %1 = stablehlo.reshape %var : (tensor<2xf32>) -> tensor<1x2x1x1xf32>
  %2 = stablehlo.convert %eps : (tensor<1xf64>) -> tensor<1xf32>
  %3 = stablehlo.reshape %2 : (tensor<1xf32>) -> tensor<f32>
  %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x2x1x1xf32>
  %5 = stablehlo.add %1, %4 : tensor<1x2x1x1xf32>
  %6 = stablehlo.rsqrt %5 : tensor<1x2x1x1xf32>
  %7 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x2x1x1xf32>) -> tensor<1x2x2x2xf32>
  %8 = stablehlo.subtract %x, %7 : tensor<1x2x2x2xf32>
  %9 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2, 3] : (tensor<1x2x1x1xf32>) -> tensor<1x2x2x2xf32>
  %10 = stablehlo.multiply %8, %9 : tensor<1x2x2x2xf32>
  %11 = stablehlo.reshape %gamma : (tensor<2xf32>) -> tensor<1x2x1x1xf32>
  %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2, 3] : (tensor<1x2x1x1xf32>) -> tensor<1x2x2x2xf32>
  %13 = stablehlo.multiply %10, %12 : tensor<1x2x2x2xf32>
  %14 = stablehlo.reshape %beta : (tensor<2xf32>) -> tensor<1x2x1x1xf32>
  %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1, 2, 3] : (tensor<1x2x1x1xf32>) -> tensor<1x2x2x2xf32>
  %16 = stablehlo.add %13, %15 : tensor<1x2x2x2xf32>
  return %16 : tensor<1x2x2x2xf32>
}
// CHECK-LABEL: func.func @batchnorm

// CHECK:         stablehlo.rsqrt
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [5.000000e-01, 1.000000e+00, 0.71783702885822209, 3.600000e-02]
// CHECK-SAME:      ]

// Normalize multiply: running statistics propagated (mean 0, variance 1).
// CHECK:         stablehlo.multiply
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-3.000000e+00, 1.500000e+00, 0.000000e+00, 1.000000e+00]
// CHECK-SAME:      ]

// Gamma scaling flows through the abstract product transfer:
// variance = 1 * Var(gamma) + 1 * Mean(gamma)^2 = 0.5625 + 1.5625 = 2.125.
// CHECK:         stablehlo.multiply
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-6.000000e+00, 3.000000e+00, 0.000000e+00, 2.125000e+00]
// CHECK-SAME:      ]

// Beta shift: mean = Mean(beta) = 0, variance = 2.125 + Var(beta) = 2.375.
// CHECK:         stablehlo.add
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-6.500000e+00, 3.500000e+00, 0.000000e+00, 2.375000e+00]
// CHECK-SAME:      ]
