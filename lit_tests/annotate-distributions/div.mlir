// RUN: split-file %s %t
// RUN: vishap-opt --annotate-distributions %t/constant.mlir | FileCheck %t/constant.mlir
// RUN: vishap-opt --annotate-distributions="input-distributions=-1.,4.,1.5,1.;1.,3.,2.,0.25" %t/nonconst.mlir | FileCheck %t/nonconst.mlir

//--- constant.mlir
func.func @div() -> tensor<2x3xf32> {
  %two = stablehlo.constant dense<2.0> : tensor<2x3xf32>
  %cst = stablehlo.constant dense<[[0.42, 2., -1.], [4., 0.55, 2.]]> : tensor<2x3xf32>
  %res = stablehlo.divide %cst, %two : tensor<2x3xf32>
  return %res : tensor<2x3xf32>
}
// CHECK-LABEL: func.func @div
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.constant
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-1.000000e+00, 4.000000e+00, 1.3283333331346512, 2.4820138897664017]
// CHECK-SAME:      ]
// CHECK-NEXT:    stablehlo.divide
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-5.000000e-01, 2.000000e+00, 0.66416666656732559, 0.62050347244160042]
// CHECK-SAME:      ]

// The divisor may be a scalar constant broadcasted to the shape of the
// dividend.
func.func @div_broadcasted_divisor() -> tensor<2x3xf32> {
  %two = stablehlo.constant dense<2.0> : tensor<f32>
  %bcast = stablehlo.broadcast_in_dim %two, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
  %cst = stablehlo.constant dense<[[0.42, 2., -1.], [4., 0.55, 2.]]> : tensor<2x3xf32>
  %res = stablehlo.divide %cst, %bcast : tensor<2x3xf32>
  return %res : tensor<2x3xf32>
}
// CHECK-LABEL: func.func @div_broadcasted_divisor
// CHECK:         stablehlo.divide
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-5.000000e-01, 2.000000e+00, 0.66416666656732559, 0.62050347244160042]
// CHECK-SAME:      ]

// Softmax: exp(x - max(x)) / broadcast(sum(...)). The divide uses the
// dependent normalize path: [0, 1], mean = 1/N, variance = mean*(1 - mean).
func.func @softmax() -> tensor<1x4xf32> {
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %logits = stablehlo.constant dense<[[1.0, 2.0, 3.0, 4.0]]> : tensor<1x4xf32>
  %max = stablehlo.reduce(%logits init: %neg_inf) applies stablehlo.maximum across dimensions = [1] : (tensor<1x4xf32>, tensor<f32>) -> tensor<1xf32>
  %max_r = stablehlo.reshape %max : (tensor<1xf32>) -> tensor<1x1xf32>
  %max_b = stablehlo.broadcast_in_dim %max_r, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x4xf32>
  %shifted = stablehlo.subtract %logits, %max_b : tensor<1x4xf32>
  %exps = stablehlo.exponential %shifted : tensor<1x4xf32>
  %sum = stablehlo.reduce(%exps init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<1x4xf32>, tensor<f32>) -> tensor<1xf32>
  %sum_r = stablehlo.reshape %sum : (tensor<1xf32>) -> tensor<1x1xf32>
  %sum_b = stablehlo.broadcast_in_dim %sum_r, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x4xf32>
  %probs = stablehlo.divide %exps, %sum_b : tensor<1x4xf32>
  return %probs : tensor<1x4xf32>
}
// CHECK-LABEL: func.func @softmax
// CHECK:         stablehlo.divide
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [0.000000e+00, 1.000000e+00, 2.500000e-01, 1.875000e-01]
// CHECK-SAME:      ]

//--- nonconst.mlir
// Non-constant dividend and divisor: mean/variance via the independent
// delta-method approximation; min/max from interval corners.
func.func @div_nonconst(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %res = stablehlo.divide %arg0, %arg1 : tensor<2x3xf32>
  return %res : tensor<2x3xf32>
}
// CHECK-LABEL: func.func @div_nonconst
// CHECK:         stablehlo.divide
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-1.000000e+00, 4.000000e+00, 7.500000e-01, 0.28515625]
// CHECK-SAME:      ]
