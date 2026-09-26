// RUN: vishap-opt --annotate-distributions %s | FileCheck %s

func.func @rsqrt(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {vishap.args_distribution = [[1.0, 16.0, 4.0, 2.0]]} {
  %res = stablehlo.rsqrt %arg0 : tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @rsqrt
// CHECK:         stablehlo.rsqrt
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [2.500000e-01, 1.000000e+00, 0.5234375, 7.812500e-03]
// CHECK-SAME:      ]
