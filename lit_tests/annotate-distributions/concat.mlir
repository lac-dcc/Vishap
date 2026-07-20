// RUN: vishap-opt --annotate-distributions %s | FileCheck %s

func.func @concat_2d() -> tensor<6x2xf64> {
  %0 = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
  %1 = stablehlo.constant dense<[[5.0, 6.0]]> : tensor<1x2xf64>
  %2 = stablehlo.constant dense<[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]> : tensor<3x2xf64>
  %concat = stablehlo.concatenate %0, %1, %2, dim = 0 : (tensor<2x2xf64>, tensor<1x2xf64>, tensor<3x2xf64>) -> tensor<6x2xf64>
  return %concat : tensor<6x2xf64>
}
// CHECK-LABEL: func.func @concat_2d
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.concatenate
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [1.000000e+00, 1.200000e+01, 6.500000e+00, 11.916666666666666]
// CHECK-SAME:      ]
// CHECK-NEXT:    return

func.func @concat_3d() -> tensor<5x10x2xf32> {
  %0 = stablehlo.constant dense<0.0> : tensor<5x2x2xf32>
  %1 = stablehlo.constant dense<1.0> : tensor<5x8x2xf32>
  %concat = stablehlo.concatenate %0, %1, dim = 1 : (tensor<5x2x2xf32>, tensor<5x8x2xf32>) -> tensor<5x10x2xf32>
  return %concat : tensor<5x10x2xf32>
}
// CHECK-LABEL: func.func @concat_3d
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.concatenate
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [0.000000e+00, 1.000000e+00, 8.000000e-01, 0.16000000000000003]
// CHECK-SAME:      ]
// CHECK-NEXT:    return
