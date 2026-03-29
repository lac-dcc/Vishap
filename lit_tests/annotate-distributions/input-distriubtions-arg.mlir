// RUN: vishap-opt --annotate-distributions="input-distributions=1.,2.,3.,4.;5.,6.,7.,8.;9.,10.,11.,12." %s | FileCheck %s

func.func @arg_test(%arg0: tensor<1x2x3xf32>, %arg1: tensor<3x2x1xf64>, %arg2: tensor<10xf32>) -> (tensor<1x2x3xf32>, tensor<3x2x1xf64>, tensor<10xf32>) {
  return %arg0, %arg1, %arg2 : tensor<1x2x3xf32>, tensor<3x2x1xf64>, tensor<10xf32>
}
// CHECK-LABEL: func.func @arg_test
// CHECK-SAME:      vishap.args_distribution = [
// CHECK-SAME:        [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00],
// CHECK-SAME:        [5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00],
// CHECK-SAME:        [9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]
// CHECK-SAME:      ]
