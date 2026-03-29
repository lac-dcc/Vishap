// RUN: vishap-opt --annotate-distributions="input-distributions=-0.5,20.,2.,2.5;0.5,2.,1.,0.25" %s | FileCheck %s

func.func @conv_2d_nchw_fchw(%input: tensor<1x3x34x34xf32>, %kernel: tensor<16x3x3x3xf32>) -> tensor<1x16x32x32xf32> {
  %out = tensor.empty() : tensor<1x16x32x32xf32>
  %res = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
          ins(%input, %kernel : tensor<1x3x34x34xf32>, tensor<16x3x3x3xf32>) outs(%out : tensor<1x16x32x32xf32>) -> tensor<1x16x32x32xf32>
  return %res : tensor<1x16x32x32xf32>
}
// CHECK-LABEL: func.func @conv_2d_nchw_fchw
// CHECK-SAME:      vishap.args_distribution
// CHECK-NEXT:    tensor.empty
// CHECK-NEXT:    linalg.conv_2d_nchw_fchw
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-9.3206127576163027, 117.3206127576163, 5.400000e+01, 1.113750e+02]
// CHECK-SAME:      ]
// CHECK-NEXT:    return
