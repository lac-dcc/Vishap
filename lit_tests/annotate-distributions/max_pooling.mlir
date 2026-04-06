// RUN: vishap-opt --annotate-distributions="input-distributions=-1.5,3.,1.,0.5;0.,1.,1.,0." %s | FileCheck %s

func.func @pooling_nchw_max(%input: tensor<1x48x32x32xf32>, %kernel: tensor<32x32xf32>) -> tensor<1x48x1x1xf32> {
  %out = tensor.empty() : tensor<1x48x1x1xf32>
  %res = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%input, %kernel : tensor<1x48x32x32xf32>, tensor<32x32xf32>) outs(%out : tensor<1x48x1x1xf32>) -> tensor<1x48x1x1xf32>
  return %res : tensor<1x48x1x1xf32>
}
// CHECK-LABEL: func.func @pooling_nchw_max
// CHECK-NEXT:    tensor.empty
// CHECK-NEXT:    linalg.pooling_nchw_max
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-1.500000e+00, 3.000000e+00, 1.500000e+00, 5.000000e-01]
// CHECK-SAME:      ]
// CHECK-NEXT:    return
