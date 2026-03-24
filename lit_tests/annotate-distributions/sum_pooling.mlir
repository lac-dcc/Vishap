func.func @pooling_nchw_sum(%input: tensor<1x48x32x32xf32>, %kernel: tensor<32x32xf32>) -> tensor<1x48x1x1xf32> 
            attributes {vishap.args_distribution = [[-1.5, 3., 1., 0.5], [0., 1., 1., 0.]]} {
  %out = tensor.empty() : tensor<1x48x1x1xf32>
  %res = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%input, %kernel : tensor<1x48x32x32xf32>, tensor<32x32xf32>) outs(%out : tensor<1x48x1x1xf32>) -> tensor<1x48x1x1xf32>
  return %res : tensor<1x48x1x1xf32>
}
// CHECK-LABEL: func.func @pooling_nchw_sum
// CHECK-NEXT:    tensor.empty
// CHECK-NEXT:    linalg.pooling_nchw_sum
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-1.536000e+03, 3.072000e+03, 1.024000e+03, 5.120000e+02]
// CHECK-SAME:      ]
// CHECK-NEXT:    return
