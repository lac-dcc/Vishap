// RUN: vishap-opt --annotate-distributions="input-distributions=-0.5,20.,2.,2.5;0.5,2.,1.,0.25" %s | FileCheck %s

func.func @conv_2d_nchw_fchw(%input: tensor<1x3x34x34xf32>, %kernel: tensor<16x3x3x3xf32>) -> tensor<1x16x32x32xf32> {
  %res = stablehlo.convolution(%input, %kernel)
          dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
          window = {stride = [1, 1], pad = [[0, 0], [0, 0]]}
          {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
          : (tensor<1x3x34x34xf32>, tensor<16x3x3x3xf32>) -> tensor<1x16x32x32xf32>
  return %res : tensor<1x16x32x32xf32>
}
// CHECK-LABEL: func.func @conv_2d_nchw_fchw
// CHECK-SAME:      vishap.args_distribution
// CHECK-NEXT:    stablehlo.convolution
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-9.3206127576163027, 117.3206127576163, 5.400000e+01, 1.113750e+02]
// CHECK-SAME:      ]
// CHECK-NEXT:    return

// Same convolution but with NHWC/HWIO layout. The result must be the same,
// since the analysis is layout-agnostic.
func.func @conv_2d_nhwc_hwio(%input: tensor<1x34x34x3xf32>, %kernel: tensor<3x3x3x16xf32>) -> tensor<1x32x32x16xf32> {
  %res = stablehlo.convolution(%input, %kernel)
          dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
          window = {stride = [1, 1], pad = [[0, 0], [0, 0]]}
          {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
          : (tensor<1x34x34x3xf32>, tensor<3x3x3x16xf32>) -> tensor<1x32x32x16xf32>
  return %res : tensor<1x32x32x16xf32>
}
// CHECK-LABEL: func.func @conv_2d_nhwc_hwio
// CHECK-SAME:      vishap.args_distribution
// CHECK-NEXT:    stablehlo.convolution
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-9.3206127576163027, 117.3206127576163, 5.400000e+01, 1.113750e+02]
// CHECK-SAME:      ]
// CHECK-NEXT:    return
