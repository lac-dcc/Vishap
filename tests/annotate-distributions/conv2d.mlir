// RUN: vishap-opt --annotate-distributions="per-axis=false" %s | FileCheck %s --check-prefixes=CHECK,TENSOR
// RUN: vishap-opt --annotate-distributions="per-axis=true annotate-axes=true" %s | FileCheck %s --check-prefixes=CHECK,SLICE

func.func @conv_2d_nchw_fchw() -> tensor<1x2x2x2xf32> {
  %input = stablehlo.constant dense<[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]> : tensor<1x2x2x2xf32>
  %kernel = stablehlo.constant dense<[[[[1.0]], [[2.0]]], [[[0.5]], [[1.5]]]]> : tensor<2x2x1x1xf32>
  %res = stablehlo.convolution(%input, %kernel)
          dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
          window = {stride = [1, 1], pad = [[0, 0], [0, 0]]}
          {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
          : (tensor<1x2x2x2xf32>, tensor<2x2x1x1xf32>) -> tensor<1x2x2x2xf32>
  return %res : tensor<1x2x2x2xf32>
}
// CHECK-LABEL: func.func @conv_2d_nchw_fchw
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.convolution
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          [],
// SLICE-SAME:          {{\[\[}}-10.219642299223374, 41.219642299223374, 1.550000e+01, 1.837500e+01], [-12.430749027719962, 34.430749027719962, 1.100000e+01, 1.525000e+01{{\]\]}},
// SLICE-SAME:          [],
// SLICE-SAME:          []
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// SLICE-SAME:      vishap.distribution = [
// SLICE-SAME:        [-12.430749027719962, 41.219642299223374, 1.325000e+01, 2.187500e+01]
// SLICE-SAME:      ]
// TENSOR-SAME:     vishap.distribution = [
// TENSOR-SAME:       [-22.872939498231972, 45.372939498231972, 1.125000e+01, 32.34375]
// TENSOR-SAME:     ]
// CHECK-NEXT:    return

// Same values in NHWC/HWIO. The per-tensor summary matches NCHW. Per-slice
// information follows the output feature axis, which is axis 1 in NCHW and
// axis 3 in NHWC.
func.func @conv_2d_nhwc_hwio() -> tensor<1x2x2x2xf32> {
  %input = stablehlo.constant dense<[[[[1.0, 5.0], [2.0, 6.0]], [[3.0, 7.0], [4.0, 8.0]]]]> : tensor<1x2x2x2xf32>
  %kernel = stablehlo.constant dense<[[[[1.0, 0.5], [2.0, 1.5]]]]> : tensor<1x1x2x2xf32>
  %res = stablehlo.convolution(%input, %kernel)
          dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
          window = {stride = [1, 1], pad = [[0, 0], [0, 0]]}
          {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
          : (tensor<1x2x2x2xf32>, tensor<1x1x2x2xf32>) -> tensor<1x2x2x2xf32>
  return %res : tensor<1x2x2x2xf32>
}
// CHECK-LABEL: func.func @conv_2d_nhwc_hwio
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.convolution
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          [],
// SLICE-SAME:          [],
// SLICE-SAME:          [],
// SLICE-SAME:          {{\[\[}}-10.219642299223374, 41.219642299223374, 1.550000e+01, 1.837500e+01], [-12.430749027719962, 34.430749027719962, 1.100000e+01, 1.525000e+01{{\]\]}}
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// SLICE-SAME:      vishap.distribution = [
// SLICE-SAME:        [-12.430749027719962, 41.219642299223374, 1.325000e+01, 2.187500e+01]
// SLICE-SAME:      ]
// TENSOR-SAME:     vishap.distribution = [
// TENSOR-SAME:       [-22.872939498231972, 45.372939498231972, 1.125000e+01, 32.34375]
// TENSOR-SAME:     ]
// CHECK-NEXT:    return
