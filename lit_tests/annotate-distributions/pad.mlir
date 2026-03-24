// RUN: vishap-opt --annotate-distributions %s | FileCheck %s

func.func @pad_zero(%arg0: tensor<1x32x10x3xf32>) -> tensor<1x32x32x32xf32>
            attributes {vishap.args_distribution = [[-1., 12., 3., 4.]]} {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %padded = tensor.pad %arg0 low[0, 0, 10, 11] high[0, 0, 12, 18] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_0 : f32
  } : tensor<1x32x10x3xf32> to tensor<1x32x32x32xf32>
  return %padded : tensor<1x32x32x32xf32>
}
// CHECK-LABEL: func.func @pad
// CHECK-NEXT:    arith.constant
// CHECK-NEXT:    tensor.pad
// CHECK-NEXT:      ^bb0
// CHECK-NEXT:        tensor.yield
// CHECK-NEXT:      vishap.distribution = [
// CHECK-SAME:        [-1.000000e+00, 1.200000e+01, 0.087890625, 0.37313461303710938]
// CHECK-SAME:      ]

func.func @pad_one(%arg0: tensor<1x16x10x3xf32>) -> tensor<1x16x16x16xf32>
            attributes {vishap.args_distribution = [[-1., 1., 0.55, 0.05]]} {
  %cst_1 = arith.constant 1.000000e+00 : f32
  %padded = tensor.pad %arg0 low[0, 0, 3, 5] high[0, 0, 3, 8] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_1 : f32
  } : tensor<1x16x10x3xf32> to tensor<1x16x16x16xf32>
  return %padded : tensor<1x16x16x16xf32>
}
// CHECK-LABEL: func.func @pad
// CHECK-NEXT:    arith.constant
// CHECK-NEXT:    tensor.pad
// CHECK-NEXT:      ^bb0
// CHECK-NEXT:        tensor.yield
// CHECK-NEXT:      vishap.distribution = [
// CHECK-SAME:        [-1.000000e+00, 1.000000e+00, 0.947265625, 0.026808929443359369]
// CHECK-SAME:      ]
