// RUN: vishap-opt --add-probe-calls %s | FileCheck %s --check-prefixes=CHECK,DIST-ONLY
// RUN: vishap-opt --add-probe-calls="all-float-ops" %s | FileCheck %s --check-prefixes=CHECK,ALL-OPS

func.func @no_dist_info(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>
  %res = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @no_dist_info
// CHECK-NEXT:    %[[ADD_RES:.*]] = stablehlo.add
// ALL-OPS-NEXT:  probe.observe(%[[ADD_RES]] : tensor<2x2xf32>) {opID = 0 : i32, resultID = 0 : i32}
// CHECK-NEXT:    %[[DOT_RES:.*]] = stablehlo.dot_general
// ALL-OPS-NEXT:  probe.observe(%[[DOT_RES]] : tensor<2x2xf32>) {opID = 1 : i32, resultID = 0 : i32}
// ALL-OPS-NEXT:  probe.report
// CHECK-NEXT:    return

func.func @with_dist_info(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.add %arg0, %arg1 {vishap.distribution = [[-2., 2., 0., 2.]]} : tensor<2x2xf32>
  %res = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0] {vishap.distribution = [[-12., 12., 0., 4.]]} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @with_dist_info
// CHECK-NEXT:    %[[ADD_RES:.*]] = stablehlo.add
// CHECK-NEXT:    probe.observe(%[[ADD_RES]] : tensor<2x2xf32>) {opID = 0 : i32, resultID = 0 : i32}
// CHECK-NEXT:    %[[DOT_RES:.*]] = stablehlo.dot_general
// CHECK-NEXT:    probe.observe(%[[DOT_RES]] : tensor<2x2xf32>) {opID = 1 : i32, resultID = 0 : i32}
// CHECK-NEXT:    probe.report
// CHECK-NEXT:    return

func.func @mixed(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<2x2xf32>
  %res = stablehlo.add %0, %arg2 {vishap.distribution = [[-2., 2., 0., 2.]]} : tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @mixed
// CHECK-NEXT:    %[[MUL_RES:.*]] = stablehlo.multiply
// ALL-OPS-NEXT:  probe.observe(%[[MUL_RES]] : tensor<2x2xf32>) {opID = 0 : i32, resultID = 0 : i32}
// CHECK-NEXT:    %[[ADD_RES:.*]] = stablehlo.add
// CHECK-NEXT:    probe.observe(%[[ADD_RES]] : tensor<2x2xf32>)
// DIST-ONLY-SAME:  {opID = 0 : i32, resultID = 0 : i32}
// ALL-OPS-SAME:    {opID = 1 : i32, resultID = 0 : i32}
// CHECK-NEXT:    probe.report
// CHECK-NEXT:    return

func.func @with_constant() -> tensor<2x2xf32> {
  %c = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %res = stablehlo.add %c, %c : tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @with_constant
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    %[[ADD_RES:.*]] = stablehlo.add
// ALL-OPS-NEXT:  probe.observe(%[[ADD_RES]] : tensor<2x2xf32>) {opID = 0 : i32, resultID = 0 : i32}
// ALL-OPS-NEXT:  probe.report
// CHECK-NEXT:    return

func.func @nested_ops(%input: tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xf32> {
  %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %res = "stablehlo.reduce_window"(%input, %init) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %max = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %max : tensor<f32>
  }) {
    window_dimensions = array<i64: 1, 1, 3, 3>,
    window_strides = array<i64: 1, 1, 2, 2>,
    padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>
  } : (tensor<1x64x112x112xf32>, tensor<f32>) -> tensor<1x64x56x56xf32>
  return %res : tensor<1x64x56x56xf32>
}
// CHECK-LABEL: func.func @nested_ops
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    %[[RES:.*]] = "stablehlo.reduce_window"
// ALL-OPS:           stablehlo.maximum
// ALL-OPS-NOT:       probe.observe
// ALL-OPS:           stablehlo.return
// ALL-OPS:       probe.observe(%[[RES]] : tensor<1x64x56x56xf32>)
// ALL-OPS:       probe.report
// CHECK-NOT:     probe.report
// CHECK:         return %[[RES]]
