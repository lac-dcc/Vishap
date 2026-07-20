// RUN: vishap-opt --add-probe-calls %s | FileCheck %s

func.func @no_dist_info(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>
  %res = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @no_dist_info
// CHECK-NEXT:    stablehlo.add
// CHECK-NEXT:    stablehlo.dot_general
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
// CHECK-NEXT:    stablehlo.multiply
// CHECK-NEXT:    %[[ADD_RES:.*]] = stablehlo.add
// CHECK-NEXT:    probe.observe(%[[ADD_RES]] : tensor<2x2xf32>) {opID = 0 : i32, resultID = 0 : i32}
// CHECK-NEXT:    probe.report
// CHECK-NEXT:    return
