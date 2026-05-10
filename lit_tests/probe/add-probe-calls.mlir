// RUN: vishap-opt --add-probe-calls %s | FileCheck %s

func.func @no_dist_info(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = tensor.empty() : tensor<2x2xf32>
  %1 = linalg.add ins(%arg0, %arg1 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%0: tensor<2x2xf32>) -> tensor<2x2xf32>
  %2 = tensor.empty() : tensor<2x2xf32>
  %res = linalg.matmul ins(%1, %arg2 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%2: tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @no_dist_info
// CHECK-NEXT:    tensor.empty
// CHECK-NEXT:    linalg.add
// CHECK-NEXT:    tensor.empty
// CHECK-NEXT:    linalg.matmul
// CHECK-NEXT:    return

func.func @with_dist_info(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = tensor.empty() : tensor<2x2xf32>
  %1 = linalg.add {vishap.distribution = [[-2., 2., 0., 2.]]} ins(%arg0, %arg1 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%0 : tensor<2x2xf32>) -> tensor<2x2xf32>
  %2 = tensor.empty() : tensor<2x2xf32>
  %res = linalg.matmul {vishap.distribution = [[-12., 12., 0., 4.]]} ins(%1, %arg2 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%2 : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @with_dist_info
// CHECK-NEXT:    tensor.empty
// CHECK-NEXT:    %[[ADD_RES:.*]] = linalg.add
// CHECK-NEXT:    probe.observe(%[[ADD_RES]] : tensor<2x2xf32>) {opID = 0 : i32, resultID = 0 : i32}
// CHECK-NEXT:    tensor.empty
// CHECK-NEXT:    %[[MATMUL_RES:.*]] = linalg.matmul
// CHECK-NEXT:    probe.observe(%[[MATMUL_RES]] : tensor<2x2xf32>) {opID = 1 : i32, resultID = 0 : i32}
// CHECK-NEXT:    probe.report
// CHECK-NEXT:    return

func.func @mixed(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = tensor.empty() : tensor<2x2xf32>
  %1 = linalg.mul ins(%arg0, %arg1 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%0 : tensor<2x2xf32>) -> tensor<2x2xf32>
  %2 = tensor.empty() : tensor<2x2xf32>
  %res = linalg.add {vishap.distribution = [[-2., 2., 0., 2.]]} ins(%1, %arg2 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%2 : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @mixed
// CHECK-NEXT:    tensor.empty
// CHECK-NEXT:    linalg.mul
// CHECK-NEXT:    tensor.empty
// CHECK-NEXT:    %[[ADD_RES:.*]] = linalg.add
// CHECK-NEXT:    probe.observe(%[[ADD_RES]] : tensor<2x2xf32>) {opID = 0 : i32, resultID = 0 : i32}
// CHECK-NEXT:    probe.report
// CHECK-NEXT:    return
