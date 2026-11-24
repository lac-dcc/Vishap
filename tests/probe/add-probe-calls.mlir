// RUN: vishap-opt --add-probe-calls %s | FileCheck %s --check-prefixes=CHECK,DIST-ONLY
// RUN: vishap-opt --add-probe-calls="all-float-ops" %s | FileCheck %s --check-prefixes=CHECK,ALL-OPS
// RUN: vishap-opt --add-probe-calls="all-float-ops report-file=my_report.csv" %s | FileCheck %s --check-prefix=REPORT-FILE
// RUN: vishap-opt --add-probe-calls="all-float-ops observe-func-args skip-redundant-ops" %s | FileCheck %s --check-prefix=CALIB

// The report file path is forwarded to the runtime via the reportArgs
// attribute of probe.report.
// REPORT-FILE: probe.report() {reportArgs = ["my_report.csv"]}

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
// ALL-OPS-NEXT:  probe.report() {reportArgs = ["vishap_probe_report.csv"]}
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

func.func @args_and_redundant(%arg0: tensor<2x2xf32>, %idx: tensor<1xi64>, %arg2: tensor<2xf32>) -> tensor<2x2xf32> {
  %cst_i32 = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %cst_f32 = stablehlo.convert %cst_i32 : (tensor<2x2xi32>) -> tensor<2x2xf32>
  %bcast = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<2xf32>) -> tensor<2x2xf32>
  %trans = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %sum = stablehlo.add %trans, %bcast : tensor<2x2xf32>
  %out = stablehlo.multiply %sum, %cst_f32 : tensor<2x2xf32>
  %reshaped = stablehlo.reshape %out : (tensor<2x2xf32>) -> tensor<4xf32>
  %final = stablehlo.reshape %reshaped : (tensor<4xf32>) -> tensor<2x2xf32>
  return %final : tensor<2x2xf32>
}
// CALIB-LABEL: func.func @args_and_redundant(
// CALIB-SAME:    %[[ARG0:.*]]: tensor<2x2xf32>, %[[IDX:.*]]: tensor<1xi64>, %[[ARG2:.*]]: tensor<2xf32>)
// CALIB-NEXT:    probe.observe(%[[ARG0]] : tensor<2x2xf32>) {opID = -1 : i32, resultID = 0 : i32}
// CALIB-NEXT:    probe.observe(%[[ARG2]] : tensor<2xf32>) {opID = -1 : i32, resultID = 2 : i32}
// CALIB-NEXT:    stablehlo.constant
// CALIB-NEXT:    stablehlo.convert
// CALIB-NEXT:    stablehlo.broadcast_in_dim
// CALIB-NEXT:    stablehlo.transpose
// CALIB-NEXT:    %[[SUM:.*]] = stablehlo.add
// CALIB-NEXT:    probe.observe(%[[SUM]] : tensor<2x2xf32>) {opID = 0 : i32, resultID = 0 : i32}
// CALIB-NEXT:    %[[OUT:.*]] = stablehlo.multiply
// CALIB-NEXT:    probe.observe(%[[OUT]] : tensor<2x2xf32>) {opID = 1 : i32, resultID = 0 : i32}
// CALIB-NEXT:    stablehlo.reshape
// CALIB-NEXT:    stablehlo.reshape
// CALIB-NEXT:    probe.report() {reportArgs = ["vishap_probe_report.csv"]}
// CALIB-NEXT:    return
// ALL-OPS-LABEL: func.func @args_and_redundant(
// ALL-OPS-NEXT:    stablehlo.constant
// ALL-OPS-NEXT:    %[[CVT:.*]] = stablehlo.convert
// ALL-OPS-NEXT:    probe.observe(%[[CVT]] : tensor<2x2xf32>) {opID = 0 : i32, resultID = 0 : i32}
// ALL-OPS-NEXT:    %[[BCAST:.*]] = stablehlo.broadcast_in_dim
// ALL-OPS-NEXT:    probe.observe(%[[BCAST]] : tensor<2x2xf32>) {opID = 1 : i32, resultID = 0 : i32}
// ALL-OPS-NEXT:    %[[TRANS:.*]] = stablehlo.transpose
// ALL-OPS-NEXT:    probe.observe(%[[TRANS]] : tensor<2x2xf32>) {opID = 2 : i32, resultID = 0 : i32}
// ALL-OPS:         probe.report() {reportArgs = ["vishap_probe_report.csv"]}
// ALL-OPS-NEXT:    return

func.func @only_arg_observed(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  return %arg0 : tensor<4xf32>
}
// CALIB-LABEL: func.func @only_arg_observed(
// CALIB-SAME:    %[[ONLY_ARG:.*]]: tensor<4xf32>)
// CALIB-NEXT:    probe.observe(%[[ONLY_ARG]] : tensor<4xf32>) {opID = -1 : i32, resultID = 0 : i32}
// CALIB-NEXT:    probe.report() {reportArgs = ["vishap_probe_report.csv"]}
// CALIB-NEXT:    return
