// RUN: vishap-opt --annotate-distributions %s | FileCheck %s

func.func @exp() -> tensor<2x3xf32> {
  %cst = stablehlo.constant dense<[[0.42, 0.33, -1.], [-0.5, 0.55, 0.02]]> : tensor<2x3xf32>
  %res = stablehlo.exponential %cst : tensor<2x3xf32>
  return %res : tensor<2x3xf32>
}
// CHECK-LABEL: func.func @exp
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.exponential
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [0.36787944117144233, 1.7332530385293814, 1.1305828952124937, 0.45665380614109868]
// CHECK-SAME:      ]
// CHECK-NEXT:    return
