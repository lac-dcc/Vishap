// RUN: vishap-opt --annotate-distributions="per-axis=false" %s | FileCheck %s --check-prefixes=CHECK,TENSOR
// RUN: vishap-opt --annotate-distributions="per-axis=true annotate-axes=true" %s | FileCheck %s --check-prefixes=CHECK,SLICE

func.func @rsqrt() -> tensor<2x2xf32> {
  %cst = stablehlo.constant dense<[[1.0, 4.0], [9.0, 16.0]]> : tensor<2x2xf32>
  %res = stablehlo.rsqrt %cst : tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @rsqrt
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.rsqrt
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          {{\[\[}}5.000000e-01, 1.000000e+00, 0.71783702885822209, 3.600000e-02], [2.500000e-01, 0.33333333333333331, 0.29115828822137285, 1.568000e-03{{\]\]}},
// SLICE-SAME:          {{\[\[}}0.33333333333333331, 1.000000e+00, 0.55454485841994783, 3.200000e-02], [2.500000e-01, 5.000000e-01, 0.35891851442911105, 0.0089999999999999993{{\]\]}}
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// SLICE-SAME:      vishap.distribution = [
// SLICE-SAME:        [2.500000e-01, 1.000000e+00, 0.50449765853979744, 0.064297686927861919]
// SLICE-SAME:      ]
// TENSOR-SAME:     vishap.distribution = [
// TENSOR-SAME:       [2.500000e-01, 1.000000e+00, 0.44365527157918455, 0.01911111111111111]
// TENSOR-SAME:     ]
// CHECK-NEXT:    return
