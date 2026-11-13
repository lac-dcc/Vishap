// RUN: vishap-opt --annotate-distributions="per-axis=false" %s | FileCheck --allow-unused-prefixes %s --check-prefixes=CHECK,TENSOR
// RUN: vishap-opt --annotate-distributions="per-axis=true annotate-axes=true" %s | FileCheck --allow-unused-prefixes %s --check-prefixes=CHECK,SLICE

func.func @exp() -> tensor<2x3xf32> {
  %cst = stablehlo.constant dense<[[0.42, 0.33, -1.], [-0.5, 0.55, 0.02]]> : tensor<2x3xf32>
  %res = stablehlo.exponential %cst : tensor<2x3xf32>
  return %res : tensor<2x3xf32>
}
// CHECK-LABEL: func.func @exp
// CHECK-NEXT:    stablehlo.constant
// CHECK-NEXT:    stablehlo.exponential
// SLICE-SAME:      vishap.axis_distribution = [
// SLICE-SAME:        [
// SLICE-SAME:          {{\[\[}}0.36787944117144233, 1.5219615356611187, 1.1358838958492918, 0.6763775023193519], [0.60653065971263342, 1.7332530385293814, 1.1221103091619533, 0.25399458148716547{{\]\]}},
// SLICE-SAME:          {{\[\[}}0.60653065971263342, 1.5219615356611187, 1.0680130829621086, 0.26879864300012196], [1.3909681467035759, 1.7332530385293814, 1.562129590510384, 0.029706371963170556], [0.36787944117144233, 1.0202013395706904, 0.69771121052395912, 0.14460898100156394{{\]\]}}
// SLICE-SAME:        ]
// SLICE-SAME:      ]
// SLICE-SAME:      vishap.distribution = [
// SLICE-SAME:        [0.36787944117144233, 1.7332530385293814, 1.1092846279988171, 0.27309285814620321]
// SLICE-SAME:      ]
// TENSOR-SAME:     vishap.distribution = [
// TENSOR-SAME:       [0.36787944117144233, 1.7332530385293814, 1.1305828952124937, 0.45665380614109868]
// TENSOR-SAME:     ]
// CHECK-NEXT:    return
