// RUN: vishap-opt --annotate-distributions %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @div() -> tensor<2x3xf32> {
  %two = arith.constant 2.0 : f32
  %cst = arith.constant dense<[[0.42, 2., -1.], [4., 0.55, 2.]]> : tensor<2x3xf32>
  %divided = tensor.empty() : tensor<2x3xf32>
  %res = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
            ins(%cst: tensor<2x3xf32>)
            outs(%divided: tensor<2x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %div = arith.divf %in, %two : f32
      linalg.yield %div : f32
  } -> tensor<2x3xf32>
  return %res : tensor<2x3xf32>
}
// CHECK-LABEL: func.func @div
// CHECK-NEXT:    arith.constant
// CHECK-NEXT:    arith.constant
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-1.000000e+00, 4.000000e+00, 1.3283333331346512, 2.4820138897664017]
// CHECK-SAME:      ]
// CHECK-NEXT:    tensor.empty
// CHECK-NEXT:    linalg.generic
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [-5.000000e-01, 2.000000e+00, 0.66416666656732559, 0.62050347244160042]
// CHECK-SAME:      ]
