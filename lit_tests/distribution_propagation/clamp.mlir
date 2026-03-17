// RUN: vishap-opt --annotate-distributions %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @relu() -> tensor<3x2xf32> {
  %cst_0 = arith.constant 0. : f32
  %0 = arith.constant dense<[[1.0, 2.0], [-3.0, -4.0], [5.0, -6.0]]> : tensor<3x2xf32>
  %clamped = tensor.empty() : tensor<3x2xf32>
  %res = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
            ins(%0: tensor<3x2xf32>)
            outs(%clamped: tensor<3x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cmp = arith.cmpf ugt, %in, %cst_0 : f32
      %clamp = arith.select %cmp, %in, %cst_0 : f32
      linalg.yield %clamp : f32
  } -> tensor<3x2xf32>
  return %res : tensor<3x2xf32>
}
// CHECK-LABEL: func.func @relu
// CHECK-NEXT:    arith.constant
// CHECK-NEXT:    arith.constant
// CHECK-NEXT:    tensor.empty
// CHECK-NEXT:    linalg.generic
// CHECK-SAME:      vishap.distribution = [
// CHECK-SAME:        [0.000000e+00, 5.000000e+00, 1.1372720262294531, 3.7403129901231531]
// CHECK-SAME:      ]
