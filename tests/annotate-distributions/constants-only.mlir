// RUN: vishap-opt --annotate-distributions="constants-only=true per-axis=false" %s | FileCheck %s --check-prefix=CONST-ONLY
// RUN: vishap-opt --annotate-distributions="input-distributions=1.,5.,3.,2.;unit" %s | FileCheck %s --check-prefix=FULL

func.func @constants_and_dynamic(%arg0: tensor<3x3xf32>, %indices: tensor<1xi64>) -> (tensor<1x2xf32>, tensor<1xi64>) {
  %splat = stablehlo.constant dense<2.5> : tensor<1x2xf32>
  %cst_i32 = stablehlo.constant dense<[10, 20]> : tensor<2xi32>
  %cst_f32 = stablehlo.convert %cst_i32 : (tensor<2xi32>) -> tensor<2xf32>
  %cst_bcast = stablehlo.broadcast_in_dim %cst_f32, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
  %gathered = "stablehlo.gather"(%arg0, %indices) <{
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1
    >,
    slice_sizes = array<i64: 1, 2>,
    indices_are_sorted = false
  }> : (tensor<3x3xf32>, tensor<1xi64>) -> tensor<1x2xf32>
  %preloaded = stablehlo.add %gathered, %splat {
    vishap.calibrated,
    vishap.distribution = [[3.5, 7.5, 5.5, 2.0]]
  } : tensor<1x2xf32>
  %out = stablehlo.add %preloaded, %cst_bcast : tensor<1x2xf32>
  return %out, %indices : tensor<1x2xf32>, tensor<1xi64>
}
// CONST-ONLY-LABEL: func.func @constants_and_dynamic
// CONST-ONLY-NOT:     vishap.distribution
// CONST-ONLY-NEXT:    stablehlo.constant {vishap.distribution = {{\[\[}}2.500000e+00, 2.500000e+00, 2.500000e+00, 0.000000e+00{{\]\]}}} dense<2.500000e+00>
// CONST-ONLY-NEXT:    stablehlo.constant {vishap.distribution = {{\[\[}}1.000000e+01, 2.000000e+01, 1.500000e+01, 2.500000e+01{{\]\]}}} dense<[10, 20]>
// CONST-ONLY-NEXT:    stablehlo.convert {{.*}} {vishap.distribution = {{\[\[}}1.000000e+01, 2.000000e+01, 1.500000e+01, 2.500000e+01{{\]\]}}}
// CONST-ONLY-NEXT:    stablehlo.broadcast_in_dim {{.*}} {vishap.distribution = {{\[\[}}1.000000e+01, 2.000000e+01, 1.500000e+01, 2.500000e+01{{\]\]}}}
// CONST-ONLY-NEXT:    "stablehlo.gather"
// CONST-ONLY-NOT:       vishap.distribution
// CONST-ONLY:         stablehlo.add {{.*}} {vishap.calibrated, vishap.distribution = {{\[\[}}3.500000e+00, 7.500000e+00, 5.500000e+00, 2.000000e+00{{\]\]}}}
// CONST-ONLY-NEXT:    %[[DYN_OUT:.*]] = stablehlo.add
// CONST-ONLY-NOT:     vishap.distribution
// CONST-ONLY-NEXT:    return %[[DYN_OUT]]

// FULL-LABEL: func.func @constants_and_dynamic
// FULL-SAME:    vishap.args_distribution = {{\[\[}}1.000000e+00, 5.000000e+00, 3.000000e+00, 2.000000e+00], unit]
// FULL-SAME:    vishap.distribution = {{\[\[}}1.350000e+01, 2.750000e+01, 2.050000e+01, 2.700000e+01], unit]
// FULL:         "stablehlo.gather"
// FULL-SAME:      vishap.distribution = {{\[\[}}1.000000e+00, 5.000000e+00, 3.000000e+00, 2.000000e+00{{\]\]}}
// FULL-NEXT:    stablehlo.add {{.*}} {vishap.calibrated, vishap.distribution = {{\[\[}}3.500000e+00, 7.500000e+00, 5.500000e+00, 2.000000e+00{{\]\]}}}
// FULL-NEXT:    stablehlo.add {{.*}} {vishap.distribution = {{\[\[}}1.350000e+01, 2.750000e+01, 2.050000e+01, 2.700000e+01{{\]\]}}}
// FULL-NEXT:    return

func.func @only_constant_return(%arg0: tensor<2xf32>, %idx: tensor<1xi64>) -> (tensor<2xf32>, tensor<1xi64>) {
  %cst = stablehlo.constant dense<[1.0, 3.0]> : tensor<2xf32>
  return %cst, %idx : tensor<2xf32>, tensor<1xi64>
}
// CONST-ONLY-LABEL: func.func @only_constant_return
// CONST-ONLY-SAME:    vishap.distribution = {{\[\[}}1.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00], unit]
