func.func @sensor_classifier_i8(%input: tensor<1x1x2xi8>) -> tensor<1x1x1xi8> {
  // 1. Weights: [1 x 2 x 1] in i8
  %weights = "tosa.const"() <{
    values = dense<[[[64], [102]]]> : tensor<1x2x1xi8>
  }> : () -> tensor<1x2x1xi8>

  // 2. Bias: i32
  %bias = "tosa.const"() <{
    values = dense<[[[-128]]]> : tensor<1x1x1xi32>
  }> : () -> tensor<1x1x1xi32>

  // 3. Zero-points for MatMul (Both must be i8 to match input/weights)
  %zp_i8 = "tosa.const"() <{
    values = dense<[0]> : tensor<1xi8>
  }> : () -> tensor<1xi8>

  // 4. Numerical parameters for Rescale
  // Operand 1 (in_zp) remains i32 to match the i32 accumulator input
  %in_zp  = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  
  // Operand 2 (out_zp) MUST be i8 to match the i8 output tensor
  %out_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  
  // Operand 3 (multiplier) is i32
  %mult   = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
  
  // Operand 4 (shift) is i8
  %shift  = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>

  // 5. MatMul: (i8, i8, i8, i8) -> i32
  %0 = "tosa.matmul"(%input, %weights, %zp_i8, %zp_i8)
       : (tensor<1x1x2xi8>, tensor<1x2x1xi8>, tensor<1xi8>, tensor<1xi8>)
         -> tensor<1x1x1xi32>

  // 6. Add bias (i32 + i32)
  %1 = "tosa.add"(%0, %bias)
       : (tensor<1x1x1xi32>, tensor<1x1x1xi32>)
         -> tensor<1x1x1xi32>

  // 7. Rescale: (i32 tensor, i32 zp, i8 zp, i32 mult, i8 shift) -> i8 tensor
  %2 = "tosa.rescale"(%1, %mult, %shift, %in_zp, %out_zp) <{
       input_unsigned = false,
       output_unsigned = false,
       per_channel = false,
       rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>,
       scale32 = true
  }> : (
    tensor<1x1x1xi32>,
    tensor<1xi32>,
    tensor<1xi8>,
    tensor<1xi32>,
    tensor<1xi8>
  ) -> tensor<1x1x1xi8>

  // 8. ReLU (Clamp)
  %3 = "tosa.clamp"(%2) <{
    min_val = 0 : i8,
    max_val = 127 : i8,
    nan_mode = #tosa.nan_mode<PROPAGATE>
  }> : (tensor<1x1x1xi8>) -> tensor<1x1x1xi8>

  return %3 : tensor<1x1x1xi8>
}
