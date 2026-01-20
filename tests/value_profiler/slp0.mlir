func.func @sensor_classifier(%input: tensor<1x1x2xf32>) -> tensor<1x1x1xf32> {
  // Weights: [1 x 2 x 1]
  %weights = "tosa.const"() <{
    values = dense<[[[0.5], [0.8]]]> : tensor<1x2x1xf32>
  }> : () -> tensor<1x2x1xf32>

  // Bias: [1 x 1 x 1]
  %bias = "tosa.const"() <{
    values = dense<[[[-1.0]]]> : tensor<1x1x1xf32>
  }> : () -> tensor<1x1x1xf32>

  // Zero-points: MUST be rank-1 tensor with one element
  %zp = "tosa.const"() <{
    values = dense<[0.0]> : tensor<1xf32>
  }> : () -> tensor<1xf32>

  // MatMul: (1x1x2) * (1x2x1) = (1x1x1)
  %0 = "tosa.matmul"(%input, %weights, %zp, %zp)
       : (tensor<1x1x2xf32>, tensor<1x2x1xf32>, tensor<1xf32>, tensor<1xf32>)
         -> tensor<1x1x1xf32>

  // Add bias
  %1 = "tosa.add"(%0, %bias)
       : (tensor<1x1x1xf32>, tensor<1x1x1xf32>)
         -> tensor<1x1x1xf32>

  // ReLU
  %2 = "tosa.clamp"(%1) <{
    min_val = 0.0 : f32,
    max_val = 3.402823466e+38 : f32,
    nan_mode = #tosa.nan_mode<PROPAGATE>
  }> : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>

  return %2 : tensor<1x1x1xf32>
}

