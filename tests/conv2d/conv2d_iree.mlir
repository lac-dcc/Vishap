func.func @main() -> tensor<1x1x2x2xf32> {
  // Input: 1 batch, 1 channel, 4x4 spatial
  %input = arith.constant dense<[
    [[[ 1.0,  2.0,  3.0,  4.0],
      [ 5.0,  6.0,  7.0,  8.0],
      [ 9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0]]]
  ]> : tensor<1x1x4x4xf32>

  // Filter: 1 out-channel, 1 in-channel, 3x3 Sobel kernel
  %filter = arith.constant dense<[
    [[[ 1.0,  0.0, -1.0],
      [ 2.0,  0.0, -2.0],
      [ 1.0,  0.0, -1.0]]]
  ]> : tensor<1x1x3x3xf32>

  // Output: 2x2 (valid convolution: 4-3+1=2)
  %zero = arith.constant dense<0.0> : tensor<1x1x2x2xf32>

  %output = linalg.conv_2d_nchw_fchw
    ins(%input, %filter : tensor<1x1x4x4xf32>, tensor<1x1x3x3xf32>)
    outs(%zero : tensor<1x1x2x2xf32>)
    -> tensor<1x1x2x2xf32>

  return %output : tensor<1x1x2x2xf32>
}
