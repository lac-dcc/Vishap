// Large conv2d benchmark:
//   Input:  1 batch, 32 channels, 128x128 spatial  -> tensor<1x32x128x128xf32>
//   Filter: 64 filters, 32 channels, 3x3 kernel    -> tensor<64x32x3x3xf32>
//   Output: 1 batch, 64 channels, 126x126 spatial  -> tensor<1x64x126x126xf32>
//
// Total multiply-accumulate ops: 1 * 64 * 126 * 126 * 32 * 3 * 3 = ~184 million
// Expected output: 2.88 (= 32 channels * 9 kernel elements * 1.0 * 0.01)

func.func @main() {
  // --- Input: filled with 1.0 ---
  %input_init = tensor.empty() : tensor<1x32x128x128xf32>
  %c1f = arith.constant 1.0 : f32
  %input = linalg.fill ins(%c1f : f32) outs(%input_init : tensor<1x32x128x128xf32>)
           -> tensor<1x32x128x128xf32>

  // --- Filter: filled with 0.01 ---
  %filter_init = tensor.empty() : tensor<64x32x3x3xf32>
  %c0_01f = arith.constant 0.01 : f32
  %filter = linalg.fill ins(%c0_01f : f32) outs(%filter_init : tensor<64x32x3x3xf32>)
            -> tensor<64x32x3x3xf32>

  // --- Output: zero-initialized ---
  %output_init = tensor.empty() : tensor<1x64x126x126xf32>
  %zerof = arith.constant 0.0 : f32
  %zero = linalg.fill ins(%zerof : f32) outs(%output_init : tensor<1x64x126x126xf32>)
          -> tensor<1x64x126x126xf32>

  // --- Convolution ---
  %output = linalg.conv_2d_nchw_fchw
    ins(%input, %filter : tensor<1x32x128x128xf32>, tensor<64x32x3x3xf32>)
    outs(%zero          : tensor<1x64x126x126xf32>)
    -> tensor<1x64x126x126xf32>

  // --- Print just the first output value to verify correctness ---
  // Expected: 2.88 (32 channels * 9 kernel positions * 1.0 * 0.01)
  %c0 = arith.constant 0 : index
  %val = tensor.extract %output[%c0, %c0, %c0, %c0] : tensor<1x64x126x126xf32>
  vector.print %val : f32

  return
}
