func.func @test_profiler() {
  // Tensor E representation: 3 slices (dim 0), each slice is 2x2 (dim 1, 2)
  // Slice 0: [1, 0, 2, 0] 
  // Slice 1: [0, 0, 0, 0] 
  // Slice 2: [3, 0, 4, 0] 
  
  %0 = "tosa.const"() {
    values = dense<[[[1.0, 0.0], [2.0, 0.0]], 
                   [[0.0, 0.0], [0.0, 0.0]], 
                   [[3.0, 0.0], [4.0, 0.0]]]> : tensor<3x2x2xf32>
  } : () -> tensor<3x2x2xf32>
  
  return
}
