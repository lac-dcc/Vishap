module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<?x224x224x3xf32> {ml_program.identifier = "serving_default_keras_layer_input:0", tf_saved_model.index_path = ["keras_layer_input"]}) -> (tensor<?x1001xf32> {ml_program.identifier = "StatefulPartitionedCall:0", tf_saved_model.index_path = ["keras_layer"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %1 = "tosa.const"() <{values = dense<0.0204081628> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
    %2 = "tosa.const"() <{values = dense<[[[[1.236800e+02, 1.167800e+02, 1.039400e+02]]]]> : tensor<1x1x1x3xf32>}> : () -> tensor<1x1x1x3xf32>
    %3 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<1001x1x1x2048xf32>}> : () -> tensor<1001x1x1x2048xf32>
    %4 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<1001xf32>}> : () -> tensor<1001xf32>
    %5 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<2048xf32>}> : () -> tensor<2048xf32>
    %6 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<2048x1x1x512xf32>}> : () -> tensor<2048x1x1x512xf32>
    %7 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<512xf32>}> : () -> tensor<512xf32>
    %8 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<512x3x3x512xf32>}> : () -> tensor<512x3x3x512xf32>
    %9 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<512xf32>}> : () -> tensor<512xf32>
    %10 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<512x1x1x2048xf32>}> : () -> tensor<512x1x1x2048xf32>
    %11 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<2048xf32>}> : () -> tensor<2048xf32>
    %12 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<2048x1x1x512xf32>}> : () -> tensor<2048x1x1x512xf32>
    %13 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<512xf32>}> : () -> tensor<512xf32>
    %14 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<512x3x3x512xf32>}> : () -> tensor<512x3x3x512xf32>
    %15 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<512xf32>}> : () -> tensor<512xf32>
    %16 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<512x1x1x2048xf32>}> : () -> tensor<512x1x1x2048xf32>
    %17 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<2048xf32>}> : () -> tensor<2048xf32>
    %18 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<2048x1x1x512xf32>}> : () -> tensor<2048x1x1x512xf32>
    %19 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<512xf32>}> : () -> tensor<512xf32>
    %20 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<512x3x3x512xf32>}> : () -> tensor<512x3x3x512xf32>
    %21 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<512xf32>}> : () -> tensor<512xf32>
    %22 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<512x1x1x1024xf32>}> : () -> tensor<512x1x1x1024xf32>
    %23 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<2048xf32>}> : () -> tensor<2048xf32>
    %24 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<2048x1x1x1024xf32>}> : () -> tensor<2048x1x1x1024xf32>
    %25 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<1024xf32>}> : () -> tensor<1024xf32>
    %26 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<1024x1x1x256xf32>}> : () -> tensor<1024x1x1x256xf32>
    %27 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256xf32>}> : () -> tensor<256xf32>
    %28 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256x3x3x256xf32>}> : () -> tensor<256x3x3x256xf32>
    %29 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256xf32>}> : () -> tensor<256xf32>
    %30 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256x1x1x1024xf32>}> : () -> tensor<256x1x1x1024xf32>
    %31 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<1024xf32>}> : () -> tensor<1024xf32>
    %32 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<1024x1x1x256xf32>}> : () -> tensor<1024x1x1x256xf32>
    %33 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256xf32>}> : () -> tensor<256xf32>
    %34 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256x3x3x256xf32>}> : () -> tensor<256x3x3x256xf32>
    %35 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256xf32>}> : () -> tensor<256xf32>
    %36 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256x1x1x1024xf32>}> : () -> tensor<256x1x1x1024xf32>
    %37 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<1024xf32>}> : () -> tensor<1024xf32>
    %38 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<1024x1x1x256xf32>}> : () -> tensor<1024x1x1x256xf32>
    %39 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256xf32>}> : () -> tensor<256xf32>
    %40 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256x3x3x256xf32>}> : () -> tensor<256x3x3x256xf32>
    %41 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256xf32>}> : () -> tensor<256xf32>
    %42 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256x1x1x1024xf32>}> : () -> tensor<256x1x1x1024xf32>
    %43 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<1024xf32>}> : () -> tensor<1024xf32>
    %44 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<1024x1x1x256xf32>}> : () -> tensor<1024x1x1x256xf32>
    %45 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256xf32>}> : () -> tensor<256xf32>
    %46 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256x3x3x256xf32>}> : () -> tensor<256x3x3x256xf32>
    %47 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256xf32>}> : () -> tensor<256xf32>
    %48 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256x1x1x1024xf32>}> : () -> tensor<256x1x1x1024xf32>
    %49 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<1024xf32>}> : () -> tensor<1024xf32>
    %50 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<1024x1x1x256xf32>}> : () -> tensor<1024x1x1x256xf32>
    %51 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256xf32>}> : () -> tensor<256xf32>
    %52 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256x3x3x256xf32>}> : () -> tensor<256x3x3x256xf32>
    %53 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256xf32>}> : () -> tensor<256xf32>
    %54 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256x1x1x1024xf32>}> : () -> tensor<256x1x1x1024xf32>
    %55 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<1024xf32>}> : () -> tensor<1024xf32>
    %56 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<1024x1x1x256xf32>}> : () -> tensor<1024x1x1x256xf32>
    %57 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256xf32>}> : () -> tensor<256xf32>
    %58 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256x3x3x256xf32>}> : () -> tensor<256x3x3x256xf32>
    %59 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256xf32>}> : () -> tensor<256xf32>
    %60 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256x1x1x512xf32>}> : () -> tensor<256x1x1x512xf32>
    %61 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<1024xf32>}> : () -> tensor<1024xf32>
    %62 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<1024x1x1x512xf32>}> : () -> tensor<1024x1x1x512xf32>
    %63 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<512xf32>}> : () -> tensor<512xf32>
    %64 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<512x1x1x128xf32>}> : () -> tensor<512x1x1x128xf32>
    %65 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<128xf32>}> : () -> tensor<128xf32>
    %66 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<128x3x3x128xf32>}> : () -> tensor<128x3x3x128xf32>
    %67 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<128xf32>}> : () -> tensor<128xf32>
    %68 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<128x1x1x512xf32>}> : () -> tensor<128x1x1x512xf32>
    %69 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<512xf32>}> : () -> tensor<512xf32>
    %70 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<512x1x1x128xf32>}> : () -> tensor<512x1x1x128xf32>
    %71 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<128xf32>}> : () -> tensor<128xf32>
    %72 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<128x3x3x128xf32>}> : () -> tensor<128x3x3x128xf32>
    %73 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<128xf32>}> : () -> tensor<128xf32>
    %74 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<128x1x1x512xf32>}> : () -> tensor<128x1x1x512xf32>
    %75 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<512xf32>}> : () -> tensor<512xf32>
    %76 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<512x1x1x128xf32>}> : () -> tensor<512x1x1x128xf32>
    %77 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<128xf32>}> : () -> tensor<128xf32>
    %78 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<128x3x3x128xf32>}> : () -> tensor<128x3x3x128xf32>
    %79 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<128xf32>}> : () -> tensor<128xf32>
    %80 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<128x1x1x512xf32>}> : () -> tensor<128x1x1x512xf32>
    %81 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<512xf32>}> : () -> tensor<512xf32>
    %82 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<512x1x1x128xf32>}> : () -> tensor<512x1x1x128xf32>
    %83 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<128xf32>}> : () -> tensor<128xf32>
    %84 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<128x3x3x128xf32>}> : () -> tensor<128x3x3x128xf32>
    %85 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<128xf32>}> : () -> tensor<128xf32>
    %86 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<128x1x1x256xf32>}> : () -> tensor<128x1x1x256xf32>
    %87 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<512xf32>}> : () -> tensor<512xf32>
    %88 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<512x1x1x256xf32>}> : () -> tensor<512x1x1x256xf32>
    %89 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256xf32>}> : () -> tensor<256xf32>
    %90 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256x1x1x64xf32>}> : () -> tensor<256x1x1x64xf32>
    %91 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<64xf32>}> : () -> tensor<64xf32>
    %92 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<64x3x3x64xf32>}> : () -> tensor<64x3x3x64xf32>
    %93 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<64xf32>}> : () -> tensor<64xf32>
    %94 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<64x1x1x256xf32>}> : () -> tensor<64x1x1x256xf32>
    %95 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256xf32>}> : () -> tensor<256xf32>
    %96 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256x1x1x64xf32>}> : () -> tensor<256x1x1x64xf32>
    %97 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<64xf32>}> : () -> tensor<64xf32>
    %98 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<64x3x3x64xf32>}> : () -> tensor<64x3x3x64xf32>
    %99 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<64xf32>}> : () -> tensor<64xf32>
    %100 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<64x1x1x256xf32>}> : () -> tensor<64x1x1x256xf32>
    %101 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256xf32>}> : () -> tensor<256xf32>
    %102 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256x1x1x64xf32>}> : () -> tensor<256x1x1x64xf32>
    %103 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<64xf32>}> : () -> tensor<64xf32>
    %104 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<64x3x3x64xf32>}> : () -> tensor<64x3x3x64xf32>
    %105 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<64xf32>}> : () -> tensor<64xf32>
    %106 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<64x1x1x64xf32>}> : () -> tensor<64x1x1x64xf32>
    %107 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256xf32>}> : () -> tensor<256xf32>
    %108 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<256x1x1x64xf32>}> : () -> tensor<256x1x1x64xf32>
    %109 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<64xf32>}> : () -> tensor<64xf32>
    %110 = "tosa.const"() <{values = dense_resource<__elided__> : tensor<64x7x7x3xf32>}> : () -> tensor<64x7x7x3xf32>
    %111 = "tosa.const"() <{values = dense<[[0, 0], [3, 3], [3, 3], [0, 0]]> : tensor<4x2xi32>}> : () -> tensor<4x2xi32>
    %112 = "tosa.const"() <{values = dense<2.550000e+02> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
    %113 = tosa.mul %arg0, %112 {shift = 0 : i8} : (tensor<?x224x224x3xf32>, tensor<1x1x1x1xf32>) -> tensor<?x224x224x3xf32>
    %114 = tosa.sub %113, %2 : (tensor<?x224x224x3xf32>, tensor<1x1x1x3xf32>) -> tensor<?x224x224x3xf32>
    %115 = tosa.pad %114, %111, %0 : (tensor<?x224x224x3xf32>, tensor<4x2xi32>, tensor<f32>) -> tensor<?x230x230x3xf32>
    %116 = tosa.conv2d %115, %110, %109 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<?x230x230x3xf32>, tensor<64x7x7x3xf32>, tensor<64xf32>) -> tensor<?x112x112x64xf32>
    %117 = tosa.clamp %116 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x112x112x64xf32>) -> tensor<?x112x112x64xf32>
    %118 = tosa.max_pool2d %117 {kernel = array<i64: 3, 3>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 2, 2>} : (tensor<?x112x112x64xf32>) -> tensor<?x56x56x64xf32>
    %119 = tosa.conv2d %118, %108, %107 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x56x56x64xf32>, tensor<256x1x1x64xf32>, tensor<256xf32>) -> tensor<?x56x56x256xf32>
    %120 = tosa.conv2d %118, %106, %105 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x56x56x64xf32>, tensor<64x1x1x64xf32>, tensor<64xf32>) -> tensor<?x56x56x64xf32>
    %121 = tosa.clamp %120 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x56x56x64xf32>) -> tensor<?x56x56x64xf32>
    %122 = tosa.conv2d %121, %104, %103 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<?x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<?x56x56x64xf32>
    %123 = tosa.clamp %122 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x56x56x64xf32>) -> tensor<?x56x56x64xf32>
    %124 = tosa.conv2d %123, %102, %101 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x56x56x64xf32>, tensor<256x1x1x64xf32>, tensor<256xf32>) -> tensor<?x56x56x256xf32>
    %125 = tosa.add %124, %119 : (tensor<?x56x56x256xf32>, tensor<?x56x56x256xf32>) -> tensor<?x56x56x256xf32>
    %126 = tosa.clamp %125 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x56x56x256xf32>) -> tensor<?x56x56x256xf32>
    %127 = tosa.conv2d %126, %100, %99 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x56x56x256xf32>, tensor<64x1x1x256xf32>, tensor<64xf32>) -> tensor<?x56x56x64xf32>
    %128 = tosa.clamp %127 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x56x56x64xf32>) -> tensor<?x56x56x64xf32>
    %129 = tosa.conv2d %128, %98, %97 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<?x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<?x56x56x64xf32>
    %130 = tosa.clamp %129 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x56x56x64xf32>) -> tensor<?x56x56x64xf32>
    %131 = tosa.conv2d %130, %96, %95 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x56x56x64xf32>, tensor<256x1x1x64xf32>, tensor<256xf32>) -> tensor<?x56x56x256xf32>
    %132 = tosa.add %131, %126 : (tensor<?x56x56x256xf32>, tensor<?x56x56x256xf32>) -> tensor<?x56x56x256xf32>
    %133 = tosa.clamp %132 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x56x56x256xf32>) -> tensor<?x56x56x256xf32>
    %134 = tosa.conv2d %133, %94, %93 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x56x56x256xf32>, tensor<64x1x1x256xf32>, tensor<64xf32>) -> tensor<?x56x56x64xf32>
    %135 = tosa.clamp %134 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x56x56x64xf32>) -> tensor<?x56x56x64xf32>
    %136 = tosa.conv2d %135, %92, %91 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<?x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<?x56x56x64xf32>
    %137 = tosa.clamp %136 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x56x56x64xf32>) -> tensor<?x56x56x64xf32>
    %138 = tosa.conv2d %137, %90, %89 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x56x56x64xf32>, tensor<256x1x1x64xf32>, tensor<256xf32>) -> tensor<?x56x56x256xf32>
    %139 = tosa.add %138, %133 : (tensor<?x56x56x256xf32>, tensor<?x56x56x256xf32>) -> tensor<?x56x56x256xf32>
    %140 = tosa.clamp %139 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x56x56x256xf32>) -> tensor<?x56x56x256xf32>
    %141 = tosa.conv2d %140, %88, %87 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<?x56x56x256xf32>, tensor<512x1x1x256xf32>, tensor<512xf32>) -> tensor<?x28x28x512xf32>
    %142 = tosa.conv2d %140, %86, %85 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x56x56x256xf32>, tensor<128x1x1x256xf32>, tensor<128xf32>) -> tensor<?x56x56x128xf32>
    %143 = tosa.clamp %142 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x56x56x128xf32>) -> tensor<?x56x56x128xf32>
    %144 = tosa.conv2d %143, %84, %83 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 2, 2>} : (tensor<?x56x56x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<?x28x28x128xf32>
    %145 = tosa.clamp %144 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x28x28x128xf32>) -> tensor<?x28x28x128xf32>
    %146 = tosa.conv2d %145, %82, %81 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x28x28x128xf32>, tensor<512x1x1x128xf32>, tensor<512xf32>) -> tensor<?x28x28x512xf32>
    %147 = tosa.add %146, %141 : (tensor<?x28x28x512xf32>, tensor<?x28x28x512xf32>) -> tensor<?x28x28x512xf32>
    %148 = tosa.clamp %147 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x28x28x512xf32>) -> tensor<?x28x28x512xf32>
    %149 = tosa.conv2d %148, %80, %79 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x28x28x512xf32>, tensor<128x1x1x512xf32>, tensor<128xf32>) -> tensor<?x28x28x128xf32>
    %150 = tosa.clamp %149 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x28x28x128xf32>) -> tensor<?x28x28x128xf32>
    %151 = tosa.conv2d %150, %78, %77 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<?x28x28x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<?x28x28x128xf32>
    %152 = tosa.clamp %151 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x28x28x128xf32>) -> tensor<?x28x28x128xf32>
    %153 = tosa.conv2d %152, %76, %75 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x28x28x128xf32>, tensor<512x1x1x128xf32>, tensor<512xf32>) -> tensor<?x28x28x512xf32>
    %154 = tosa.add %153, %148 : (tensor<?x28x28x512xf32>, tensor<?x28x28x512xf32>) -> tensor<?x28x28x512xf32>
    %155 = tosa.clamp %154 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x28x28x512xf32>) -> tensor<?x28x28x512xf32>
    %156 = tosa.conv2d %155, %74, %73 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x28x28x512xf32>, tensor<128x1x1x512xf32>, tensor<128xf32>) -> tensor<?x28x28x128xf32>
    %157 = tosa.clamp %156 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x28x28x128xf32>) -> tensor<?x28x28x128xf32>
    %158 = tosa.conv2d %157, %72, %71 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<?x28x28x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<?x28x28x128xf32>
    %159 = tosa.clamp %158 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x28x28x128xf32>) -> tensor<?x28x28x128xf32>
    %160 = tosa.conv2d %159, %70, %69 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x28x28x128xf32>, tensor<512x1x1x128xf32>, tensor<512xf32>) -> tensor<?x28x28x512xf32>
    %161 = tosa.add %160, %155 : (tensor<?x28x28x512xf32>, tensor<?x28x28x512xf32>) -> tensor<?x28x28x512xf32>
    %162 = tosa.clamp %161 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x28x28x512xf32>) -> tensor<?x28x28x512xf32>
    %163 = tosa.conv2d %162, %68, %67 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x28x28x512xf32>, tensor<128x1x1x512xf32>, tensor<128xf32>) -> tensor<?x28x28x128xf32>
    %164 = tosa.clamp %163 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x28x28x128xf32>) -> tensor<?x28x28x128xf32>
    %165 = tosa.conv2d %164, %66, %65 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<?x28x28x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<?x28x28x128xf32>
    %166 = tosa.clamp %165 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x28x28x128xf32>) -> tensor<?x28x28x128xf32>
    %167 = tosa.conv2d %166, %64, %63 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x28x28x128xf32>, tensor<512x1x1x128xf32>, tensor<512xf32>) -> tensor<?x28x28x512xf32>
    %168 = tosa.add %167, %162 : (tensor<?x28x28x512xf32>, tensor<?x28x28x512xf32>) -> tensor<?x28x28x512xf32>
    %169 = tosa.clamp %168 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x28x28x512xf32>) -> tensor<?x28x28x512xf32>
    %170 = tosa.conv2d %169, %62, %61 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<?x28x28x512xf32>, tensor<1024x1x1x512xf32>, tensor<1024xf32>) -> tensor<?x14x14x1024xf32>
    %171 = tosa.conv2d %169, %60, %59 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x28x28x512xf32>, tensor<256x1x1x512xf32>, tensor<256xf32>) -> tensor<?x28x28x256xf32>
    %172 = tosa.clamp %171 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x28x28x256xf32>) -> tensor<?x28x28x256xf32>
    %173 = tosa.conv2d %172, %58, %57 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 2, 2>} : (tensor<?x28x28x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<?x14x14x256xf32>
    %174 = tosa.clamp %173 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x14x14x256xf32>) -> tensor<?x14x14x256xf32>
    %175 = tosa.conv2d %174, %56, %55 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x14x14x256xf32>, tensor<1024x1x1x256xf32>, tensor<1024xf32>) -> tensor<?x14x14x1024xf32>
    %176 = tosa.add %175, %170 : (tensor<?x14x14x1024xf32>, tensor<?x14x14x1024xf32>) -> tensor<?x14x14x1024xf32>
    %177 = tosa.clamp %176 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x14x14x1024xf32>) -> tensor<?x14x14x1024xf32>
    %178 = tosa.conv2d %177, %54, %53 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x14x14x1024xf32>, tensor<256x1x1x1024xf32>, tensor<256xf32>) -> tensor<?x14x14x256xf32>
    %179 = tosa.clamp %178 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x14x14x256xf32>) -> tensor<?x14x14x256xf32>
    %180 = tosa.conv2d %179, %52, %51 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<?x14x14x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<?x14x14x256xf32>
    %181 = tosa.clamp %180 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x14x14x256xf32>) -> tensor<?x14x14x256xf32>
    %182 = tosa.conv2d %181, %50, %49 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x14x14x256xf32>, tensor<1024x1x1x256xf32>, tensor<1024xf32>) -> tensor<?x14x14x1024xf32>
    %183 = tosa.add %182, %177 : (tensor<?x14x14x1024xf32>, tensor<?x14x14x1024xf32>) -> tensor<?x14x14x1024xf32>
    %184 = tosa.clamp %183 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x14x14x1024xf32>) -> tensor<?x14x14x1024xf32>
    %185 = tosa.conv2d %184, %48, %47 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x14x14x1024xf32>, tensor<256x1x1x1024xf32>, tensor<256xf32>) -> tensor<?x14x14x256xf32>
    %186 = tosa.clamp %185 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x14x14x256xf32>) -> tensor<?x14x14x256xf32>
    %187 = tosa.conv2d %186, %46, %45 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<?x14x14x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<?x14x14x256xf32>
    %188 = tosa.clamp %187 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x14x14x256xf32>) -> tensor<?x14x14x256xf32>
    %189 = tosa.conv2d %188, %44, %43 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x14x14x256xf32>, tensor<1024x1x1x256xf32>, tensor<1024xf32>) -> tensor<?x14x14x1024xf32>
    %190 = tosa.add %189, %184 : (tensor<?x14x14x1024xf32>, tensor<?x14x14x1024xf32>) -> tensor<?x14x14x1024xf32>
    %191 = tosa.clamp %190 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x14x14x1024xf32>) -> tensor<?x14x14x1024xf32>
    %192 = tosa.conv2d %191, %42, %41 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x14x14x1024xf32>, tensor<256x1x1x1024xf32>, tensor<256xf32>) -> tensor<?x14x14x256xf32>
    %193 = tosa.clamp %192 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x14x14x256xf32>) -> tensor<?x14x14x256xf32>
    %194 = tosa.conv2d %193, %40, %39 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<?x14x14x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<?x14x14x256xf32>
    %195 = tosa.clamp %194 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x14x14x256xf32>) -> tensor<?x14x14x256xf32>
    %196 = tosa.conv2d %195, %38, %37 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x14x14x256xf32>, tensor<1024x1x1x256xf32>, tensor<1024xf32>) -> tensor<?x14x14x1024xf32>
    %197 = tosa.add %196, %191 : (tensor<?x14x14x1024xf32>, tensor<?x14x14x1024xf32>) -> tensor<?x14x14x1024xf32>
    %198 = tosa.clamp %197 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x14x14x1024xf32>) -> tensor<?x14x14x1024xf32>
    %199 = tosa.conv2d %198, %36, %35 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x14x14x1024xf32>, tensor<256x1x1x1024xf32>, tensor<256xf32>) -> tensor<?x14x14x256xf32>
    %200 = tosa.clamp %199 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x14x14x256xf32>) -> tensor<?x14x14x256xf32>
    %201 = tosa.conv2d %200, %34, %33 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<?x14x14x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<?x14x14x256xf32>
    %202 = tosa.clamp %201 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x14x14x256xf32>) -> tensor<?x14x14x256xf32>
    %203 = tosa.conv2d %202, %32, %31 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x14x14x256xf32>, tensor<1024x1x1x256xf32>, tensor<1024xf32>) -> tensor<?x14x14x1024xf32>
    %204 = tosa.add %203, %198 : (tensor<?x14x14x1024xf32>, tensor<?x14x14x1024xf32>) -> tensor<?x14x14x1024xf32>
    %205 = tosa.clamp %204 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x14x14x1024xf32>) -> tensor<?x14x14x1024xf32>
    %206 = tosa.conv2d %205, %30, %29 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x14x14x1024xf32>, tensor<256x1x1x1024xf32>, tensor<256xf32>) -> tensor<?x14x14x256xf32>
    %207 = tosa.clamp %206 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x14x14x256xf32>) -> tensor<?x14x14x256xf32>
    %208 = tosa.conv2d %207, %28, %27 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<?x14x14x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<?x14x14x256xf32>
    %209 = tosa.clamp %208 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x14x14x256xf32>) -> tensor<?x14x14x256xf32>
    %210 = tosa.conv2d %209, %26, %25 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x14x14x256xf32>, tensor<1024x1x1x256xf32>, tensor<1024xf32>) -> tensor<?x14x14x1024xf32>
    %211 = tosa.add %210, %205 : (tensor<?x14x14x1024xf32>, tensor<?x14x14x1024xf32>) -> tensor<?x14x14x1024xf32>
    %212 = tosa.clamp %211 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x14x14x1024xf32>) -> tensor<?x14x14x1024xf32>
    %213 = tosa.conv2d %212, %24, %23 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<?x14x14x1024xf32>, tensor<2048x1x1x1024xf32>, tensor<2048xf32>) -> tensor<?x7x7x2048xf32>
    %214 = tosa.conv2d %212, %22, %21 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x14x14x1024xf32>, tensor<512x1x1x1024xf32>, tensor<512xf32>) -> tensor<?x14x14x512xf32>
    %215 = tosa.clamp %214 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x14x14x512xf32>) -> tensor<?x14x14x512xf32>
    %216 = tosa.conv2d %215, %20, %19 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 2, 2>} : (tensor<?x14x14x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<?x7x7x512xf32>
    %217 = tosa.clamp %216 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x7x7x512xf32>) -> tensor<?x7x7x512xf32>
    %218 = tosa.conv2d %217, %18, %17 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x7x7x512xf32>, tensor<2048x1x1x512xf32>, tensor<2048xf32>) -> tensor<?x7x7x2048xf32>
    %219 = tosa.add %218, %213 : (tensor<?x7x7x2048xf32>, tensor<?x7x7x2048xf32>) -> tensor<?x7x7x2048xf32>
    %220 = tosa.clamp %219 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x7x7x2048xf32>) -> tensor<?x7x7x2048xf32>
    %221 = tosa.conv2d %220, %16, %15 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x7x7x2048xf32>, tensor<512x1x1x2048xf32>, tensor<512xf32>) -> tensor<?x7x7x512xf32>
    %222 = tosa.clamp %221 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x7x7x512xf32>) -> tensor<?x7x7x512xf32>
    %223 = tosa.conv2d %222, %14, %13 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<?x7x7x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<?x7x7x512xf32>
    %224 = tosa.clamp %223 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x7x7x512xf32>) -> tensor<?x7x7x512xf32>
    %225 = tosa.conv2d %224, %12, %11 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x7x7x512xf32>, tensor<2048x1x1x512xf32>, tensor<2048xf32>) -> tensor<?x7x7x2048xf32>
    %226 = tosa.add %225, %220 : (tensor<?x7x7x2048xf32>, tensor<?x7x7x2048xf32>) -> tensor<?x7x7x2048xf32>
    %227 = tosa.clamp %226 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x7x7x2048xf32>) -> tensor<?x7x7x2048xf32>
    %228 = tosa.conv2d %227, %10, %9 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x7x7x2048xf32>, tensor<512x1x1x2048xf32>, tensor<512xf32>) -> tensor<?x7x7x512xf32>
    %229 = tosa.clamp %228 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x7x7x512xf32>) -> tensor<?x7x7x512xf32>
    %230 = tosa.conv2d %229, %8, %7 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<?x7x7x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<?x7x7x512xf32>
    %231 = tosa.clamp %230 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x7x7x512xf32>) -> tensor<?x7x7x512xf32>
    %232 = tosa.conv2d %231, %6, %5 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x7x7x512xf32>, tensor<2048x1x1x512xf32>, tensor<2048xf32>) -> tensor<?x7x7x2048xf32>
    %233 = tosa.add %232, %227 : (tensor<?x7x7x2048xf32>, tensor<?x7x7x2048xf32>) -> tensor<?x7x7x2048xf32>
    %234 = tosa.clamp %233 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x7x7x2048xf32>) -> tensor<?x7x7x2048xf32>
    %235 = tosa.reduce_sum %234 {axis = 1 : i32} : (tensor<?x7x7x2048xf32>) -> tensor<?x1x7x2048xf32>
    %236 = tosa.reduce_sum %235 {axis = 2 : i32} : (tensor<?x1x7x2048xf32>) -> tensor<?x1x1x2048xf32>
    %237 = tosa.reshape %236 {new_shape = array<i64: -1, 2048>} : (tensor<?x1x1x2048xf32>) -> tensor<?x2048xf32>
    %238 = tosa.mul %237, %1 {shift = 0 : i8} : (tensor<?x2048xf32>, tensor<1x1xf32>) -> tensor<?x2048xf32>
    %239 = tosa.reshape %238 {new_shape = array<i64: -9223372036854775808, 1, 1, 2048>} : (tensor<?x2048xf32>) -> tensor<?x1x1x2048xf32>
    %240 = tosa.conv2d %239, %3, %4 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x1x1x2048xf32>, tensor<1001x1x1x2048xf32>, tensor<1001xf32>) -> tensor<?x1x1x1001xf32>
    %241 = tosa.reshape %240 {new_shape = array<i64: -9223372036854775808, 1001>} : (tensor<?x1x1x1001xf32>) -> tensor<?x1001xf32>
    %242 = tosa.reduce_max %241 {axis = 1 : i32} : (tensor<?x1001xf32>) -> tensor<?x1xf32>
    %243 = tosa.sub %241, %242 : (tensor<?x1001xf32>, tensor<?x1xf32>) -> tensor<?x1001xf32>
    %244 = tosa.exp %243 : (tensor<?x1001xf32>) -> tensor<?x1001xf32>
    %245 = tosa.reduce_sum %244 {axis = 1 : i32} : (tensor<?x1001xf32>) -> tensor<?x1xf32>
    %246 = tosa.reciprocal %245 : (tensor<?x1xf32>) -> tensor<?x1xf32>
    %247 = tosa.mul %244, %246 {shift = 0 : i8} : (tensor<?x1001xf32>, tensor<?x1xf32>) -> tensor<?x1001xf32>
    return %247 : tensor<?x1001xf32>
  }
}

