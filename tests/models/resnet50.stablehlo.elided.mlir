module {
  func.func @torch_jit(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %cst = stablehlo.constant dense_resource<__elided__> : tensor<1000x2048xf32>
    %cst_0 = stablehlo.constant dense_resource<__elided__> : tensor<1000xf32>
    %cst_1 = stablehlo.constant dense_resource<__elided__> : tensor<64x3x7x7xf32>
    %cst_2 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_3 = stablehlo.constant dense_resource<__elided__> : tensor<64x64x1x1xf32>
    %cst_4 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_5 = stablehlo.constant dense_resource<__elided__> : tensor<64x64x3x3xf32>
    %cst_6 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_7 = stablehlo.constant dense_resource<__elided__> : tensor<256x64x1x1xf32>
    %cst_8 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_9 = stablehlo.constant dense_resource<__elided__> : tensor<256x64x1x1xf32>
    %cst_10 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_11 = stablehlo.constant dense_resource<__elided__> : tensor<64x256x1x1xf32>
    %cst_12 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_13 = stablehlo.constant dense_resource<__elided__> : tensor<64x64x3x3xf32>
    %cst_14 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_15 = stablehlo.constant dense_resource<__elided__> : tensor<256x64x1x1xf32>
    %cst_16 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_17 = stablehlo.constant dense_resource<__elided__> : tensor<64x256x1x1xf32>
    %cst_18 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_19 = stablehlo.constant dense_resource<__elided__> : tensor<64x64x3x3xf32>
    %cst_20 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_21 = stablehlo.constant dense_resource<__elided__> : tensor<256x64x1x1xf32>
    %cst_22 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_23 = stablehlo.constant dense_resource<__elided__> : tensor<128x256x1x1xf32>
    %cst_24 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_25 = stablehlo.constant dense_resource<__elided__> : tensor<128x128x3x3xf32>
    %cst_26 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_27 = stablehlo.constant dense_resource<__elided__> : tensor<512x128x1x1xf32>
    %cst_28 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_29 = stablehlo.constant dense_resource<__elided__> : tensor<512x256x1x1xf32>
    %cst_30 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_31 = stablehlo.constant dense_resource<__elided__> : tensor<128x512x1x1xf32>
    %cst_32 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_33 = stablehlo.constant dense_resource<__elided__> : tensor<128x128x3x3xf32>
    %cst_34 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_35 = stablehlo.constant dense_resource<__elided__> : tensor<512x128x1x1xf32>
    %cst_36 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_37 = stablehlo.constant dense_resource<__elided__> : tensor<128x512x1x1xf32>
    %cst_38 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_39 = stablehlo.constant dense_resource<__elided__> : tensor<128x128x3x3xf32>
    %cst_40 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_41 = stablehlo.constant dense_resource<__elided__> : tensor<512x128x1x1xf32>
    %cst_42 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_43 = stablehlo.constant dense_resource<__elided__> : tensor<128x512x1x1xf32>
    %cst_44 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_45 = stablehlo.constant dense_resource<__elided__> : tensor<128x128x3x3xf32>
    %cst_46 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_47 = stablehlo.constant dense_resource<__elided__> : tensor<512x128x1x1xf32>
    %cst_48 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_49 = stablehlo.constant dense_resource<__elided__> : tensor<256x512x1x1xf32>
    %cst_50 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_51 = stablehlo.constant dense_resource<__elided__> : tensor<256x256x3x3xf32>
    %cst_52 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_53 = stablehlo.constant dense_resource<__elided__> : tensor<1024x256x1x1xf32>
    %cst_54 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_55 = stablehlo.constant dense_resource<__elided__> : tensor<1024x512x1x1xf32>
    %cst_56 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_57 = stablehlo.constant dense_resource<__elided__> : tensor<256x1024x1x1xf32>
    %cst_58 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_59 = stablehlo.constant dense_resource<__elided__> : tensor<256x256x3x3xf32>
    %cst_60 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_61 = stablehlo.constant dense_resource<__elided__> : tensor<1024x256x1x1xf32>
    %cst_62 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_63 = stablehlo.constant dense_resource<__elided__> : tensor<256x1024x1x1xf32>
    %cst_64 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_65 = stablehlo.constant dense_resource<__elided__> : tensor<256x256x3x3xf32>
    %cst_66 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_67 = stablehlo.constant dense_resource<__elided__> : tensor<1024x256x1x1xf32>
    %cst_68 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_69 = stablehlo.constant dense_resource<__elided__> : tensor<256x1024x1x1xf32>
    %cst_70 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_71 = stablehlo.constant dense_resource<__elided__> : tensor<256x256x3x3xf32>
    %cst_72 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_73 = stablehlo.constant dense_resource<__elided__> : tensor<1024x256x1x1xf32>
    %cst_74 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_75 = stablehlo.constant dense_resource<__elided__> : tensor<256x1024x1x1xf32>
    %cst_76 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_77 = stablehlo.constant dense_resource<__elided__> : tensor<256x256x3x3xf32>
    %cst_78 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_79 = stablehlo.constant dense_resource<__elided__> : tensor<1024x256x1x1xf32>
    %cst_80 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_81 = stablehlo.constant dense_resource<__elided__> : tensor<256x1024x1x1xf32>
    %cst_82 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_83 = stablehlo.constant dense_resource<__elided__> : tensor<256x256x3x3xf32>
    %cst_84 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_85 = stablehlo.constant dense_resource<__elided__> : tensor<1024x256x1x1xf32>
    %cst_86 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_87 = stablehlo.constant dense_resource<__elided__> : tensor<512x1024x1x1xf32>
    %cst_88 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_89 = stablehlo.constant dense_resource<__elided__> : tensor<512x512x3x3xf32>
    %cst_90 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_91 = stablehlo.constant dense_resource<__elided__> : tensor<2048x512x1x1xf32>
    %cst_92 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_93 = stablehlo.constant dense_resource<__elided__> : tensor<2048x1024x1x1xf32>
    %cst_94 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_95 = stablehlo.constant dense_resource<__elided__> : tensor<512x2048x1x1xf32>
    %cst_96 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_97 = stablehlo.constant dense_resource<__elided__> : tensor<512x512x3x3xf32>
    %cst_98 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_99 = stablehlo.constant dense_resource<__elided__> : tensor<2048x512x1x1xf32>
    %cst_100 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_101 = stablehlo.constant dense_resource<__elided__> : tensor<512x2048x1x1xf32>
    %cst_102 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_103 = stablehlo.constant dense_resource<__elided__> : tensor<512x512x3x3xf32>
    %cst_104 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_105 = stablehlo.constant dense_resource<__elided__> : tensor<2048x512x1x1xf32>
    %cst_106 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_107 = stablehlo.constant dense<0.000000e+00> : tensor<1x64x112x112xf32>
    %cst_108 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_109 = stablehlo.constant dense<0.000000e+00> : tensor<1x64x56x56xf32>
    %cst_110 = stablehlo.constant dense<0.000000e+00> : tensor<1x256x56x56xf32>
    %cst_111 = stablehlo.constant dense<0.000000e+00> : tensor<1x128x56x56xf32>
    %cst_112 = stablehlo.constant dense<0.000000e+00> : tensor<1x128x28x28xf32>
    %cst_113 = stablehlo.constant dense<0.000000e+00> : tensor<1x512x28x28xf32>
    %cst_114 = stablehlo.constant dense<0.000000e+00> : tensor<1x256x28x28xf32>
    %cst_115 = stablehlo.constant dense<0.000000e+00> : tensor<1x256x14x14xf32>
    %cst_116 = stablehlo.constant dense<0.000000e+00> : tensor<1x1024x14x14xf32>
    %cst_117 = stablehlo.constant dense<0.000000e+00> : tensor<1x512x14x14xf32>
    %cst_118 = stablehlo.constant dense<0.000000e+00> : tensor<1x512x7x7xf32>
    %cst_119 = stablehlo.constant dense<0.000000e+00> : tensor<1x2048x7x7xf32>
    %cst_120 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_121 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.convolution(%arg0, %cst_1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
    %1 = stablehlo.reshape %cst_2 : (tensor<64xf32>) -> tensor<64x1x1xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x64x112x112xf32>
    %4 = stablehlo.maximum %3, %cst_107 : tensor<1x64x112x112xf32>
    %5 = "stablehlo.reduce_window"(%4, %cst_108) <{padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %297 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %297 : tensor<f32>
    }) : (tensor<1x64x112x112xf32>, tensor<f32>) -> tensor<1x64x56x56xf32>
    %6 = stablehlo.convolution(%5, %cst_3) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %7 = stablehlo.reshape %cst_4 : (tensor<64xf32>) -> tensor<64x1x1xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %9 = stablehlo.add %6, %8 : tensor<1x64x56x56xf32>
    %10 = stablehlo.maximum %9, %cst_109 : tensor<1x64x56x56xf32>
    %11 = stablehlo.convolution(%10, %cst_5) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x56x56xf32>
    %12 = stablehlo.reshape %cst_6 : (tensor<64xf32>) -> tensor<64x1x1xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x64x56x56xf32>
    %15 = stablehlo.maximum %14, %cst_109 : tensor<1x64x56x56xf32>
    %16 = stablehlo.convolution(%15, %cst_7) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<1x256x56x56xf32>
    %17 = stablehlo.reshape %cst_8 : (tensor<256xf32>) -> tensor<256x1x1xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %19 = stablehlo.add %16, %18 : tensor<1x256x56x56xf32>
    %20 = stablehlo.convolution(%5, %cst_9) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<1x256x56x56xf32>
    %21 = stablehlo.reshape %cst_10 : (tensor<256xf32>) -> tensor<256x1x1xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %23 = stablehlo.add %20, %22 : tensor<1x256x56x56xf32>
    %24 = stablehlo.add %19, %23 : tensor<1x256x56x56xf32>
    %25 = stablehlo.maximum %24, %cst_110 : tensor<1x256x56x56xf32>
    %26 = stablehlo.convolution(%25, %cst_11) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<1x64x56x56xf32>
    %27 = stablehlo.reshape %cst_12 : (tensor<64xf32>) -> tensor<64x1x1xf32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %29 = stablehlo.add %26, %28 : tensor<1x64x56x56xf32>
    %30 = stablehlo.maximum %29, %cst_109 : tensor<1x64x56x56xf32>
    %31 = stablehlo.convolution(%30, %cst_13) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x56x56xf32>
    %32 = stablehlo.reshape %cst_14 : (tensor<64xf32>) -> tensor<64x1x1xf32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %34 = stablehlo.add %31, %33 : tensor<1x64x56x56xf32>
    %35 = stablehlo.maximum %34, %cst_109 : tensor<1x64x56x56xf32>
    %36 = stablehlo.convolution(%35, %cst_15) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<1x256x56x56xf32>
    %37 = stablehlo.reshape %cst_16 : (tensor<256xf32>) -> tensor<256x1x1xf32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %39 = stablehlo.add %36, %38 : tensor<1x256x56x56xf32>
    %40 = stablehlo.add %39, %25 : tensor<1x256x56x56xf32>
    %41 = stablehlo.maximum %40, %cst_110 : tensor<1x256x56x56xf32>
    %42 = stablehlo.convolution(%41, %cst_17) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<1x64x56x56xf32>
    %43 = stablehlo.reshape %cst_18 : (tensor<64xf32>) -> tensor<64x1x1xf32>
    %44 = stablehlo.broadcast_in_dim %43, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %45 = stablehlo.add %42, %44 : tensor<1x64x56x56xf32>
    %46 = stablehlo.maximum %45, %cst_109 : tensor<1x64x56x56xf32>
    %47 = stablehlo.convolution(%46, %cst_19) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x56x56xf32>
    %48 = stablehlo.reshape %cst_20 : (tensor<64xf32>) -> tensor<64x1x1xf32>
    %49 = stablehlo.broadcast_in_dim %48, dims = [1, 2, 3] : (tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %50 = stablehlo.add %47, %49 : tensor<1x64x56x56xf32>
    %51 = stablehlo.maximum %50, %cst_109 : tensor<1x64x56x56xf32>
    %52 = stablehlo.convolution(%51, %cst_21) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<1x256x56x56xf32>
    %53 = stablehlo.reshape %cst_22 : (tensor<256xf32>) -> tensor<256x1x1xf32>
    %54 = stablehlo.broadcast_in_dim %53, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %55 = stablehlo.add %52, %54 : tensor<1x256x56x56xf32>
    %56 = stablehlo.add %55, %41 : tensor<1x256x56x56xf32>
    %57 = stablehlo.maximum %56, %cst_110 : tensor<1x256x56x56xf32>
    %58 = stablehlo.convolution(%57, %cst_23) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x56x56xf32>, tensor<128x256x1x1xf32>) -> tensor<1x128x56x56xf32>
    %59 = stablehlo.reshape %cst_24 : (tensor<128xf32>) -> tensor<128x1x1xf32>
    %60 = stablehlo.broadcast_in_dim %59, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x56x56xf32>
    %61 = stablehlo.add %58, %60 : tensor<1x128x56x56xf32>
    %62 = stablehlo.maximum %61, %cst_111 : tensor<1x128x56x56xf32>
    %63 = stablehlo.convolution(%62, %cst_25) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x56x56xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x28x28xf32>
    %64 = stablehlo.reshape %cst_26 : (tensor<128xf32>) -> tensor<128x1x1xf32>
    %65 = stablehlo.broadcast_in_dim %64, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %66 = stablehlo.add %63, %65 : tensor<1x128x28x28xf32>
    %67 = stablehlo.maximum %66, %cst_112 : tensor<1x128x28x28xf32>
    %68 = stablehlo.convolution(%67, %cst_27) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<1x512x28x28xf32>
    %69 = stablehlo.reshape %cst_28 : (tensor<512xf32>) -> tensor<512x1x1xf32>
    %70 = stablehlo.broadcast_in_dim %69, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %71 = stablehlo.add %68, %70 : tensor<1x512x28x28xf32>
    %72 = stablehlo.convolution(%57, %cst_29) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x56x56xf32>, tensor<512x256x1x1xf32>) -> tensor<1x512x28x28xf32>
    %73 = stablehlo.reshape %cst_30 : (tensor<512xf32>) -> tensor<512x1x1xf32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %75 = stablehlo.add %72, %74 : tensor<1x512x28x28xf32>
    %76 = stablehlo.add %71, %75 : tensor<1x512x28x28xf32>
    %77 = stablehlo.maximum %76, %cst_113 : tensor<1x512x28x28xf32>
    %78 = stablehlo.convolution(%77, %cst_31) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<1x128x28x28xf32>
    %79 = stablehlo.reshape %cst_32 : (tensor<128xf32>) -> tensor<128x1x1xf32>
    %80 = stablehlo.broadcast_in_dim %79, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %81 = stablehlo.add %78, %80 : tensor<1x128x28x28xf32>
    %82 = stablehlo.maximum %81, %cst_112 : tensor<1x128x28x28xf32>
    %83 = stablehlo.convolution(%82, %cst_33) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x28x28xf32>
    %84 = stablehlo.reshape %cst_34 : (tensor<128xf32>) -> tensor<128x1x1xf32>
    %85 = stablehlo.broadcast_in_dim %84, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %86 = stablehlo.add %83, %85 : tensor<1x128x28x28xf32>
    %87 = stablehlo.maximum %86, %cst_112 : tensor<1x128x28x28xf32>
    %88 = stablehlo.convolution(%87, %cst_35) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<1x512x28x28xf32>
    %89 = stablehlo.reshape %cst_36 : (tensor<512xf32>) -> tensor<512x1x1xf32>
    %90 = stablehlo.broadcast_in_dim %89, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %91 = stablehlo.add %88, %90 : tensor<1x512x28x28xf32>
    %92 = stablehlo.add %91, %77 : tensor<1x512x28x28xf32>
    %93 = stablehlo.maximum %92, %cst_113 : tensor<1x512x28x28xf32>
    %94 = stablehlo.convolution(%93, %cst_37) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<1x128x28x28xf32>
    %95 = stablehlo.reshape %cst_38 : (tensor<128xf32>) -> tensor<128x1x1xf32>
    %96 = stablehlo.broadcast_in_dim %95, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %97 = stablehlo.add %94, %96 : tensor<1x128x28x28xf32>
    %98 = stablehlo.maximum %97, %cst_112 : tensor<1x128x28x28xf32>
    %99 = stablehlo.convolution(%98, %cst_39) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x28x28xf32>
    %100 = stablehlo.reshape %cst_40 : (tensor<128xf32>) -> tensor<128x1x1xf32>
    %101 = stablehlo.broadcast_in_dim %100, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %102 = stablehlo.add %99, %101 : tensor<1x128x28x28xf32>
    %103 = stablehlo.maximum %102, %cst_112 : tensor<1x128x28x28xf32>
    %104 = stablehlo.convolution(%103, %cst_41) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<1x512x28x28xf32>
    %105 = stablehlo.reshape %cst_42 : (tensor<512xf32>) -> tensor<512x1x1xf32>
    %106 = stablehlo.broadcast_in_dim %105, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %107 = stablehlo.add %104, %106 : tensor<1x512x28x28xf32>
    %108 = stablehlo.add %107, %93 : tensor<1x512x28x28xf32>
    %109 = stablehlo.maximum %108, %cst_113 : tensor<1x512x28x28xf32>
    %110 = stablehlo.convolution(%109, %cst_43) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<1x128x28x28xf32>
    %111 = stablehlo.reshape %cst_44 : (tensor<128xf32>) -> tensor<128x1x1xf32>
    %112 = stablehlo.broadcast_in_dim %111, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %113 = stablehlo.add %110, %112 : tensor<1x128x28x28xf32>
    %114 = stablehlo.maximum %113, %cst_112 : tensor<1x128x28x28xf32>
    %115 = stablehlo.convolution(%114, %cst_45) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x28x28xf32>
    %116 = stablehlo.reshape %cst_46 : (tensor<128xf32>) -> tensor<128x1x1xf32>
    %117 = stablehlo.broadcast_in_dim %116, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %118 = stablehlo.add %115, %117 : tensor<1x128x28x28xf32>
    %119 = stablehlo.maximum %118, %cst_112 : tensor<1x128x28x28xf32>
    %120 = stablehlo.convolution(%119, %cst_47) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<1x512x28x28xf32>
    %121 = stablehlo.reshape %cst_48 : (tensor<512xf32>) -> tensor<512x1x1xf32>
    %122 = stablehlo.broadcast_in_dim %121, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %123 = stablehlo.add %120, %122 : tensor<1x512x28x28xf32>
    %124 = stablehlo.add %123, %109 : tensor<1x512x28x28xf32>
    %125 = stablehlo.maximum %124, %cst_113 : tensor<1x512x28x28xf32>
    %126 = stablehlo.convolution(%125, %cst_49) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x28x28xf32>, tensor<256x512x1x1xf32>) -> tensor<1x256x28x28xf32>
    %127 = stablehlo.reshape %cst_50 : (tensor<256xf32>) -> tensor<256x1x1xf32>
    %128 = stablehlo.broadcast_in_dim %127, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x28x28xf32>
    %129 = stablehlo.add %126, %128 : tensor<1x256x28x28xf32>
    %130 = stablehlo.maximum %129, %cst_114 : tensor<1x256x28x28xf32>
    %131 = stablehlo.convolution(%130, %cst_51) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x28x28xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32>
    %132 = stablehlo.reshape %cst_52 : (tensor<256xf32>) -> tensor<256x1x1xf32>
    %133 = stablehlo.broadcast_in_dim %132, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %134 = stablehlo.add %131, %133 : tensor<1x256x14x14xf32>
    %135 = stablehlo.maximum %134, %cst_115 : tensor<1x256x14x14xf32>
    %136 = stablehlo.convolution(%135, %cst_53) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %137 = stablehlo.reshape %cst_54 : (tensor<1024xf32>) -> tensor<1024x1x1xf32>
    %138 = stablehlo.broadcast_in_dim %137, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %139 = stablehlo.add %136, %138 : tensor<1x1024x14x14xf32>
    %140 = stablehlo.convolution(%125, %cst_55) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x28x28xf32>, tensor<1024x512x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %141 = stablehlo.reshape %cst_56 : (tensor<1024xf32>) -> tensor<1024x1x1xf32>
    %142 = stablehlo.broadcast_in_dim %141, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %143 = stablehlo.add %140, %142 : tensor<1x1024x14x14xf32>
    %144 = stablehlo.add %139, %143 : tensor<1x1024x14x14xf32>
    %145 = stablehlo.maximum %144, %cst_116 : tensor<1x1024x14x14xf32>
    %146 = stablehlo.convolution(%145, %cst_57) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<1x256x14x14xf32>
    %147 = stablehlo.reshape %cst_58 : (tensor<256xf32>) -> tensor<256x1x1xf32>
    %148 = stablehlo.broadcast_in_dim %147, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %149 = stablehlo.add %146, %148 : tensor<1x256x14x14xf32>
    %150 = stablehlo.maximum %149, %cst_115 : tensor<1x256x14x14xf32>
    %151 = stablehlo.convolution(%150, %cst_59) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32>
    %152 = stablehlo.reshape %cst_60 : (tensor<256xf32>) -> tensor<256x1x1xf32>
    %153 = stablehlo.broadcast_in_dim %152, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %154 = stablehlo.add %151, %153 : tensor<1x256x14x14xf32>
    %155 = stablehlo.maximum %154, %cst_115 : tensor<1x256x14x14xf32>
    %156 = stablehlo.convolution(%155, %cst_61) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %157 = stablehlo.reshape %cst_62 : (tensor<1024xf32>) -> tensor<1024x1x1xf32>
    %158 = stablehlo.broadcast_in_dim %157, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %159 = stablehlo.add %156, %158 : tensor<1x1024x14x14xf32>
    %160 = stablehlo.add %159, %145 : tensor<1x1024x14x14xf32>
    %161 = stablehlo.maximum %160, %cst_116 : tensor<1x1024x14x14xf32>
    %162 = stablehlo.convolution(%161, %cst_63) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<1x256x14x14xf32>
    %163 = stablehlo.reshape %cst_64 : (tensor<256xf32>) -> tensor<256x1x1xf32>
    %164 = stablehlo.broadcast_in_dim %163, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %165 = stablehlo.add %162, %164 : tensor<1x256x14x14xf32>
    %166 = stablehlo.maximum %165, %cst_115 : tensor<1x256x14x14xf32>
    %167 = stablehlo.convolution(%166, %cst_65) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32>
    %168 = stablehlo.reshape %cst_66 : (tensor<256xf32>) -> tensor<256x1x1xf32>
    %169 = stablehlo.broadcast_in_dim %168, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %170 = stablehlo.add %167, %169 : tensor<1x256x14x14xf32>
    %171 = stablehlo.maximum %170, %cst_115 : tensor<1x256x14x14xf32>
    %172 = stablehlo.convolution(%171, %cst_67) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %173 = stablehlo.reshape %cst_68 : (tensor<1024xf32>) -> tensor<1024x1x1xf32>
    %174 = stablehlo.broadcast_in_dim %173, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %175 = stablehlo.add %172, %174 : tensor<1x1024x14x14xf32>
    %176 = stablehlo.add %175, %161 : tensor<1x1024x14x14xf32>
    %177 = stablehlo.maximum %176, %cst_116 : tensor<1x1024x14x14xf32>
    %178 = stablehlo.convolution(%177, %cst_69) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<1x256x14x14xf32>
    %179 = stablehlo.reshape %cst_70 : (tensor<256xf32>) -> tensor<256x1x1xf32>
    %180 = stablehlo.broadcast_in_dim %179, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %181 = stablehlo.add %178, %180 : tensor<1x256x14x14xf32>
    %182 = stablehlo.maximum %181, %cst_115 : tensor<1x256x14x14xf32>
    %183 = stablehlo.convolution(%182, %cst_71) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32>
    %184 = stablehlo.reshape %cst_72 : (tensor<256xf32>) -> tensor<256x1x1xf32>
    %185 = stablehlo.broadcast_in_dim %184, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %186 = stablehlo.add %183, %185 : tensor<1x256x14x14xf32>
    %187 = stablehlo.maximum %186, %cst_115 : tensor<1x256x14x14xf32>
    %188 = stablehlo.convolution(%187, %cst_73) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %189 = stablehlo.reshape %cst_74 : (tensor<1024xf32>) -> tensor<1024x1x1xf32>
    %190 = stablehlo.broadcast_in_dim %189, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %191 = stablehlo.add %188, %190 : tensor<1x1024x14x14xf32>
    %192 = stablehlo.add %191, %177 : tensor<1x1024x14x14xf32>
    %193 = stablehlo.maximum %192, %cst_116 : tensor<1x1024x14x14xf32>
    %194 = stablehlo.convolution(%193, %cst_75) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<1x256x14x14xf32>
    %195 = stablehlo.reshape %cst_76 : (tensor<256xf32>) -> tensor<256x1x1xf32>
    %196 = stablehlo.broadcast_in_dim %195, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %197 = stablehlo.add %194, %196 : tensor<1x256x14x14xf32>
    %198 = stablehlo.maximum %197, %cst_115 : tensor<1x256x14x14xf32>
    %199 = stablehlo.convolution(%198, %cst_77) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32>
    %200 = stablehlo.reshape %cst_78 : (tensor<256xf32>) -> tensor<256x1x1xf32>
    %201 = stablehlo.broadcast_in_dim %200, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %202 = stablehlo.add %199, %201 : tensor<1x256x14x14xf32>
    %203 = stablehlo.maximum %202, %cst_115 : tensor<1x256x14x14xf32>
    %204 = stablehlo.convolution(%203, %cst_79) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %205 = stablehlo.reshape %cst_80 : (tensor<1024xf32>) -> tensor<1024x1x1xf32>
    %206 = stablehlo.broadcast_in_dim %205, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %207 = stablehlo.add %204, %206 : tensor<1x1024x14x14xf32>
    %208 = stablehlo.add %207, %193 : tensor<1x1024x14x14xf32>
    %209 = stablehlo.maximum %208, %cst_116 : tensor<1x1024x14x14xf32>
    %210 = stablehlo.convolution(%209, %cst_81) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<1x256x14x14xf32>
    %211 = stablehlo.reshape %cst_82 : (tensor<256xf32>) -> tensor<256x1x1xf32>
    %212 = stablehlo.broadcast_in_dim %211, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %213 = stablehlo.add %210, %212 : tensor<1x256x14x14xf32>
    %214 = stablehlo.maximum %213, %cst_115 : tensor<1x256x14x14xf32>
    %215 = stablehlo.convolution(%214, %cst_83) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32>
    %216 = stablehlo.reshape %cst_84 : (tensor<256xf32>) -> tensor<256x1x1xf32>
    %217 = stablehlo.broadcast_in_dim %216, dims = [1, 2, 3] : (tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %218 = stablehlo.add %215, %217 : tensor<1x256x14x14xf32>
    %219 = stablehlo.maximum %218, %cst_115 : tensor<1x256x14x14xf32>
    %220 = stablehlo.convolution(%219, %cst_85) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %221 = stablehlo.reshape %cst_86 : (tensor<1024xf32>) -> tensor<1024x1x1xf32>
    %222 = stablehlo.broadcast_in_dim %221, dims = [1, 2, 3] : (tensor<1024x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %223 = stablehlo.add %220, %222 : tensor<1x1024x14x14xf32>
    %224 = stablehlo.add %223, %209 : tensor<1x1024x14x14xf32>
    %225 = stablehlo.maximum %224, %cst_116 : tensor<1x1024x14x14xf32>
    %226 = stablehlo.convolution(%225, %cst_87) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x14x14xf32>, tensor<512x1024x1x1xf32>) -> tensor<1x512x14x14xf32>
    %227 = stablehlo.reshape %cst_88 : (tensor<512xf32>) -> tensor<512x1x1xf32>
    %228 = stablehlo.broadcast_in_dim %227, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x14x14xf32>
    %229 = stablehlo.add %226, %228 : tensor<1x512x14x14xf32>
    %230 = stablehlo.maximum %229, %cst_117 : tensor<1x512x14x14xf32>
    %231 = stablehlo.convolution(%230, %cst_89) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x14x14xf32>, tensor<512x512x3x3xf32>) -> tensor<1x512x7x7xf32>
    %232 = stablehlo.reshape %cst_90 : (tensor<512xf32>) -> tensor<512x1x1xf32>
    %233 = stablehlo.broadcast_in_dim %232, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %234 = stablehlo.add %231, %233 : tensor<1x512x7x7xf32>
    %235 = stablehlo.maximum %234, %cst_118 : tensor<1x512x7x7xf32>
    %236 = stablehlo.convolution(%235, %cst_91) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %237 = stablehlo.reshape %cst_92 : (tensor<2048xf32>) -> tensor<2048x1x1xf32>
    %238 = stablehlo.broadcast_in_dim %237, dims = [1, 2, 3] : (tensor<2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %239 = stablehlo.add %236, %238 : tensor<1x2048x7x7xf32>
    %240 = stablehlo.convolution(%225, %cst_93) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x14x14xf32>, tensor<2048x1024x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %241 = stablehlo.reshape %cst_94 : (tensor<2048xf32>) -> tensor<2048x1x1xf32>
    %242 = stablehlo.broadcast_in_dim %241, dims = [1, 2, 3] : (tensor<2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %243 = stablehlo.add %240, %242 : tensor<1x2048x7x7xf32>
    %244 = stablehlo.add %239, %243 : tensor<1x2048x7x7xf32>
    %245 = stablehlo.maximum %244, %cst_119 : tensor<1x2048x7x7xf32>
    %246 = stablehlo.convolution(%245, %cst_95) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<1x512x7x7xf32>
    %247 = stablehlo.reshape %cst_96 : (tensor<512xf32>) -> tensor<512x1x1xf32>
    %248 = stablehlo.broadcast_in_dim %247, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %249 = stablehlo.add %246, %248 : tensor<1x512x7x7xf32>
    %250 = stablehlo.maximum %249, %cst_118 : tensor<1x512x7x7xf32>
    %251 = stablehlo.convolution(%250, %cst_97) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<1x512x7x7xf32>
    %252 = stablehlo.reshape %cst_98 : (tensor<512xf32>) -> tensor<512x1x1xf32>
    %253 = stablehlo.broadcast_in_dim %252, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %254 = stablehlo.add %251, %253 : tensor<1x512x7x7xf32>
    %255 = stablehlo.maximum %254, %cst_118 : tensor<1x512x7x7xf32>
    %256 = stablehlo.convolution(%255, %cst_99) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %257 = stablehlo.reshape %cst_100 : (tensor<2048xf32>) -> tensor<2048x1x1xf32>
    %258 = stablehlo.broadcast_in_dim %257, dims = [1, 2, 3] : (tensor<2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %259 = stablehlo.add %256, %258 : tensor<1x2048x7x7xf32>
    %260 = stablehlo.add %259, %245 : tensor<1x2048x7x7xf32>
    %261 = stablehlo.maximum %260, %cst_119 : tensor<1x2048x7x7xf32>
    %262 = stablehlo.convolution(%261, %cst_101) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<1x512x7x7xf32>
    %263 = stablehlo.reshape %cst_102 : (tensor<512xf32>) -> tensor<512x1x1xf32>
    %264 = stablehlo.broadcast_in_dim %263, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %265 = stablehlo.add %262, %264 : tensor<1x512x7x7xf32>
    %266 = stablehlo.maximum %265, %cst_118 : tensor<1x512x7x7xf32>
    %267 = stablehlo.convolution(%266, %cst_103) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<1x512x7x7xf32>
    %268 = stablehlo.reshape %cst_104 : (tensor<512xf32>) -> tensor<512x1x1xf32>
    %269 = stablehlo.broadcast_in_dim %268, dims = [1, 2, 3] : (tensor<512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %270 = stablehlo.add %267, %269 : tensor<1x512x7x7xf32>
    %271 = stablehlo.maximum %270, %cst_118 : tensor<1x512x7x7xf32>
    %272 = stablehlo.convolution(%271, %cst_105) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %273 = stablehlo.reshape %cst_106 : (tensor<2048xf32>) -> tensor<2048x1x1xf32>
    %274 = stablehlo.broadcast_in_dim %273, dims = [1, 2, 3] : (tensor<2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %275 = stablehlo.add %272, %274 : tensor<1x2048x7x7xf32>
    %276 = stablehlo.add %275, %261 : tensor<1x2048x7x7xf32>
    %277 = stablehlo.maximum %276, %cst_119 : tensor<1x2048x7x7xf32>
    %278 = "stablehlo.reduce_window"(%277, %cst_120) <{padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 7, 7>, window_strides = array<i64: 1, 1, 1, 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %297 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %297 : tensor<f32>
    }) : (tensor<1x2048x7x7xf32>, tensor<f32>) -> tensor<1x2048x1x1xf32>
    %279 = stablehlo.broadcast_in_dim %cst_121, dims = [] : (tensor<f32>) -> tensor<1x2048x7x7xf32>
    %280 = "stablehlo.reduce_window"(%279, %cst_120) <{padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 7, 7>, window_strides = array<i64: 1, 1, 1, 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %297 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %297 : tensor<f32>
    }) : (tensor<1x2048x7x7xf32>, tensor<f32>) -> tensor<1x2048x1x1xf32>
    %281 = stablehlo.divide %278, %280 : tensor<1x2048x1x1xf32>
    %282 = stablehlo.reshape %281 : (tensor<1x2048x1x1xf32>) -> tensor<1x2048xf32>
    %283 = stablehlo.reshape %282 : (tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %284 = stablehlo.transpose %cst, dims = [1, 0] : (tensor<1000x2048xf32>) -> tensor<2048x1000xf32>
    %285 = stablehlo.dot_general %283, %284, contracting_dims = [1] x [0] : (tensor<1x2048xf32>, tensor<2048x1000xf32>) -> tensor<1x1000xf32>
    %286 = stablehlo.broadcast_in_dim %cst_0, dims = [1] : (tensor<1000xf32>) -> tensor<1x1000xf32>
    %287 = stablehlo.add %285, %286 : tensor<1x1000xf32>
    %288 = stablehlo.reduce(%287 init: %cst_108) applies stablehlo.maximum across dimensions = [1] : (tensor<1x1000xf32>, tensor<f32>) -> tensor<1xf32>
    %289 = stablehlo.reshape %288 : (tensor<1xf32>) -> tensor<1x1xf32>
    %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1000xf32>
    %291 = stablehlo.subtract %287, %290 : tensor<1x1000xf32>
    %292 = stablehlo.exponential %291 : tensor<1x1000xf32>
    %293 = stablehlo.reduce(%292 init: %cst_120) applies stablehlo.add across dimensions = [1] : (tensor<1x1000xf32>, tensor<f32>) -> tensor<1xf32>
    %294 = stablehlo.reshape %293 : (tensor<1xf32>) -> tensor<1x1xf32>
    %295 = stablehlo.broadcast_in_dim %294, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1000xf32>
    %296 = stablehlo.divide %292, %295 : tensor<1x1000xf32>
    return %296 : tensor<1x1000xf32>
  }
}
