#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @main_graph(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %cst = arith.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 4.900000e+01 : f32
    %cst_3 = arith.constant dense_resource<__elided__> : tensor<2048x512x1x1xf32>
    %cst_4 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_5 = arith.constant dense_resource<__elided__> : tensor<512x512x3x3xf32>
    %cst_6 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_7 = arith.constant dense_resource<__elided__> : tensor<512x2048x1x1xf32>
    %cst_8 = arith.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_9 = arith.constant dense_resource<__elided__> : tensor<2048x512x1x1xf32>
    %cst_10 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_11 = arith.constant dense_resource<__elided__> : tensor<512x512x3x3xf32>
    %cst_12 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_13 = arith.constant dense_resource<__elided__> : tensor<512x2048x1x1xf32>
    %cst_14 = arith.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_15 = arith.constant dense_resource<__elided__> : tensor<2048x1024x1x1xf32>
    %cst_16 = arith.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_17 = arith.constant dense_resource<__elided__> : tensor<2048x512x1x1xf32>
    %cst_18 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_19 = arith.constant dense_resource<__elided__> : tensor<512x512x3x3xf32>
    %cst_20 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_21 = arith.constant dense_resource<__elided__> : tensor<512x1024x1x1xf32>
    %cst_22 = arith.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_23 = arith.constant dense_resource<__elided__> : tensor<1024x256x1x1xf32>
    %cst_24 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_25 = arith.constant dense_resource<__elided__> : tensor<256x256x3x3xf32>
    %cst_26 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_27 = arith.constant dense_resource<__elided__> : tensor<256x1024x1x1xf32>
    %cst_28 = arith.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_29 = arith.constant dense_resource<__elided__> : tensor<1024x256x1x1xf32>
    %cst_30 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_31 = arith.constant dense_resource<__elided__> : tensor<256x256x3x3xf32>
    %cst_32 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_33 = arith.constant dense_resource<__elided__> : tensor<256x1024x1x1xf32>
    %cst_34 = arith.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_35 = arith.constant dense_resource<__elided__> : tensor<1024x256x1x1xf32>
    %cst_36 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_37 = arith.constant dense_resource<__elided__> : tensor<256x256x3x3xf32>
    %cst_38 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_39 = arith.constant dense_resource<__elided__> : tensor<256x1024x1x1xf32>
    %cst_40 = arith.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_41 = arith.constant dense_resource<__elided__> : tensor<1024x256x1x1xf32>
    %cst_42 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_43 = arith.constant dense_resource<__elided__> : tensor<256x256x3x3xf32>
    %cst_44 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_45 = arith.constant dense_resource<__elided__> : tensor<256x1024x1x1xf32>
    %cst_46 = arith.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_47 = arith.constant dense_resource<__elided__> : tensor<1024x256x1x1xf32>
    %cst_48 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_49 = arith.constant dense_resource<__elided__> : tensor<256x256x3x3xf32>
    %cst_50 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_51 = arith.constant dense_resource<__elided__> : tensor<256x1024x1x1xf32>
    %cst_52 = arith.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_53 = arith.constant dense_resource<__elided__> : tensor<1024x512x1x1xf32>
    %cst_54 = arith.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_55 = arith.constant dense_resource<__elided__> : tensor<1024x256x1x1xf32>
    %cst_56 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_57 = arith.constant dense_resource<__elided__> : tensor<256x256x3x3xf32>
    %cst_58 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_59 = arith.constant dense_resource<__elided__> : tensor<256x512x1x1xf32>
    %cst_60 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_61 = arith.constant dense_resource<__elided__> : tensor<512x128x1x1xf32>
    %cst_62 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_63 = arith.constant dense_resource<__elided__> : tensor<128x128x3x3xf32>
    %cst_64 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_65 = arith.constant dense_resource<__elided__> : tensor<128x512x1x1xf32>
    %cst_66 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_67 = arith.constant dense_resource<__elided__> : tensor<512x128x1x1xf32>
    %cst_68 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_69 = arith.constant dense_resource<__elided__> : tensor<128x128x3x3xf32>
    %cst_70 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_71 = arith.constant dense_resource<__elided__> : tensor<128x512x1x1xf32>
    %cst_72 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_73 = arith.constant dense_resource<__elided__> : tensor<512x128x1x1xf32>
    %cst_74 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_75 = arith.constant dense_resource<__elided__> : tensor<128x128x3x3xf32>
    %cst_76 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_77 = arith.constant dense_resource<__elided__> : tensor<128x512x1x1xf32>
    %cst_78 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_79 = arith.constant dense_resource<__elided__> : tensor<512x256x1x1xf32>
    %cst_80 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_81 = arith.constant dense_resource<__elided__> : tensor<512x128x1x1xf32>
    %cst_82 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_83 = arith.constant dense_resource<__elided__> : tensor<128x128x3x3xf32>
    %cst_84 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_85 = arith.constant dense_resource<__elided__> : tensor<128x256x1x1xf32>
    %cst_86 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_87 = arith.constant dense_resource<__elided__> : tensor<256x64x1x1xf32>
    %cst_88 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_89 = arith.constant dense_resource<__elided__> : tensor<64x64x3x3xf32>
    %cst_90 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_91 = arith.constant dense_resource<__elided__> : tensor<64x256x1x1xf32>
    %cst_92 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_93 = arith.constant dense_resource<__elided__> : tensor<256x64x1x1xf32>
    %cst_94 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_95 = arith.constant dense_resource<__elided__> : tensor<64x64x3x3xf32>
    %cst_96 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_97 = arith.constant dense_resource<__elided__> : tensor<64x256x1x1xf32>
    %cst_98 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_99 = arith.constant dense_resource<__elided__> : tensor<256x64x1x1xf32>
    %cst_100 = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_101 = arith.constant dense_resource<__elided__> : tensor<256x64x1x1xf32>
    %cst_102 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_103 = arith.constant dense_resource<__elided__> : tensor<64x64x3x3xf32>
    %cst_104 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_105 = arith.constant dense_resource<__elided__> : tensor<64x64x1x1xf32>
    %cst_106 = arith.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_107 = arith.constant dense_resource<__elided__> : tensor<64x3x7x7xf32>
    %cst_108 = arith.constant dense_resource<__elided__> : tensor<1000xf32>
    %cst_109 = arith.constant dense_resource<__elided__> : tensor<1000x2048xf32>
    %padded = tensor.pad %arg0 low[0, 0, 3, 3] high[0, 0, 3, 3] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x3x224x224xf32> to tensor<1x3x230x230xf32>
    %0 = tensor.empty() : tensor<1x64x112x112xf32>
    %broadcasted = linalg.broadcast ins(%cst_106 : tensor<64xf32>) outs(%0 : tensor<1x64x112x112xf32>) dimensions = [0, 2, 3] 
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded, %cst_107 : tensor<1x3x230x230xf32>, tensor<64x3x7x7xf32>) outs(%broadcasted : tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<1x64x112x112xf32>) outs(%0 : tensor<1x64x112x112xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x64x112x112xf32>
    %padded_110 = tensor.pad %2 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x64x112x112xf32> to tensor<1x64x114x114xf32>
    %3 = tensor.empty() : tensor<1x64x56x56xf32>
    %4 = linalg.fill ins(%cst_1 : f32) outs(%3 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %5 = tensor.empty() : tensor<3x3xf32>
    %6 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_110, %5 : tensor<1x64x114x114xf32>, tensor<3x3xf32>) outs(%4 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %broadcasted_111 = linalg.broadcast ins(%cst_104 : tensor<64xf32>) outs(%3 : tensor<1x64x56x56xf32>) dimensions = [0, 2, 3] 
    %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%6, %cst_105 : tensor<1x64x56x56xf32>, tensor<64x64x1x1xf32>) outs(%broadcasted_111 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7 : tensor<1x64x56x56xf32>) outs(%3 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x64x56x56xf32>
    %padded_112 = tensor.pad %8 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %broadcasted_113 = linalg.broadcast ins(%cst_102 : tensor<64xf32>) outs(%3 : tensor<1x64x56x56xf32>) dimensions = [0, 2, 3] 
    %9 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_112, %cst_103 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%broadcasted_113 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9 : tensor<1x64x56x56xf32>) outs(%3 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x64x56x56xf32>
    %11 = tensor.empty() : tensor<1x256x56x56xf32>
    %broadcasted_114 = linalg.broadcast ins(%cst_100 : tensor<256xf32>) outs(%11 : tensor<1x256x56x56xf32>) dimensions = [0, 2, 3] 
    %12 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%10, %cst_101 : tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) outs(%broadcasted_114 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %broadcasted_115 = linalg.broadcast ins(%cst_98 : tensor<256xf32>) outs(%11 : tensor<1x256x56x56xf32>) dimensions = [0, 2, 3] 
    %13 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%6, %cst_99 : tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) outs(%broadcasted_115 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %14 = linalg.add ins(%12, %13 : tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) outs(%11 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14 : tensor<1x256x56x56xf32>) outs(%11 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x256x56x56xf32>
    %broadcasted_116 = linalg.broadcast ins(%cst_96 : tensor<64xf32>) outs(%3 : tensor<1x64x56x56xf32>) dimensions = [0, 2, 3] 
    %16 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%15, %cst_97 : tensor<1x256x56x56xf32>, tensor<64x256x1x1xf32>) outs(%broadcasted_116 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %17 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%16 : tensor<1x64x56x56xf32>) outs(%3 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x64x56x56xf32>
    %padded_117 = tensor.pad %17 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %broadcasted_118 = linalg.broadcast ins(%cst_94 : tensor<64xf32>) outs(%3 : tensor<1x64x56x56xf32>) dimensions = [0, 2, 3] 
    %18 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_117, %cst_95 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%broadcasted_118 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%18 : tensor<1x64x56x56xf32>) outs(%3 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x64x56x56xf32>
    %broadcasted_119 = linalg.broadcast ins(%cst_92 : tensor<256xf32>) outs(%11 : tensor<1x256x56x56xf32>) dimensions = [0, 2, 3] 
    %20 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%19, %cst_93 : tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) outs(%broadcasted_119 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %21 = linalg.add ins(%20, %15 : tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) outs(%11 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %22 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21 : tensor<1x256x56x56xf32>) outs(%11 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x256x56x56xf32>
    %broadcasted_120 = linalg.broadcast ins(%cst_90 : tensor<64xf32>) outs(%3 : tensor<1x64x56x56xf32>) dimensions = [0, 2, 3] 
    %23 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%22, %cst_91 : tensor<1x256x56x56xf32>, tensor<64x256x1x1xf32>) outs(%broadcasted_120 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %24 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23 : tensor<1x64x56x56xf32>) outs(%3 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x64x56x56xf32>
    %padded_121 = tensor.pad %24 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %broadcasted_122 = linalg.broadcast ins(%cst_88 : tensor<64xf32>) outs(%3 : tensor<1x64x56x56xf32>) dimensions = [0, 2, 3] 
    %25 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_121, %cst_89 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%broadcasted_122 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %26 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%25 : tensor<1x64x56x56xf32>) outs(%3 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x64x56x56xf32>
    %broadcasted_123 = linalg.broadcast ins(%cst_86 : tensor<256xf32>) outs(%11 : tensor<1x256x56x56xf32>) dimensions = [0, 2, 3] 
    %27 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%26, %cst_87 : tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) outs(%broadcasted_123 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %28 = linalg.add ins(%27, %22 : tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) outs(%11 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %29 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28 : tensor<1x256x56x56xf32>) outs(%11 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x256x56x56xf32>
    %30 = tensor.empty() : tensor<1x128x56x56xf32>
    %broadcasted_124 = linalg.broadcast ins(%cst_84 : tensor<128xf32>) outs(%30 : tensor<1x128x56x56xf32>) dimensions = [0, 2, 3] 
    %31 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%29, %cst_85 : tensor<1x256x56x56xf32>, tensor<128x256x1x1xf32>) outs(%broadcasted_124 : tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %32 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31 : tensor<1x128x56x56xf32>) outs(%30 : tensor<1x128x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x128x56x56xf32>
    %padded_125 = tensor.pad %32 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x56x56xf32> to tensor<1x128x58x58xf32>
    %33 = tensor.empty() : tensor<1x128x28x28xf32>
    %broadcasted_126 = linalg.broadcast ins(%cst_82 : tensor<128xf32>) outs(%33 : tensor<1x128x28x28xf32>) dimensions = [0, 2, 3] 
    %34 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_125, %cst_83 : tensor<1x128x58x58xf32>, tensor<128x128x3x3xf32>) outs(%broadcasted_126 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %35 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%34 : tensor<1x128x28x28xf32>) outs(%33 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x128x28x28xf32>
    %36 = tensor.empty() : tensor<1x512x28x28xf32>
    %broadcasted_127 = linalg.broadcast ins(%cst_80 : tensor<512xf32>) outs(%36 : tensor<1x512x28x28xf32>) dimensions = [0, 2, 3] 
    %37 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%35, %cst_81 : tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) outs(%broadcasted_127 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %broadcasted_128 = linalg.broadcast ins(%cst_78 : tensor<512xf32>) outs(%36 : tensor<1x512x28x28xf32>) dimensions = [0, 2, 3] 
    %38 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%29, %cst_79 : tensor<1x256x56x56xf32>, tensor<512x256x1x1xf32>) outs(%broadcasted_128 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %39 = linalg.add ins(%37, %38 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%36 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %40 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%39 : tensor<1x512x28x28xf32>) outs(%36 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x512x28x28xf32>
    %broadcasted_129 = linalg.broadcast ins(%cst_76 : tensor<128xf32>) outs(%33 : tensor<1x128x28x28xf32>) dimensions = [0, 2, 3] 
    %41 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%40, %cst_77 : tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) outs(%broadcasted_129 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %42 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%41 : tensor<1x128x28x28xf32>) outs(%33 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_130 = tensor.pad %42 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %broadcasted_131 = linalg.broadcast ins(%cst_74 : tensor<128xf32>) outs(%33 : tensor<1x128x28x28xf32>) dimensions = [0, 2, 3] 
    %43 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_130, %cst_75 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%broadcasted_131 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %44 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%43 : tensor<1x128x28x28xf32>) outs(%33 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x128x28x28xf32>
    %broadcasted_132 = linalg.broadcast ins(%cst_72 : tensor<512xf32>) outs(%36 : tensor<1x512x28x28xf32>) dimensions = [0, 2, 3] 
    %45 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%44, %cst_73 : tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) outs(%broadcasted_132 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %46 = linalg.add ins(%45, %40 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%36 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %47 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%46 : tensor<1x512x28x28xf32>) outs(%36 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x512x28x28xf32>
    %broadcasted_133 = linalg.broadcast ins(%cst_70 : tensor<128xf32>) outs(%33 : tensor<1x128x28x28xf32>) dimensions = [0, 2, 3] 
    %48 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%47, %cst_71 : tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) outs(%broadcasted_133 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %49 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%48 : tensor<1x128x28x28xf32>) outs(%33 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_134 = tensor.pad %49 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %broadcasted_135 = linalg.broadcast ins(%cst_68 : tensor<128xf32>) outs(%33 : tensor<1x128x28x28xf32>) dimensions = [0, 2, 3] 
    %50 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_134, %cst_69 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%broadcasted_135 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %51 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%50 : tensor<1x128x28x28xf32>) outs(%33 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x128x28x28xf32>
    %broadcasted_136 = linalg.broadcast ins(%cst_66 : tensor<512xf32>) outs(%36 : tensor<1x512x28x28xf32>) dimensions = [0, 2, 3] 
    %52 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%51, %cst_67 : tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) outs(%broadcasted_136 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %53 = linalg.add ins(%52, %47 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%36 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %54 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%53 : tensor<1x512x28x28xf32>) outs(%36 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x512x28x28xf32>
    %broadcasted_137 = linalg.broadcast ins(%cst_64 : tensor<128xf32>) outs(%33 : tensor<1x128x28x28xf32>) dimensions = [0, 2, 3] 
    %55 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%54, %cst_65 : tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) outs(%broadcasted_137 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %56 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%55 : tensor<1x128x28x28xf32>) outs(%33 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_138 = tensor.pad %56 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %broadcasted_139 = linalg.broadcast ins(%cst_62 : tensor<128xf32>) outs(%33 : tensor<1x128x28x28xf32>) dimensions = [0, 2, 3] 
    %57 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_138, %cst_63 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%broadcasted_139 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %58 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%57 : tensor<1x128x28x28xf32>) outs(%33 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x128x28x28xf32>
    %broadcasted_140 = linalg.broadcast ins(%cst_60 : tensor<512xf32>) outs(%36 : tensor<1x512x28x28xf32>) dimensions = [0, 2, 3] 
    %59 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%58, %cst_61 : tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) outs(%broadcasted_140 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %60 = linalg.add ins(%59, %54 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%36 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %61 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%60 : tensor<1x512x28x28xf32>) outs(%36 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x512x28x28xf32>
    %62 = tensor.empty() : tensor<1x256x28x28xf32>
    %broadcasted_141 = linalg.broadcast ins(%cst_58 : tensor<256xf32>) outs(%62 : tensor<1x256x28x28xf32>) dimensions = [0, 2, 3] 
    %63 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%61, %cst_59 : tensor<1x512x28x28xf32>, tensor<256x512x1x1xf32>) outs(%broadcasted_141 : tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %64 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%63 : tensor<1x256x28x28xf32>) outs(%62 : tensor<1x256x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x256x28x28xf32>
    %padded_142 = tensor.pad %64 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x28x28xf32> to tensor<1x256x30x30xf32>
    %65 = tensor.empty() : tensor<1x256x14x14xf32>
    %broadcasted_143 = linalg.broadcast ins(%cst_56 : tensor<256xf32>) outs(%65 : tensor<1x256x14x14xf32>) dimensions = [0, 2, 3] 
    %66 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_142, %cst_57 : tensor<1x256x30x30xf32>, tensor<256x256x3x3xf32>) outs(%broadcasted_143 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %67 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%66 : tensor<1x256x14x14xf32>) outs(%65 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x256x14x14xf32>
    %68 = tensor.empty() : tensor<1x1024x14x14xf32>
    %broadcasted_144 = linalg.broadcast ins(%cst_54 : tensor<1024xf32>) outs(%68 : tensor<1x1024x14x14xf32>) dimensions = [0, 2, 3] 
    %69 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%67, %cst_55 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%broadcasted_144 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %broadcasted_145 = linalg.broadcast ins(%cst_52 : tensor<1024xf32>) outs(%68 : tensor<1x1024x14x14xf32>) dimensions = [0, 2, 3] 
    %70 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%61, %cst_53 : tensor<1x512x28x28xf32>, tensor<1024x512x1x1xf32>) outs(%broadcasted_145 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %71 = linalg.add ins(%69, %70 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%68 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %72 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%71 : tensor<1x1024x14x14xf32>) outs(%68 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x1024x14x14xf32>
    %broadcasted_146 = linalg.broadcast ins(%cst_50 : tensor<256xf32>) outs(%65 : tensor<1x256x14x14xf32>) dimensions = [0, 2, 3] 
    %73 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%72, %cst_51 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%broadcasted_146 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %74 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%73 : tensor<1x256x14x14xf32>) outs(%65 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_147 = tensor.pad %74 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %broadcasted_148 = linalg.broadcast ins(%cst_48 : tensor<256xf32>) outs(%65 : tensor<1x256x14x14xf32>) dimensions = [0, 2, 3] 
    %75 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_147, %cst_49 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%broadcasted_148 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %76 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%75 : tensor<1x256x14x14xf32>) outs(%65 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x256x14x14xf32>
    %broadcasted_149 = linalg.broadcast ins(%cst_46 : tensor<1024xf32>) outs(%68 : tensor<1x1024x14x14xf32>) dimensions = [0, 2, 3] 
    %77 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%76, %cst_47 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%broadcasted_149 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %78 = linalg.add ins(%77, %72 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%68 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %79 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%78 : tensor<1x1024x14x14xf32>) outs(%68 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x1024x14x14xf32>
    %broadcasted_150 = linalg.broadcast ins(%cst_44 : tensor<256xf32>) outs(%65 : tensor<1x256x14x14xf32>) dimensions = [0, 2, 3] 
    %80 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%79, %cst_45 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%broadcasted_150 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %81 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%80 : tensor<1x256x14x14xf32>) outs(%65 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_151 = tensor.pad %81 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %broadcasted_152 = linalg.broadcast ins(%cst_42 : tensor<256xf32>) outs(%65 : tensor<1x256x14x14xf32>) dimensions = [0, 2, 3] 
    %82 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_151, %cst_43 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%broadcasted_152 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %83 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%82 : tensor<1x256x14x14xf32>) outs(%65 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x256x14x14xf32>
    %broadcasted_153 = linalg.broadcast ins(%cst_40 : tensor<1024xf32>) outs(%68 : tensor<1x1024x14x14xf32>) dimensions = [0, 2, 3] 
    %84 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%83, %cst_41 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%broadcasted_153 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %85 = linalg.add ins(%84, %79 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%68 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %86 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%85 : tensor<1x1024x14x14xf32>) outs(%68 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x1024x14x14xf32>
    %broadcasted_154 = linalg.broadcast ins(%cst_38 : tensor<256xf32>) outs(%65 : tensor<1x256x14x14xf32>) dimensions = [0, 2, 3] 
    %87 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%86, %cst_39 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%broadcasted_154 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %88 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%87 : tensor<1x256x14x14xf32>) outs(%65 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_155 = tensor.pad %88 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %broadcasted_156 = linalg.broadcast ins(%cst_36 : tensor<256xf32>) outs(%65 : tensor<1x256x14x14xf32>) dimensions = [0, 2, 3] 
    %89 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_155, %cst_37 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%broadcasted_156 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %90 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%89 : tensor<1x256x14x14xf32>) outs(%65 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x256x14x14xf32>
    %broadcasted_157 = linalg.broadcast ins(%cst_34 : tensor<1024xf32>) outs(%68 : tensor<1x1024x14x14xf32>) dimensions = [0, 2, 3] 
    %91 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%90, %cst_35 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%broadcasted_157 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %92 = linalg.add ins(%91, %86 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%68 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %93 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%92 : tensor<1x1024x14x14xf32>) outs(%68 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x1024x14x14xf32>
    %broadcasted_158 = linalg.broadcast ins(%cst_32 : tensor<256xf32>) outs(%65 : tensor<1x256x14x14xf32>) dimensions = [0, 2, 3] 
    %94 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%93, %cst_33 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%broadcasted_158 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %95 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%94 : tensor<1x256x14x14xf32>) outs(%65 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_159 = tensor.pad %95 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %broadcasted_160 = linalg.broadcast ins(%cst_30 : tensor<256xf32>) outs(%65 : tensor<1x256x14x14xf32>) dimensions = [0, 2, 3] 
    %96 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_159, %cst_31 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%broadcasted_160 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %97 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%96 : tensor<1x256x14x14xf32>) outs(%65 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x256x14x14xf32>
    %broadcasted_161 = linalg.broadcast ins(%cst_28 : tensor<1024xf32>) outs(%68 : tensor<1x1024x14x14xf32>) dimensions = [0, 2, 3] 
    %98 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%97, %cst_29 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%broadcasted_161 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %99 = linalg.add ins(%98, %93 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%68 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %100 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%99 : tensor<1x1024x14x14xf32>) outs(%68 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x1024x14x14xf32>
    %broadcasted_162 = linalg.broadcast ins(%cst_26 : tensor<256xf32>) outs(%65 : tensor<1x256x14x14xf32>) dimensions = [0, 2, 3] 
    %101 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%100, %cst_27 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%broadcasted_162 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %102 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%101 : tensor<1x256x14x14xf32>) outs(%65 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_163 = tensor.pad %102 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %broadcasted_164 = linalg.broadcast ins(%cst_24 : tensor<256xf32>) outs(%65 : tensor<1x256x14x14xf32>) dimensions = [0, 2, 3] 
    %103 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_163, %cst_25 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%broadcasted_164 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %104 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%103 : tensor<1x256x14x14xf32>) outs(%65 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x256x14x14xf32>
    %broadcasted_165 = linalg.broadcast ins(%cst_22 : tensor<1024xf32>) outs(%68 : tensor<1x1024x14x14xf32>) dimensions = [0, 2, 3] 
    %105 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%104, %cst_23 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%broadcasted_165 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %106 = linalg.add ins(%105, %100 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%68 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %107 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%106 : tensor<1x1024x14x14xf32>) outs(%68 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x1024x14x14xf32>
    %108 = tensor.empty() : tensor<1x512x14x14xf32>
    %broadcasted_166 = linalg.broadcast ins(%cst_20 : tensor<512xf32>) outs(%108 : tensor<1x512x14x14xf32>) dimensions = [0, 2, 3] 
    %109 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%107, %cst_21 : tensor<1x1024x14x14xf32>, tensor<512x1024x1x1xf32>) outs(%broadcasted_166 : tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
    %110 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%109 : tensor<1x512x14x14xf32>) outs(%108 : tensor<1x512x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x512x14x14xf32>
    %padded_167 = tensor.pad %110 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x512x14x14xf32> to tensor<1x512x16x16xf32>
    %111 = tensor.empty() : tensor<1x512x7x7xf32>
    %broadcasted_168 = linalg.broadcast ins(%cst_18 : tensor<512xf32>) outs(%111 : tensor<1x512x7x7xf32>) dimensions = [0, 2, 3] 
    %112 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_167, %cst_19 : tensor<1x512x16x16xf32>, tensor<512x512x3x3xf32>) outs(%broadcasted_168 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %113 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%112 : tensor<1x512x7x7xf32>) outs(%111 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x512x7x7xf32>
    %114 = tensor.empty() : tensor<1x2048x7x7xf32>
    %broadcasted_169 = linalg.broadcast ins(%cst_16 : tensor<2048xf32>) outs(%114 : tensor<1x2048x7x7xf32>) dimensions = [0, 2, 3] 
    %115 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%113, %cst_17 : tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>) outs(%broadcasted_169 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %broadcasted_170 = linalg.broadcast ins(%cst_14 : tensor<2048xf32>) outs(%114 : tensor<1x2048x7x7xf32>) dimensions = [0, 2, 3] 
    %116 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%107, %cst_15 : tensor<1x1024x14x14xf32>, tensor<2048x1024x1x1xf32>) outs(%broadcasted_170 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %117 = linalg.add ins(%115, %116 : tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) outs(%114 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %118 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%117 : tensor<1x2048x7x7xf32>) outs(%114 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x2048x7x7xf32>
    %broadcasted_171 = linalg.broadcast ins(%cst_12 : tensor<512xf32>) outs(%111 : tensor<1x512x7x7xf32>) dimensions = [0, 2, 3] 
    %119 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%118, %cst_13 : tensor<1x2048x7x7xf32>, tensor<512x2048x1x1xf32>) outs(%broadcasted_171 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %120 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%119 : tensor<1x512x7x7xf32>) outs(%111 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x512x7x7xf32>
    %padded_172 = tensor.pad %120 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x512x7x7xf32> to tensor<1x512x9x9xf32>
    %broadcasted_173 = linalg.broadcast ins(%cst_10 : tensor<512xf32>) outs(%111 : tensor<1x512x7x7xf32>) dimensions = [0, 2, 3] 
    %121 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_172, %cst_11 : tensor<1x512x9x9xf32>, tensor<512x512x3x3xf32>) outs(%broadcasted_173 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %122 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%121 : tensor<1x512x7x7xf32>) outs(%111 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x512x7x7xf32>
    %broadcasted_174 = linalg.broadcast ins(%cst_8 : tensor<2048xf32>) outs(%114 : tensor<1x2048x7x7xf32>) dimensions = [0, 2, 3] 
    %123 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%122, %cst_9 : tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>) outs(%broadcasted_174 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %124 = linalg.add ins(%123, %118 : tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) outs(%114 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %125 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%124 : tensor<1x2048x7x7xf32>) outs(%114 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x2048x7x7xf32>
    %broadcasted_175 = linalg.broadcast ins(%cst_6 : tensor<512xf32>) outs(%111 : tensor<1x512x7x7xf32>) dimensions = [0, 2, 3] 
    %126 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%125, %cst_7 : tensor<1x2048x7x7xf32>, tensor<512x2048x1x1xf32>) outs(%broadcasted_175 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %127 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%126 : tensor<1x512x7x7xf32>) outs(%111 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x512x7x7xf32>
    %padded_176 = tensor.pad %127 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x512x7x7xf32> to tensor<1x512x9x9xf32>
    %broadcasted_177 = linalg.broadcast ins(%cst_4 : tensor<512xf32>) outs(%111 : tensor<1x512x7x7xf32>) dimensions = [0, 2, 3] 
    %128 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_176, %cst_5 : tensor<1x512x9x9xf32>, tensor<512x512x3x3xf32>) outs(%broadcasted_177 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %129 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%128 : tensor<1x512x7x7xf32>) outs(%111 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x512x7x7xf32>
    %broadcasted_178 = linalg.broadcast ins(%cst : tensor<2048xf32>) outs(%114 : tensor<1x2048x7x7xf32>) dimensions = [0, 2, 3] 
    %130 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%129, %cst_3 : tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>) outs(%broadcasted_178 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %131 = linalg.add ins(%130, %125 : tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) outs(%114 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %132 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%131 : tensor<1x2048x7x7xf32>) outs(%114 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.cmpf ugt, %in, %cst_0 : f32
      %145 = arith.select %144, %in, %cst_0 : f32
      linalg.yield %145 : f32
    } -> tensor<1x2048x7x7xf32>
    %133 = tensor.empty() : tensor<1x2048x1x1xf32>
    %134 = linalg.fill ins(%cst_0 : f32) outs(%133 : tensor<1x2048x1x1xf32>) -> tensor<1x2048x1x1xf32>
    %135 = tensor.empty() : tensor<7x7xf32>
    %136 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%132, %135 : tensor<1x2048x7x7xf32>, tensor<7x7xf32>) outs(%134 : tensor<1x2048x1x1xf32>) -> tensor<1x2048x1x1xf32>
    %137 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%136 : tensor<1x2048x1x1xf32>) outs(%133 : tensor<1x2048x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.divf %in, %cst_2 : f32
      linalg.yield %144 : f32
    } -> tensor<1x2048x1x1xf32>
    %collapsed = tensor.collapse_shape %137 [[0], [1, 2, 3]] : tensor<1x2048x1x1xf32> into tensor<1x2048xf32>
    %138 = tensor.empty() : tensor<2048x1000xf32>
    %transposed = linalg.transpose ins(%cst_109 : tensor<1000x2048xf32>) outs(%138 : tensor<2048x1000xf32>) permutation = [1, 0] 
    %139 = tensor.empty() : tensor<1x1000xf32>
    %140 = linalg.fill ins(%cst_0 : f32) outs(%139 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %141 = linalg.matmul ins(%collapsed, %transposed : tensor<1x2048xf32>, tensor<2048x1000xf32>) outs(%140 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %142 = tensor.empty() : tensor<1x1000xf32>
    %broadcasted_179 = linalg.broadcast ins(%cst_108 : tensor<1000xf32>) outs(%142 : tensor<1x1000xf32>) dimensions = [0] 
    %143 = linalg.add ins(%141, %broadcasted_179 : tensor<1x1000xf32>, tensor<1x1000xf32>) outs(%139 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    return %143 : tensor<1x1000xf32>
  }
}

