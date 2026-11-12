add_mlir_library(VishapAnalysis
  DistributionAnalysis.cpp
  TensorDistribution.cpp

  LINK_LIBS PUBLIC
  StablehloOps
)
