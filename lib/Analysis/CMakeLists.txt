add_mlir_library(VishapAnalysis
  DistributionAnalysis.cpp

  LINK_LIBS PUBLIC
  StablehloOps
)
