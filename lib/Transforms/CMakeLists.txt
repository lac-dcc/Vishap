add_mlir_library(VishapTransforms
  DistributionPropagation.cpp

  DEPENDS
  VishapTransformsPassIncGen

  LINK_LIBS PUBLIC
  VishapAnalysis
  )
