add_mlir_library(VishapTransforms
  AnnotateDistributions.cpp

  DEPENDS
  VishapTransformsPassIncGen

  LINK_LIBS PUBLIC
  VishapAnalysis
  )
