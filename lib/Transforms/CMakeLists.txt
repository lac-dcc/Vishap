add_mlir_library(VishapTransforms
  AddProbeCalls.cpp
  AnnotateDistributions.cpp

  DEPENDS
  VishapTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  VishapAnalysis
  )
