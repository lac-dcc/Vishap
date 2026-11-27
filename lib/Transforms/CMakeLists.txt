add_mlir_library(VishapTransforms
  AddProbeCalls.cpp
  AnnotateDistributions.cpp
  SequentializeNestedParallelLoops.cpp

  DEPENDS
  VishapTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  VishapAnalysis
  )
