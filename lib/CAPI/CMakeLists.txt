add_mlir_public_c_api_library(VishapCAPI
  Dialects.cpp
  Passes.cpp

  LINK_LIBS PUBLIC
  VishapTransforms
  MLIRProbeDialect
  MLIRProbeTransforms
)
