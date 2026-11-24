set(VISHAP_ANALYSIS_LINK_LIBS StablehloOps)
if(OpenMP_CXX_FOUND)
  list(APPEND VISHAP_ANALYSIS_LINK_LIBS OpenMP::OpenMP_CXX)
endif()

add_mlir_library(VishapAnalysis
  DistributionAnalysis.cpp
  TensorDistribution.cpp

  LINK_LIBS PUBLIC
  ${VISHAP_ANALYSIS_LINK_LIBS}
)
