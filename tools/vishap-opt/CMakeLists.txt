add_llvm_executable(vishap-opt
  vishap-opt.cpp
)

llvm_update_compile_flags(vishap-opt)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(vishap-opt
  PRIVATE
  VishapTransforms
  MLIRProbeDialect
  MLIRProbeTransforms

  # MLIR dependencies
  MLIROptLib
  ${dialect_libs}
  ${conversion_libs}
)

mlir_check_link_libraries(mlir-opt)
