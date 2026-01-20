add_executable(vishap-opt
  vishap-opt.cpp
)

# RTTI handling (match LLVM build)
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(vishap-opt PRIVATE -fno-rtti)
endif()

target_link_libraries(vishap-opt
  PRIVATE
    VishapPasses
    MLIRTosaDialect
    MLIRFuncDialect
    MLIRParser
    MLIRPass
    MLIRIR
    MLIRSupport
    LLVMSupport
)

set_target_properties(vishap-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
