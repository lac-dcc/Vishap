# Rename to VishapPasses to match what tools/vishap-opt/CMakeLists.txt expects
add_library(VishapPasses STATIC
  Passes/TosaValueProfilerPass.cpp
)

target_include_directories(VishapPasses
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(VishapPasses PRIVATE -fno-rtti)
endif()

target_link_libraries(VishapPasses
  PUBLIC
    MLIRTosaDialect
    MLIRFuncDialect
    MLIRPass
    MLIRIR
)
