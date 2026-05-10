set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Vishap)
add_public_tablegen_target(VishapTransformsPassIncGen)
