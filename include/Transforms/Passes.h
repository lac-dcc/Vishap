#ifndef VISHAP_INCLUDE_TRANSFORMS_PASSES_H
#define VISHAP_INCLUDE_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace vishap {
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Transforms/Passes.h.inc"
} // namespace vishap
} // namespace mlir

#endif // VISHAP_INCLUDE_TRANSFORMS_PASSES_H
