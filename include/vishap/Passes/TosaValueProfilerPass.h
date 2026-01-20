#ifndef VISHAP_PASSES_TOSA_VALUE_PROFILER_PASS_H
#define VISHAP_PASSES_TOSA_VALUE_PROFILER_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace vishap {

/// Creates the value profiler pass for TOSA tensors.
std::unique_ptr<mlir::Pass> createValueProfilerPass();

} // namespace vishap

#endif // VISHAP_PASSES_TOSA_VALUE_PROFILER_PASS_H
