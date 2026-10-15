#ifndef VISHAP_INCLUDE_RUNTIME_PROBE_H
#define VISHAP_INCLUDE_RUNTIME_PROBE_H

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#define PROBE_EXPORT __attribute__((visibility("default")))

extern "C" PROBE_EXPORT void
_mlir_ciface_probeObserveMemrefF32(UnrankedMemRefType<float> *m, int32_t opID,
                                   int32_t resultID);

extern "C" PROBE_EXPORT void
_mlir_ciface_probeObserveMemrefF64(UnrankedMemRefType<double> *m, int32_t opID,
                                   int32_t resultID);

/// Write the probe report to the file whose path is given by the
/// null-terminated i8 memref \p pathMemref.
extern "C" PROBE_EXPORT void
_mlir_ciface_probeReport(UnrankedMemRefType<int8_t> *pathMemref);

#endif // VISHAP_INCLUDE_RUNTIME_PROBE_H
