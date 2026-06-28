#ifndef VISHAP_INCLUDE_RUNTIME_PROBE_H
#define VISHAP_INCLUDE_RUNTIME_PROBE_H

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#define PROBE_EXPORT __attribute__((visibility("default")))

extern "C" PROBE_EXPORT void
_mlir_ciface_probeObserveMemrefF32(UnrankedMemRefType<float> *m, int32_t opID,
                                   int32_t resultID);

extern "C" PROBE_EXPORT void _mlir_ciface_probeReport();

#endif // VISHAP_INCLUDE_RUNTIME_PROBE_H
