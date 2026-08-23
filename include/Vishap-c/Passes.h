#ifndef VISHAP_INCLUDE_VISHAP_C_PASSES_H
#define VISHAP_INCLUDE_VISHAP_C_PASSES_H

#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Register all Vishap passes and all probe passes with the global registry.
MLIR_CAPI_EXPORTED void vishapRegisterAllPasses(void);

#ifdef __cplusplus
}
#endif

#endif // VISHAP_INCLUDE_VISHAP_C_PASSES_H
