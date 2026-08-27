#include "Vishap-c/Dialects.h"

#include "Dialect/Probe/IR/Probe.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Probe, probe, mlir::probe::ProbeDialect)
