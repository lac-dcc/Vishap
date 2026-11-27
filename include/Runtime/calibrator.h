#ifndef VISHAP_INCLUDE_RUNTIME_CALIBRATOR_H
#define VISHAP_INCLUDE_RUNTIME_CALIBRATOR_H

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cstddef>
#include <cstdint>

#define CALIBRATOR_EXPORT __attribute__((visibility("default")))

extern "C" {

/// C-ABI entry representing the aggregated multi-batch calibration statistics
/// of a single observed tensor (opID == -1 for function arguments, opID >= 0
/// for operations).
struct VishapCalibratorEntry {
  int32_t opID;
  int32_t resultID;
  uint64_t count;
  double min;
  double max;
  double mean;
  double variance;
};

/// mlir-probe C interface hooks for multi-batch in-memory calibration without
/// per-batch disk I/O. Uses the `vishapCalibrator*` to avoid symbol
/// collisions with the `probe` runtime lib.
CALIBRATOR_EXPORT void
_mlir_ciface_vishapCalibratorObserveMemrefF32(UnrankedMemRefType<float> *m,
                                              int32_t opID, int32_t resultID);

CALIBRATOR_EXPORT void
_mlir_ciface_vishapCalibratorObserveMemrefF64(UnrankedMemRefType<double> *m,
                                              int32_t opID, int32_t resultID);

CALIBRATOR_EXPORT void
_mlir_ciface_vishapCalibratorReport(const char *reportPath);

// ============================================================================
// External API
// ============================================================================

/// Clear all accumulated calibration statistics and reset the batch counter.
CALIBRATOR_EXPORT void vishapCalibratorReset();

/// Return the number of batches completed (invocations of probe.report) since
/// the last reset.
CALIBRATOR_EXPORT uint64_t vishapCalibratorGetNumBatches();

/// Return the number of distinct (opID, resultID) tensors observed so far.
CALIBRATOR_EXPORT size_t vishapCalibratorGetNumEntries();

/// Copy up to \p maxEntries aggregated entries sorted by (opID, resultID) into
/// \p outEntries. Returns the number of entries written.
CALIBRATOR_EXPORT size_t vishapCalibratorGetEntries(
    VishapCalibratorEntry *outEntries, size_t maxEntries);

/// Write the aggregated tensor-level report in Vishap probe CSV format to
/// \p reportPath. Returns 0 on success and non-zero on I/O failure.
CALIBRATOR_EXPORT int vishapCalibratorWriteReport(const char *reportPath);

} // extern "C"

#endif // VISHAP_INCLUDE_RUNTIME_CALIBRATOR_H
