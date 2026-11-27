#include "Runtime/calibrator.h"
#include "Support/Distribution.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <vector>

using namespace mlir::vishap;

static_assert(sizeof(VishapCalibratorEntry) == 48,
              "VishapCalibratorEntry must have a packed 48-byte C ABI layout");

namespace {

struct CalibratorState {
  /// (opID, resultID) pair which is being monitored by the calibrator.
  struct EntryKey {
    int32_t opID;
    int32_t resultID;

    bool operator==(const EntryKey &other) const {
      return opID == other.opID && resultID == other.resultID;
    }
  };

  struct EntryKeyHash {
    size_t operator()(const EntryKey &key) const {
      return (static_cast<uint64_t>(static_cast<uint32_t>(key.opID)) << 32) |
             static_cast<uint64_t>(static_cast<uint32_t>(key.resultID));
    }
  };

  /// All currently existing moments for each calibrator entry.
  std::unordered_map<EntryKey, RunningMoments, EntryKeyHash> moments;

  uint64_t completedBatches = 0;

  void reset() {
    moments.clear();
    completedBatches = 0;
  }
};

/// Global state of the calibrator.
static CalibratorState gState;

RunningMoments &getOrCreateAccumulator(int32_t opID, int32_t resultID) {
  return gState.moments[{opID, resultID}];
}

template <typename T>
void observeMemref(UnrankedMemRefType<T> *m, int32_t opID, int32_t resultID) {
  DynamicMemRefType<T> memref(*m);
  const int64_t rank = memref.rank;

  int64_t expectedStride = 1;
  bool isContiguous = true;
  for (int64_t dim = rank - 1; dim >= 0; --dim) {
    const int64_t sz = memref.sizes[dim];
    if (sz <= 0) {
      (void)getOrCreateAccumulator(opID, resultID);
      return;
    }
    if (sz == 1) {
      continue;
    }
    if (memref.strides[dim] != expectedStride) {
      isContiguous = false;
    }
    expectedStride *= sz;
  }

  RunningMoments batchMoments;
  if (isContiguous) {
    // Contiguous data can be analyzed in parallel
    batchMoments = computeContiguousFloatMoments<T>(
        memref.data + memref.offset, static_cast<size_t>(expectedStride));
  } else {
    // Non-contiguous data needs special handling
    constexpr int64_t kMaxStackRank = 16;
    int64_t stackIndices[kMaxStackRank] = {};
    std::vector<int64_t> heapIndices;
    int64_t *indices = stackIndices;
    if (rank > kMaxStackRank) {
      // Keep indices in the stack up to a maximum fixed rank.
      heapIndices.assign(static_cast<size_t>(rank), 0);
      indices = heapIndices.data();
    }

    int64_t curOffset = memref.offset;
    const int64_t totalElements = expectedStride;
    // Traverse data and update running moments one-by-one
    for (int64_t elem = 0; elem < totalElements; ++elem) {
      batchMoments.update(static_cast<double>(memref.data[curOffset]));
      for (int64_t dim = rank - 1; dim >= 0; --dim) {
        if (++indices[dim] < memref.sizes[dim]) {
          curOffset += memref.strides[dim];
          break;
        }
        curOffset -= (memref.sizes[dim] - 1) * memref.strides[dim];
        indices[dim] = 0;
      }
    }
  }

  getOrCreateAccumulator(opID, resultID).merge(batchMoments);
}

VishapCalibratorEntry makeEntry(int32_t opID, int32_t resultID,
                                const RunningMoments &acc) {
  Distribution dist = acc.toDistribution();
  return VishapCalibratorEntry{
      .opID = opID,
      .resultID = resultID,
      .count = acc.count,
      .min = dist.min,
      .max = dist.max,
      .mean = dist.mean,
      .variance = dist.variance,
  };
}

std::vector<VishapCalibratorEntry> collectSortedEntries() {
  std::vector<VishapCalibratorEntry> entries;
  entries.reserve(gState.moments.size());

  for (const auto &[key, acc] : gState.moments) {
    entries.push_back(makeEntry(key.opID, key.resultID, acc));
  }

  std::sort(entries.begin(), entries.end(),
            [](const VishapCalibratorEntry &a, const VishapCalibratorEntry &b) {
              if (a.opID != b.opID)
                return a.opID < b.opID;
              return a.resultID < b.resultID;
            });
  return entries;
}

} // namespace

extern "C" CALIBRATOR_EXPORT void
_mlir_ciface_vishapCalibratorObserveMemrefF32(UnrankedMemRefType<float> *m,
                                              int32_t opID, int32_t resultID) {
  observeMemref(m, opID, resultID);
}

extern "C" CALIBRATOR_EXPORT void
_mlir_ciface_vishapCalibratorObserveMemrefF64(UnrankedMemRefType<double> *m,
                                              int32_t opID, int32_t resultID) {
  observeMemref(m, opID, resultID);
}

extern "C" CALIBRATOR_EXPORT void
_mlir_ciface_vishapCalibratorReport(const char * /*reportPath*/) {
  ++gState.completedBatches;
}

extern "C" CALIBRATOR_EXPORT void vishapCalibratorReset() { gState.reset(); }

extern "C" CALIBRATOR_EXPORT uint64_t vishapCalibratorGetNumBatches() {
  return gState.completedBatches;
}

extern "C" CALIBRATOR_EXPORT size_t vishapCalibratorGetNumEntries() {
  return gState.moments.size();
}

extern "C" CALIBRATOR_EXPORT size_t vishapCalibratorGetEntries(
    VishapCalibratorEntry *outEntries, size_t maxEntries) {
  if (!outEntries || maxEntries == 0) {
    return 0;
  }
  std::vector<VishapCalibratorEntry> entries = collectSortedEntries();
  const size_t n = std::min(entries.size(), maxEntries);
  std::copy_n(entries.data(), n, outEntries);
  return n;
}

extern "C" CALIBRATOR_EXPORT int
vishapCalibratorWriteReport(const char *reportPath) {
  if (!reportPath) {
    return 1;
  }
  std::ofstream csvOut(reportPath);
  if (!csvOut.is_open()) {
    std::cerr << "Failed to open calibration report file '" << reportPath
              << "'\n";
    return 1;
  }

  static constexpr std::string_view csvHeader =
      "opID,resultID,axis,sliceIndex,min,max,mean,variance\n";
  csvOut << csvHeader;
  csvOut << std::setprecision(16);
  for (const VishapCalibratorEntry &entry : collectSortedEntries()) {
    csvOut << entry.opID << "," << entry.resultID << ",-1,-1," << entry.min
           << "," << entry.max << "," << entry.mean << "," << entry.variance
           << "\n";
  }
  return 0;
}
