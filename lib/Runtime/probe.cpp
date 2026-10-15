#include "Runtime/probe.h"
#include "Support/Distribution.h"
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace mlir::vishap;

/// One runtime observation of a tensor: tensor-level summary plus, for every
/// axis of size > 1, the distribution of each slice along that axis.
struct Observation {
  int32_t opID;
  int32_t resultID;
  Distribution summary;
  /// Per-axis slice distributions, as computed by
  /// \ref computeFloatRangeAndPerAxisDistributions. `axes[a][i]` is the
  /// distribution of slice `i` along axis `a`. Axes of size <= 1 have an
  /// empty vector.
  std::vector<std::vector<Distribution>> axes;
};

static std::vector<Observation> observations;

template <typename T>
static void observeMemref(UnrankedMemRefType<T> *m, int32_t opID,
                          int32_t resultID) {
  auto dynMemref = DynamicMemRefType<T>(*m);
  std::vector<int64_t> shape(dynMemref.sizes, dynMemref.sizes + dynMemref.rank);
  auto dists = computeFloatRangeAndPerAxisDistributions<T, DynamicMemRefType<T>>(
      dynMemref, shape);
  observations.push_back(
      {opID, resultID, dists.summary, std::move(dists.axes)});
}

extern "C" void _mlir_ciface_probeObserveMemrefF32(UnrankedMemRefType<float> *m,
                                                   int32_t opID,
                                                   int32_t resultID) {
  observeMemref(m, opID, resultID);
}

extern "C" void
_mlir_ciface_probeObserveMemrefF64(UnrankedMemRefType<double> *m, int32_t opID,
                                   int32_t resultID) {
  observeMemref(m, opID, resultID);
}

/// Reconstruct the report file path from a null-terminated i8 memref.
static std::string getReportPath(UnrankedMemRefType<int8_t> *pathMemref) {
  auto dynMemref = DynamicMemRefType<int8_t>(*pathMemref);
  const char *data =
      reinterpret_cast<const char *>(dynMemref.data + dynMemref.offset);
  int64_t size = dynMemref.rank > 0 ? dynMemref.sizes[0] : 0;
  std::string path(data, data + size);
  // Drop the null terminator (and any padding) from the global constant.
  if (auto pos = path.find('\0'); pos != std::string::npos)
    path.resize(pos);
  return path;
}

extern "C" PROBE_EXPORT void
_mlir_ciface_probeReport(UnrankedMemRefType<int8_t> *pathMemref) {
  std::string outPath = getReportPath(pathMemref);
  std::ofstream csvOut(outPath);
  if (!csvOut.is_open()) {
    std::cerr << "Failed to open file '" << outPath << "'\n";
    return;
  }

  // Tensor-level rows use axis = -1 and sliceIndex = -1; per-slice rows carry
  // their axis and slice index.
  constexpr char csvHeader[] =
      "opID,resultID,axis,sliceIndex,min,max,mean,variance\n";
  csvOut << csvHeader;

  csvOut << std::setprecision(16);
  auto writeRow = [&csvOut](int32_t opID, int32_t resultID, int64_t axis,
                            int64_t sliceIndex, const Distribution &dist) {
    csvOut << opID << "," << resultID << "," << axis << "," << sliceIndex << ","
           << dist.min << "," << dist.max << "," << dist.mean << ","
           << dist.variance << "\n";
  };

  for (const auto &obs : observations) {
    writeRow(obs.opID, obs.resultID, /*axis=*/-1, /*sliceIndex=*/-1,
             obs.summary);
    for (size_t axis = 0; axis < obs.axes.size(); ++axis) {
      const auto &slices = obs.axes[axis];
      for (size_t i = 0; i < slices.size(); ++i)
        writeRow(obs.opID, obs.resultID, static_cast<int64_t>(axis),
                 static_cast<int64_t>(i), slices[i]);
    }
  }

  // The runtime library stays loaded across executions in one process, so
  // drop what was just reported. Otherwise the next run appends to it.
  observations.clear();
}
