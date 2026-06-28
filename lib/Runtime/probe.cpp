#include "Runtime/probe.h"
#include "Support/Distribution.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <tuple>

using namespace mlir::vishap;

static std::vector<std::tuple<int32_t, int32_t, Distribution>> distributions;

extern "C" void _mlir_ciface_probeObserveMemrefF32(UnrankedMemRefType<float> *m,
                                                   int32_t opID,
                                                   int32_t resultID) {
  auto dynMemref = DynamicMemRefType<float>(*m);
  auto dist =
      computeFloatRangeDistribution<float, DynamicMemRefType<float>>(dynMemref);
  distributions.emplace_back(opID, resultID, dist);
}

// FIXME: pass file path as argument?
extern "C" PROBE_EXPORT void _mlir_ciface_probeReport() {
  constexpr char outPath[] = "vishap_probe_report.csv";
  std::ofstream csvOut(outPath);
  if (!csvOut.is_open()) {
    std::cerr << "Failed to open file '" << outPath << "'\n";
    return;
  }

  constexpr char csvHeader[] = "opID,resultID,min,max,mean,variance\n";
  csvOut << csvHeader;

  csvOut << std::setprecision(16);
  for (const auto &[opID, resultID, dist] : distributions) {
    csvOut << opID << "," << resultID << "," << dist.min << "," << dist.max
           << "," << dist.mean << "," << dist.variance << "\n";
  }
}
