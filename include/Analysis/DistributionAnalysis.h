#ifndef VISHAP_INCLUDE_ANALYSIS_DISTRIBUTIONANALYSIS_H
#define VISHAP_INCLUDE_ANALYSIS_DISTRIBUTIONANALYSIS_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include <optional>

namespace vishap {

struct Distribution {
  double mean;
  double variance;
};

class DistributionAnalysis {
private:
  llvm::DenseMap<mlir::Value, Distribution> distributionMap;

public:
  void visitOperation(mlir::Operation *op);
};

} // namespace vishap

#endif // VISHAP_INCLUDE_ANALYSIS_DISTRIBUTIONANALYSIS_H
