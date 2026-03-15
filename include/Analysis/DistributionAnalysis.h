#ifndef VISHAP_INCLUDE_ANALYSIS_DISTRIBUTIONANALYSIS_H
#define VISHAP_INCLUDE_ANALYSIS_DISTRIBUTIONANALYSIS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include <optional>

namespace mlir::vishap {
struct Distribution {
  double min;
  double max;
  double mean;
  double variance;
};

class DistributionAnalysis {
private:
  llvm::DenseMap<Value, Distribution> distributionMap;

  LogicalResult visitConstantOp(arith::ConstantOp constOp);

  LogicalResult visitOperation(Operation *op);

public:
  LogicalResult run(func::FuncOp func);
};

} // namespace mlir::vishap

#endif // VISHAP_INCLUDE_ANALYSIS_DISTRIBUTIONANALYSIS_H
