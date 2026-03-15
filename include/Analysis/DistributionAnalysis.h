#ifndef VISHAP_INCLUDE_ANALYSIS_DISTRIBUTIONANALYSIS_H
#define VISHAP_INCLUDE_ANALYSIS_DISTRIBUTIONANALYSIS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DerivedTypes.h"
#include <optional>

namespace mlir::vishap {

constexpr char kDistributionAttrName[] = "vishap.distribution";

struct Distribution {
  double min;
  double max;
  double mean;
  double variance;
};

std::array<Attribute, 4>
distributionToArrayAttr(Builder &builder, const Distribution &distribution);

class DistributionAnalysis {
private:
  /// Set of operations for which we have propagated distribution info. We
  /// use this to know which operations should be annotated.
  llvm::DenseSet<Operation *> analyzedOperations;

  llvm::DenseMap<Value, Distribution> distributionMap;

  LogicalResult visitConstantOp(arith::ConstantOp constOp);

  LogicalResult visitOperation(Operation *op);

public:
  LogicalResult run(func::FuncOp func);

  const llvm::DenseSet<Operation *> &getAnalyzedOperations() const {
    return analyzedOperations;
  }

  const Distribution *getDistribution(Value value) const {
    auto it = distributionMap.find(value);
    if (it == distributionMap.end()) {
      return nullptr;
    }

    return &it->second;
  }
};

} // namespace mlir::vishap

#endif // VISHAP_INCLUDE_ANALYSIS_DISTRIBUTIONANALYSIS_H
