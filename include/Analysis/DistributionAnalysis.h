#ifndef VISHAP_INCLUDE_ANALYSIS_DISTRIBUTIONANALYSIS_H
#define VISHAP_INCLUDE_ANALYSIS_DISTRIBUTIONANALYSIS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <array>

namespace mlir::vishap {

/// Attribute name for function arguments distribution info.
constexpr char kArgsDistributionAttrName[] = "vishap.args_distribution";
/// Attribute name for operations results distribution info.
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

  LogicalResult visitAddOp(linalg::AddOp addOp);

  LogicalResult visitClamp(linalg::GenericOp clamp, double clampValue);

  LogicalResult visitGenericOp(linalg::GenericOp genericOp);

  /// Propagate distribution for a generic unary linalg operation
  LogicalResult visitUnaryIdentityOp(linalg::LinalgOp op);

  LogicalResult visitMatmul(linalg::MatmulOp matmulOp);

  LogicalResult visitConv2D(linalg::Conv2DNchwFchwOp convOp);

  LogicalResult visitFill(linalg::FillOp fillOp);

  LogicalResult visitPad(tensor::PadOp padOp);

  LogicalResult visitOperation(Operation *op);

  /// Get distributions for function args, if available.
  LogicalResult getDistributionForArgs(func::FuncOp funcOp);

public:
  LogicalResult run(func::FuncOp func);

  const llvm::DenseSet<Operation *> &getAnalyzedOperations() const {
    return analyzedOperations;
  }

  FailureOr<const Distribution*> getDistribution(Value value);
};

} // namespace mlir::vishap

#endif // VISHAP_INCLUDE_ANALYSIS_DISTRIBUTIONANALYSIS_H
