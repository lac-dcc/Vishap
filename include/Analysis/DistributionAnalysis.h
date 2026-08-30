#ifndef VISHAP_INCLUDE_ANALYSIS_DISTRIBUTIONANALYSIS_H
#define VISHAP_INCLUDE_ANALYSIS_DISTRIBUTIONANALYSIS_H

#include "Support/Distribution.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseSet.h"
#include <array>

namespace mlir::vishap {

/// Attribute name for function arguments distribution info.
constexpr char kArgsDistributionAttrName[] = "vishap.args_distribution";
/// Attribute name for operations results distribution info.
constexpr char kDistributionAttrName[] = "vishap.distribution";

std::array<Attribute, 4>
distributionToArrayAttr(Builder &builder, const Distribution &distribution);

class DistributionAnalysis {
private:
  /// Set of operations for which we have propagated distribution info. We
  /// use this to know which operations should be annotated.
  llvm::DenseSet<Operation *> analyzedOperations;

  llvm::DenseMap<Value, Distribution> distributionMap;

  /// Register \p dists as distributions describing the results of \p op.
  void registerDistributions(Operation *op, llvm::ArrayRef<Distribution> dists);

  /// Compute the empirical distribution of a constant-like operation whose
  /// value is given by \p valueAttr.
  LogicalResult visitConstantLikeOp(Operation *op, Attribute valueAttr);

  LogicalResult visitAddOp(stablehlo::AddOp addOp);

  LogicalResult visitSubtractOp(stablehlo::SubtractOp subOp);

  LogicalResult visitMulOp(stablehlo::MulOp mulOp);

  LogicalResult visitDivOp(stablehlo::DivOp divOp);

  /// Propagate distribution for a value that is clamped from below at
  /// \p clampValue (e.g. ReLU when \p clampValue is 0).
  LogicalResult visitClamp(Operation *op, Value input, double clampValue);

  LogicalResult visitMaxOp(stablehlo::MaxOp maxOp);

  /// Propagate distribution for a unary operation that does not change the
  /// distribution of its input (e.g. transpose, reshape, broadcast).
  LogicalResult visitUnaryIdentityOp(Operation *op);

  LogicalResult visitDotGeneral(stablehlo::DotGeneralOp dotOp);

  LogicalResult visitConvolution(stablehlo::ConvolutionOp convOp);

  /// Shared transfer function for reductions (reduce / reduce_window) that
  /// fold \p reduceSize elements of \p input with a single Add or Max body.
  LogicalResult visitReductionLike(Operation *op, Value input, Region &body,
                                   int64_t reduceSize);

  LogicalResult visitReduceWindow(stablehlo::ReduceWindowOp reduceWindowOp);

  LogicalResult visitReduce(stablehlo::ReduceOp reduceOp);

  LogicalResult visitPad(stablehlo::PadOp padOp);

  LogicalResult visitConcatenate(stablehlo::ConcatenateOp concatOp);

  LogicalResult visitExpOp(stablehlo::ExpOp expOp);

  LogicalResult visitConvert(stablehlo::ConvertOp convertOp);

  LogicalResult visitPreloadedOperation(Operation *op);

  LogicalResult visitOperation(Operation *op);

  /// Get distributions for function args, if available.
  LogicalResult getDistributionForArgs(func::FuncOp funcOp);

public:
  LogicalResult run(func::FuncOp func);

  const llvm::DenseSet<Operation *> &getAnalyzedOperations() const {
    return analyzedOperations;
  }

  FailureOr<const Distribution *> getDistribution(Value value);
};

} // namespace mlir::vishap

#endif // VISHAP_INCLUDE_ANALYSIS_DISTRIBUTIONANALYSIS_H
