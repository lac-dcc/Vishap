#ifndef VISHAP_INCLUDE_ANALYSIS_DISTRIBUTIONANALYSIS_H
#define VISHAP_INCLUDE_ANALYSIS_DISTRIBUTIONANALYSIS_H

#include "Analysis/TensorDistribution.h"
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
/// Attribute name for per-axis distribution info of operation results (debug
/// output only).
constexpr char kAxisDistributionAttrName[] = "vishap.axis_distribution";

std::array<Attribute, 4>
distributionToArrayAttr(Builder &builder, const Distribution &distribution);

class DistributionAnalysis {
public:
  struct Options {
    /// Whether to track per-axis slice distributions. When false, every value
    /// is uniform along all axes and the analysis degenerates to the per-tensor
    /// domain.
    bool perAxis = true;
  };

private:
  Options options;

  /// Set of operations for which we have propagated distribution info. We
  /// use this to know which operations should be annotated.
  llvm::DenseSet<Operation *> analyzedOperations;

  llvm::DenseMap<Value, TensorDistribution> distributionMap;

  /// Register \p dists as distributions describing the results of \p op.
  void registerDistributions(Operation *op,
                             llvm::ArrayRef<TensorDistribution> dists);

  /// Bring \p dist into canonical form for a value of type \p type: axes are
  /// normalized and, if \p recomputeSummary is set, the summary is recomputed
  /// from the per-axis information, using \p fallback (the per-tensor transfer
  /// result) when there is none. Ops that only rearrange elements (transpose,
  /// reshape, broadcast) keep \p fallback as the summary since the population
  /// of values does not change. When per-axis tracking is disabled, all axes
  /// are dropped and the summary is \p fallback.
  TensorDistribution finish(TensorDistribution dist,
                            const Distribution &fallback, Type type,
                            bool recomputeSummary = true) const;

  /// A distribution for a value of type \p type with no per-axis information.
  TensorDistribution makeUniform(const Distribution &dist, Type type) const;

  /// Compute the empirical distribution of a constant-like operation whose
  /// value is given by \p valueAttr.
  LogicalResult visitConstantLikeOp(Operation *op, Attribute valueAttr);

  LogicalResult visitAddOp(stablehlo::AddOp addOp);

  LogicalResult visitSubtractOp(stablehlo::SubtractOp subOp);

  LogicalResult visitMulOp(stablehlo::MulOp mulOp);

  LogicalResult visitDivOp(stablehlo::DivOp divOp);

  /// Propagate distribution for a value that is clamped to the interval
  /// [\p lowerBound, \p upperBound] (e.g. ReLU when the interval is
  /// [0, +inf), ReLU6 when it is [0, 6]). Either bound may be infinite for
  /// one-sided clamping.
  LogicalResult visitClamp(Operation *op, Value input, double lowerBound,
                           double upperBound);

  LogicalResult visitMaxOp(stablehlo::MaxOp maxOp);

  LogicalResult visitClampOp(stablehlo::ClampOp clampOp);

  /// Propagate distribution for a unary operation that does not change the
  /// per-tensor distribution of its input but whose per-axis structure we do
  /// not track (e.g. gather).
  LogicalResult visitUnaryIdentityOp(Operation *op);

  LogicalResult visitBroadcastInDim(stablehlo::BroadcastInDimOp broadcastOp);

  LogicalResult visitTranspose(stablehlo::TransposeOp transposeOp);

  LogicalResult visitReshape(stablehlo::ReshapeOp reshapeOp);

  LogicalResult visitSlice(stablehlo::SliceOp sliceOp);

  LogicalResult visitDotGeneral(stablehlo::DotGeneralOp dotOp);

  LogicalResult visitConvolution(stablehlo::ConvolutionOp convOp);

  LogicalResult visitReduceWindow(stablehlo::ReduceWindowOp reduceWindowOp);

  LogicalResult visitReduce(stablehlo::ReduceOp reduceOp);

  LogicalResult visitPad(stablehlo::PadOp padOp);

  LogicalResult visitConcatenate(stablehlo::ConcatenateOp concatOp);

  LogicalResult visitExpOp(stablehlo::ExpOp expOp);

  LogicalResult visitRsqrtOp(stablehlo::RsqrtOp rsqrtOp);

  LogicalResult visitConvert(stablehlo::ConvertOp convertOp);

  LogicalResult visitPreloadedOperation(Operation *op);

  LogicalResult visitOperation(Operation *op);

  /// Get distributions for function args, if available.
  LogicalResult getDistributionForArgs(func::FuncOp funcOp);

public:
  DistributionAnalysis() = default;
  explicit DistributionAnalysis(Options options) : options(options) {}

  LogicalResult run(func::FuncOp func);

  const llvm::DenseSet<Operation *> &getAnalyzedOperations() const {
    return analyzedOperations;
  }

  /// Per-tensor summary distribution of \p value.
  FailureOr<const Distribution *> getDistribution(Value value);

  /// Full per-axis distribution of \p value.
  FailureOr<const TensorDistribution *> getTensorDistribution(Value value);
};

} // namespace mlir::vishap

#endif // VISHAP_INCLUDE_ANALYSIS_DISTRIBUTIONANALYSIS_H
