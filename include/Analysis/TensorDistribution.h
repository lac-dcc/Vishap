#ifndef VISHAP_INCLUDE_ANALYSIS_TENSORDISTRIBUTION_H
#define VISHAP_INCLUDE_ANALYSIS_TENSORDISTRIBUTION_H

#include "Support/Distribution.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>

namespace mlir::vishap {

/// Distributions of the slices of a tensor along one axis. Slice `i` along
/// axis `a` is the sub-tensor obtained by fixing axis `a` to index `i`.
struct AxisDistribution {
  /// One distribution per index of the axis. Empty means the axis is
  /// uniform: every slice is described by the tensor summary.
  llvm::SmallVector<Distribution> slices;

  bool isUniform() const { return slices.empty(); }
};

/// Abstract value of the distribution analysis. It contains a per-tensor
/// summary plus, for every axis, the marginal distributions of its slices.
class TensorDistribution {
public:
  /// Per-tensor distribution summary.
  Distribution summary;
  /// Distribution summay for all axes. `axes.size()` equals the rank of the
  /// tensor.
  llvm::SmallVector<AxisDistribution> axes;

  /// \returns A distribution that is uniform along every axis of \p type (no
  /// per-axis information).
  static TensorDistribution getUniform(const Distribution &summary,
                                       RankedTensorType type);

  /// \returns The distribution for the slice at \p index along \p axis.
  const Distribution &getSliceDist(unsigned axis, int64_t index) const;

  /// \returns The number of slices along \p axis, or 0 if uniform.
  int64_t numSlices(unsigned axis) const { return axes[axis].slices.size(); }

  /// \returns Whether the distribution for \p axis is uniform.
  bool isUniform(unsigned axis) const { return axes[axis].isUniform(); }

  /// \returns Variance of the slice means along \p axis (0 for uniform axes).
  /// This is a measure of how much structure the axis explains.
  double betweenSliceVariance(unsigned axis) const;

  /// Drops per-axis information for axes that violate the invariants w.r.t.
  /// \p type (wrong slice count, size <= 1, dynamic). Also resizes \ref axes
  /// to the rank of \p type.
  void normalizeAxes(Type type);

  /// Recomputes the summary from the per-axis information. If no axis is
  /// non-uniform, the summary is \p fallback (the result of the per-tensor
  /// transfer function). Otherwise the summary is the mixture over the most
  /// informative axis, with min/max intersected across all non-uniform axes.
  void finalize(const Distribution &fallback);

  /// Forgets all per-axis information and uses \p newSummary instead.
  void makeUniform(const Distribution &newSummary);

private:
  /// \returns The non-uniform axis with the largest \ref
  /// betweenSliceVariance, if any.
  std::optional<unsigned> mostInformativeAxis() const;

  /// \returns An equal-weight mixture of the slices along \p axis. Returns the
  /// summary if the axis is uniform.
  Distribution axisMixture(unsigned axis) const;
};

/// \returns A weighted mixture of \p dists with weights \p weights (summing to
/// 1). min/max are the extrema, the mean is the weighted mean, and the
/// variance follows the law of total variance (intra + inter component
/// variance).
Distribution mixtureDistribution(llvm::ArrayRef<Distribution> dists,
                                 llvm::ArrayRef<double> weights);

/// \returns An equal-weight mixture of \p dists.
Distribution mixtureDistribution(llvm::ArrayRef<Distribution> dists);

} // namespace mlir::vishap

#endif // VISHAP_INCLUDE_ANALYSIS_TENSORDISTRIBUTION_H
