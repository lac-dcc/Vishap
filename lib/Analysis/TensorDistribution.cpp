#include "Analysis/TensorDistribution.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cassert>
#include <limits>

namespace mlir::vishap {

Distribution mixtureDistribution(llvm::ArrayRef<Distribution> dists,
                                 llvm::ArrayRef<double> weights) {
  assert(dists.size() == weights.size() && !dists.empty() &&
         "Mixture needs one weight per component");

  double min = std::numeric_limits<double>::infinity();
  double max = -std::numeric_limits<double>::infinity();
  double mean = 0.0;
  for (auto [dist, weight] : llvm::zip(dists, weights)) {
    min = std::min(min, dist.min);
    max = std::max(max, dist.max);
    mean += weight * dist.mean;
  }

  double variance = 0.0;
  for (auto [dist, weight] : llvm::zip(dists, weights)) {
    // Intra-component variance + inter-component variance
    variance +=
        weight * (dist.variance + (dist.mean - mean) * (dist.mean - mean));
  }

  return Distribution{.min = min, .max = max, .mean = mean, .variance = variance};
}

Distribution mixtureDistribution(llvm::ArrayRef<Distribution> dists) {
  llvm::SmallVector<double> weights(dists.size(),
                                    1.0 / static_cast<double>(dists.size()));
  return mixtureDistribution(dists, weights);
}

TensorDistribution TensorDistribution::uniform(const Distribution &summary,
                                               Type type) {
  TensorDistribution result;
  result.summary = summary;
  if (auto rankedType = llvm::dyn_cast<RankedTensorType>(type)) {
    result.axes.resize(rankedType.getRank());
  }
  return result;
}

const Distribution &TensorDistribution::slice(unsigned axis,
                                              int64_t index) const {
  const AxisDistribution &axisDist = axes[axis];
  if (axisDist.isUniform()) {
    return summary;
  }
  assert(index >= 0 && static_cast<size_t>(index) < axisDist.slices.size() &&
         "Slice index out of range");
  return axisDist.slices[index];
}

bool TensorDistribution::hasNonUniformAxis() const {
  return llvm::any_of(axes,
                      [](const AxisDistribution &a) { return !a.isUniform(); });
}

double TensorDistribution::betweenSliceVariance(unsigned axis) const {
  const auto &slices = axes[axis].slices;
  if (slices.empty()) {
    return 0.0;
  }
  double mean = 0.0;
  for (const Distribution &dist : slices) {
    mean += dist.mean;
  }
  mean /= static_cast<double>(slices.size());
  double variance = 0.0;
  for (const Distribution &dist : slices) {
    variance += (dist.mean - mean) * (dist.mean - mean);
  }
  return variance / static_cast<double>(slices.size());
}

Distribution TensorDistribution::axisMixture(unsigned axis) const {
  if (axes[axis].isUniform()) {
    return summary;
  }
  return mixtureDistribution(axes[axis].slices);
}

std::optional<unsigned> TensorDistribution::mostInformativeAxis() const {
  std::optional<unsigned> best;
  double bestScore = -1.0;
  for (unsigned axis = 0; axis < axes.size(); ++axis) {
    if (axes[axis].isUniform()) {
      continue;
    }
    double score = betweenSliceVariance(axis);
    // NaN scores (e.g. from empty slices) never win, unless nothing else does.
    if (!best || score > bestScore) {
      best = axis;
      bestScore = score;
    }
  }
  return best;
}

void TensorDistribution::normalizeAxes(Type type) {
  auto rankedType = llvm::dyn_cast<RankedTensorType>(type);
  if (!rankedType) {
    axes.clear();
    return;
  }

  axes.resize(rankedType.getRank());
  for (auto [axis, dim] : llvm::enumerate(rankedType.getShape())) {
    if (dim == ShapedType::kDynamic || dim <= 1 ||
        static_cast<int64_t>(axes[axis].slices.size()) != dim) {
      axes[axis].slices.clear();
    }
  }
}

void TensorDistribution::finalize(const Distribution &fallback) {
  auto bestAxis = mostInformativeAxis();
  if (!bestAxis) {
    summary = fallback;
    return;
  }

  Distribution mixed = axisMixture(*bestAxis);

  // Each non-uniform axis yields a sound range for the whole tensor, so the
  // intersection is also sound (and tighter).
  double min = mixed.min;
  double max = mixed.max;
  for (unsigned axis = 0; axis < axes.size(); ++axis) {
    if (axes[axis].isUniform() || axis == *bestAxis) {
      continue;
    }
    Distribution other = axisMixture(axis);
    min = std::max(min, other.min);
    max = std::min(max, other.max);
  }
  // Approximations along different axes may disagree; keep the best axis's
  // own range if the intersection became inconsistent.
  if (min <= max) {
    mixed.min = min;
    mixed.max = max;
  }

  summary = mixed;
}

void TensorDistribution::makeUniform(const Distribution &newSummary) {
  summary = newSummary;
  for (AxisDistribution &axis : axes) {
    axis.slices.clear();
  }
}

} // namespace mlir::vishap
