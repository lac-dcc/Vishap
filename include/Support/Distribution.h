#ifndef VISHAP_INCLUDE_SUPPORT_DISTRIBUTION_H
#define VISHAP_INCLUDE_SUPPORT_DISTRIBUTION_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

// This file defines utilities for handling distributions. This should be as
// generic as possible and not depend on LLVM/MLIR.

namespace mlir::vishap {
/// Simple struct to hold distribution information.
struct Distribution {
  double min;
  double max;
  double mean;
  double variance;
};

/// Compute distribution info for float values in \p range. We ignore non-finite
/// values (NaN/Inf) in the computation.
template <typename T, typename R,
          typename = std::enable_if_t<std::is_floating_point_v<T>>>
Distribution computeFloatRangeDistribution(R range) {
  double max = -std::numeric_limits<double>::infinity();
  double min = std::numeric_limits<double>::infinity();
  double sum = 0.0;
  size_t validElements = 0;
  for (T element : range) {
    if (!std::isfinite(element)) {
      // Handle NaN/Inf
      continue;
    }

    double elementDouble = static_cast<double>(element);
    max = max > elementDouble ? max : elementDouble;
    min = min < elementDouble ? min : elementDouble;
    sum += element;
    ++validElements;
  }

  double mean = sum / static_cast<double>(validElements);
  double variance = 0.0;
  for (T element : range) {
    if (!std::isfinite(element)) {
      // Handle NaN/Inf
      continue;
    }

    double elementDouble = static_cast<double>(element);
    variance += (elementDouble - mean) * (elementDouble - mean);
  }
  variance /= static_cast<double>(validElements);

  return {min, max, mean, variance};
}

/// Compute, for every axis of a row-major tensor of shape \p shape, the
/// distribution of each slice along that axis (i.e. the sub-tensor obtained by
/// fixing that axis to a given index).
///
/// Non-finite values are ignored, and variances are computed with a two-pass
/// algorithm, matching \ref computeFloatRangeDistribution.
template <typename T, typename R,
          typename = std::enable_if_t<std::is_floating_point_v<T>>>
std::vector<std::vector<Distribution>>
computePerAxisDistributions(R range, const std::vector<int64_t> &shape) {
  const size_t rank = shape.size();

  // Flat storage: the slices of axis `dim` start at `offsets[dim]`. Only axes
  // of size > 1 are tracked.
  std::vector<size_t> activeAxes;
  std::vector<size_t> offsets(rank, 0);
  size_t numSlices = 0;
  for (size_t dim = 0; dim < rank; ++dim) {
    if (shape[dim] > 1) {
      activeAxes.push_back(dim);
      offsets[dim] = numSlices;
      numSlices += static_cast<size_t>(shape[dim]);
    }
  }
  std::vector<Distribution> slices(
      numSlices, Distribution{.min = std::numeric_limits<double>::infinity(),
                              .max = -std::numeric_limits<double>::infinity(),
                              .mean = 0.0,
                              .variance = 0.0});
  std::vector<size_t> counts(numSlices, 0);

  std::vector<int64_t> index(rank, 0);
  auto advanceAxis = [&]() {
    for (size_t dim = rank; dim-- > 0;) {
      if (++index[dim] < shape[dim]) {
        return;
      }
      index[dim] = 0;
    }
  };

  // Pass 1: min, max and sum per slice.
  for (T element : range) {
    if (std::isfinite(element)) {
      double value = static_cast<double>(element);
      for (size_t k = 0; k < activeAxes.size(); ++k) {
        size_t axis = activeAxes[k];
        size_t pos = offsets[axis] + static_cast<size_t>(index[axis]);
        Distribution &dist = slices[pos];
        dist.min = dist.min < value ? dist.min : value;
        dist.max = dist.max > value ? dist.max : value;
        dist.mean += value;
        ++counts[pos];
      }
    }
    advanceAxis();
  }

  for (size_t i = 0; i < numSlices; ++i) {
    slices[i].mean /= static_cast<double>(counts[i]);
  }

  // Pass 2: centered second moment per slice.
  std::fill(index.begin(), index.end(), 0);
  for (T element : range) {
    if (std::isfinite(element)) {
      double value = static_cast<double>(element);
      for (size_t k = 0; k < activeAxes.size(); ++k) {
        size_t axis = activeAxes[k];
        Distribution &dist =
            slices[offsets[axis] + static_cast<size_t>(index[axis])];
        dist.variance += (value - dist.mean) * (value - dist.mean);
      }
    }
    advanceAxis();
  }

  // Final result: vector of vectors of distributions, one per-slice. Axes of
  // size <= 1 get an empty vector since their single slice is the whole tensor.
  std::vector<std::vector<Distribution>> result(rank);
  for (size_t axis : activeAxes) {
    result[axis].reserve(static_cast<size_t>(shape[axis]));
    for (int64_t i = 0; i < shape[axis]; ++i) {
      Distribution dist = slices[offsets[axis] + static_cast<size_t>(i)];
      dist.variance /=
          static_cast<double>(counts[offsets[axis] + static_cast<size_t>(i)]);
      result[axis].push_back(dist);
    }
  }

  return result;
}

} // namespace mlir::vishap

#endif // VISHAP_INCLUDE_SUPPORT_DISTRIBUTION_H
