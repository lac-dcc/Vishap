#ifndef VISHAP_INCLUDE_SUPPORT_DISTRIBUTION_H
#define VISHAP_INCLUDE_SUPPORT_DISTRIBUTION_H

#include <cmath>
#include <limits>
#include <type_traits>

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

} // namespace mlir::vishap

#endif // VISHAP_INCLUDE_SUPPORT_DISTRIBUTION_H
