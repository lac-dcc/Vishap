#ifndef VISHAP_INCLUDE_SUPPORT_DISTRIBUTION_H
#define VISHAP_INCLUDE_SUPPORT_DISTRIBUTION_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

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

/// Online / parallel accumulator for exact min/max and numerically stable mean
/// and variance (Chan's parallel Welford algorithm). Non-finite values are
/// ignored.
struct RunningMoments {
  /// Number of covered elements.
  uint64_t count = 0;

  double min = std::numeric_limits<double>::infinity();
  double max = -std::numeric_limits<double>::infinity();
  double mean = 0.0;
  double m2 = 0.0;

  void reset() { *this = RunningMoments{}; }

  /// Update current moment based on value \p x.
  void update(double x) {
    if (!std::isfinite(x)) {
      return;
    }

    ++count;
    min = x < min ? x : min;
    max = x > max ? x : max;

    double delta = x - mean;
    mean += delta / static_cast<double>(count);

    double delta2 = x - mean;
    m2 += delta * delta2;
  }

  void merge(const RunningMoments &other) {
    if (other.count == 0) {
      return;
    }
    if (count == 0) {
      *this = other;
      return;
    }
    uint64_t newCount = count + other.count;

    min = other.min < min ? other.min : min;
    max = other.max > max ? other.max : max;

    double delta = other.mean - mean;
    double weightA = static_cast<double>(count);
    double weightB = static_cast<double>(other.count);
    double invTotal = 1.0 / static_cast<double>(newCount);
    mean += delta * (weightB * invTotal);
    m2 += other.m2 + delta * delta * (weightA * weightB * invTotal);

    count = newCount;
  }

  Distribution toDistribution() const {
    if (count == 0) {
      constexpr double nan = std::numeric_limits<double>::quiet_NaN();
      return {.min = std::numeric_limits<double>::infinity(),
              .max = -std::numeric_limits<double>::infinity(),
              .mean = nan,
              .variance = nan};
    }
    return {.min = min,
            .max = max,
            .mean = mean,
            .variance = m2 / static_cast<double>(count)};
  }
};

namespace detail {
template <typename T>
RunningMoments computeBlockMoments(const T *block, size_t blockLen) {
  if (blockLen == 0) {
    return {};
  }

  constexpr double posInf = std::numeric_limits<double>::infinity();
  constexpr double negInf = -std::numeric_limits<double>::infinity();

  // 4-lane independent accumulators enable packed SIMD (vminpd/vmaxpd/vaddpd)
  // and 4x ILP under strict IEEE-754 semantics (preserving NaN/Inf without
  // requiring -ffast-math reassociation).
  alignas(32) double mins[4] = {posInf, posInf, posInf, posInf};
  alignas(32) double maxs[4] = {negInf, negInf, negInf, negInf};
  alignas(32) double sums[4] = {0.0, 0.0, 0.0, 0.0};

  size_t i = 0;
  // Round blockLen down to a multiple of 4 so the 4-lane loop never reads
  // past the block. The scalar loop below handles the leftover 0–3 elements.
  const size_t vecEnd = blockLen & ~size_t{0b11};
  for (; i < vecEnd; i += 4) {
    for (size_t lane = 0; lane < 4; ++lane) {
      double v = static_cast<double>(block[i + lane]);
      mins[lane] = v < mins[lane] ? v : mins[lane];
      maxs[lane] = v > maxs[lane] ? v : maxs[lane];
      sums[lane] += v;
    }
  }

  double bMin =
      std::min(std::min(mins[0], mins[1]), std::min(mins[2], mins[3]));
  double bMax =
      std::max(std::max(maxs[0], maxs[1]), std::max(maxs[2], maxs[3]));
  double bSum = (sums[0] + sums[1]) + (sums[2] + sums[3]);

  for (; i < blockLen; ++i) {
    double v = static_cast<double>(block[i]);
    bMin = v < bMin ? v : bMin;
    bMax = v > bMax ? v : bMax;
    bSum += v;
  }

  // In IEEE-754 arithmetic, any NaN or +-Inf among block[0..blockLen) forces
  // bSum to be NaN or +-Inf. When bSum is finite, every element in the L1-hot
  // block is guaranteed finite, so we can run a 4-lane vectorized centered M2
  // pass.
  if (std::isfinite(bSum)) {
    double bMean = bSum / static_cast<double>(blockLen);
    alignas(32) double m2s[4] = {0.0, 0.0, 0.0, 0.0};
    size_t j = 0;
    for (; j < vecEnd; j += 4) {
      for (size_t lane = 0; lane < 4; ++lane) {
        double diff = static_cast<double>(block[j + lane]) - bMean;
        m2s[lane] += diff * diff;
      }
    }
    double bM2 = (m2s[0] + m2s[1]) + (m2s[2] + m2s[3]);
    for (; j < blockLen; ++j) {
      double diff = static_cast<double>(block[j]) - bMean;
      bM2 += diff * diff;
    }
    if (std::isfinite(bM2)) {
      return RunningMoments{
          .count = static_cast<uint64_t>(blockLen),
          .min = bMin,
          .max = bMax,
          .mean = bMean,
          .m2 = bM2,
      };
    }
  }

  // Fallback for blocks containing NaN/Inf (or extreme FP64 overflow).
  RunningMoments fallback;
  for (size_t k = 0; k < blockLen; ++k) {
    fallback.update(static_cast<double>(block[k]));
  }
  return fallback;
}
} // namespace detail

/// Compute RunningMoments over a contiguous array \p data of \p n
/// floats/doubles using L1-tiled branchless SIMD reduction and conditional
/// OpenMP parallelism.
template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
RunningMoments computeContiguousFloatMoments(const T *data, size_t n) {
  if (n == 0 || data == nullptr) {
    return {};
  }

  constexpr size_t kBlockBytes = 16384; // 16 KB L1D-resident tile
  constexpr size_t kBlockSize = kBlockBytes / sizeof(T);
  const size_t numBlocks = (n + kBlockSize - 1) / kBlockSize;

#ifdef _OPENMP
  constexpr size_t kParallelThreshold = 65536;
  if (n >= kParallelThreshold && omp_get_max_threads() > 1 &&
      !omp_in_parallel()) {
    std::vector<RunningMoments> threadMoments;
#pragma omp parallel
    {
#pragma omp single
      threadMoments.resize(static_cast<size_t>(omp_get_num_threads()));

      RunningMoments local;
#pragma omp for schedule(static) nowait
      for (size_t b = 0; b < numBlocks; ++b) {
        size_t offset = b * kBlockSize;
        size_t len = std::min(kBlockSize, n - offset);
        local.merge(detail::computeBlockMoments<T>(data + offset, len));
      }
      threadMoments[static_cast<size_t>(omp_get_thread_num())] = local;
    }

    RunningMoments total;
    for (const RunningMoments &tm : threadMoments) {
      total.merge(tm);
    }
    return total;
  }
#endif

  RunningMoments total;
  for (size_t b = 0; b < numBlocks; ++b) {
    size_t offset = b * kBlockSize;
    size_t len = std::min(kBlockSize, n - offset);
    total.merge(detail::computeBlockMoments<T>(data + offset, len));
  }
  return total;
}

/// Convenience wrapper returning a Distribution over a contiguous float buffer.
template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
Distribution computeContiguousFloatDistribution(const T *data, size_t n) {
  return computeContiguousFloatMoments<T>(data, n).toDistribution();
}

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

/// Tensor-level summary plus the per-axis slice distributions of one tensor.
struct FloatRangeDistributions {
  Distribution summary;
  /// Per-axis slice distributions. `axes[a][i]` is the distribution of slice
  /// `i` along axis `a`. Axes of size <= 1 have an empty vector.
  std::vector<std::vector<Distribution>> axes;
};

/// Compute the tensor summary and, for every axis of a row-major tensor of
/// shape \p shape, the distribution of each slice along that axis.
///
/// Both results come from one two-pass scan. Non-finite values are ignored,
/// and the summary updates match \ref computeFloatRangeDistribution, so the
/// two reductions stay the same as scanning for each result separately.
template <typename T, typename R,
          typename = std::enable_if_t<std::is_floating_point_v<T>>>
FloatRangeDistributions
computeFloatRangeAndPerAxisDistributions(R range,
                                         const std::vector<int64_t> &shape) {
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

  std::vector<int64_t> sliceIdx(rank, 0);
  auto advanceAxis = [&]() {
    for (size_t dim = rank; dim-- > 0;) {
      if (++sliceIdx[dim] < shape[dim]) {
        return;
      }
      sliceIdx[dim] = 0;
    }
  };

  double max = -std::numeric_limits<double>::infinity();
  double min = std::numeric_limits<double>::infinity();
  double sum = 0.0;
  size_t validElements = 0;
  // Pass 1: min, max and sum for the whole tensor and for each slice.
  for (T element : range) {
    if (std::isfinite(element)) {
      double value = static_cast<double>(element);
      max = max > value ? max : value;
      min = min < value ? min : value;
      sum += element;
      ++validElements;
      for (size_t axis : activeAxes) {
        size_t pos = offsets[axis] + static_cast<size_t>(sliceIdx[axis]);
        Distribution &dist = slices[pos];
        dist.min = dist.min < value ? dist.min : value;
        dist.max = dist.max > value ? dist.max : value;
        dist.mean += value;
        ++counts[pos];
      }
    }
    advanceAxis();
  }

  double mean = sum / static_cast<double>(validElements);
  for (size_t i = 0; i < numSlices; ++i) {
    slices[i].mean /= static_cast<double>(counts[i]);
  }

  // Pass 2: centered second moment for the whole tensor and for each slice.
  double variance = 0.0;
  std::fill(sliceIdx.begin(), sliceIdx.end(), 0);
  for (T element : range) {
    if (std::isfinite(element)) {
      double value = static_cast<double>(element);
      variance += (value - mean) * (value - mean);
      for (size_t axis : activeAxes) {
        Distribution &dist =
            slices[offsets[axis] + static_cast<size_t>(sliceIdx[axis])];
        dist.variance += (value - dist.mean) * (value - dist.mean);
      }
    }
    advanceAxis();
  }
  variance /= static_cast<double>(validElements);

  // Axes of size <= 1 get an empty vector since their single slice is the
  // whole tensor.
  std::vector<std::vector<Distribution>> axes(rank);
  for (size_t axis : activeAxes) {
    axes[axis].reserve(static_cast<size_t>(shape[axis]));
    for (int64_t i = 0; i < shape[axis]; ++i) {
      Distribution dist = slices[offsets[axis] + static_cast<size_t>(i)];
      dist.variance /=
          static_cast<double>(counts[offsets[axis] + static_cast<size_t>(i)]);
      axes[axis].push_back(dist);
    }
  }

  return FloatRangeDistributions{.summary = {min, max, mean, variance},
                                 .axes = std::move(axes)};
}

/// Compute, for every axis of a row-major tensor of shape \p shape, the
/// distribution of each slice along that axis (i.e. the sub-tensor obtained by
/// fixing that axis to a given index).
///
/// Non-finite values are ignored, and variances are computed with a two-pass
/// algorithm, matching \ref computeFloatRangeDistribution. This is the per-axis
/// half of \ref computeFloatRangeAndPerAxisDistributions.
template <typename T, typename R,
          typename = std::enable_if_t<std::is_floating_point_v<T>>>
std::vector<std::vector<Distribution>>
computePerAxisDistributions(R range, const std::vector<int64_t> &shape) {
  return computeFloatRangeAndPerAxisDistributions<T, R>(range, shape).axes;
}

} // namespace mlir::vishap

#endif // VISHAP_INCLUDE_SUPPORT_DISTRIBUTION_H
