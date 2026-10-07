#ifndef VISHAP_INCLUDE_SUPPORT_DISTRIBUTIONTRANSFER_H
#define VISHAP_INCLUDE_SUPPORT_DISTRIBUTIONTRANSFER_H

#include "Support/Distribution.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

// Scalar transfer functions over Distribution. These are the per-tensor rules
// of the distribution analysis, factored out so they can be applied both to a
// tensor summary and to each slice of a tensor. This file must not depend on
// LLVM/MLIR.

namespace mlir::vishap {

/// A constant value.
inline Distribution pointMassDist(double value) {
  return {value, value, value, 0.0};
}

/// X + Y, assuming X and Y independent.
inline Distribution addDist(const Distribution &lhs, const Distribution &rhs) {
  Distribution out;
  out.min = lhs.min + rhs.min;
  out.max = lhs.max + rhs.max;
  out.mean = lhs.mean + rhs.mean;
  out.variance = lhs.variance + rhs.variance;
  return out;
}

/// X - Y, assuming X and Y independent.
inline Distribution subDist(const Distribution &lhs, const Distribution &rhs) {
  Distribution out;
  out.min = lhs.min - rhs.max;
  out.max = lhs.max - rhs.min;
  out.mean = lhs.mean - rhs.mean;
  out.variance = lhs.variance + rhs.variance;
  return out;
}

/// X * Y, assuming X and Y independent.
inline Distribution mulDist(const Distribution &lhs, const Distribution &rhs) {
  double prod1 = lhs.min * rhs.min;
  double prod2 = lhs.min * rhs.max;
  double prod3 = lhs.max * rhs.min;
  double prod4 = lhs.max * rhs.max;

  double mean = lhs.mean * rhs.mean;
  double variance = (lhs.variance * rhs.variance) +
                    (lhs.variance * (rhs.mean * rhs.mean)) +
                    (rhs.variance * (lhs.mean * lhs.mean));

  return Distribution{.min = std::min({prod1, prod2, prod3, prod4}),
                      .max = std::max({prod1, prod2, prod3, prod4}),
                      .mean = mean,
                      .variance = variance};
}

/// Why a division cannot be modelled.
enum class DivError { RangeContainsZero, ZeroMean };

/// X / Y, assuming X and Y independent. Fails if the divisor range contains
/// zero (unbounded result) or its mean is zero (mean estimate undefined).
/// See Seltman, "Approximations for Mean and Variance of a Ratio"
/// (https://www.stat.cmu.edu/~hseltman/files/ratio.pdf).
inline std::optional<Distribution> divDist(const Distribution &lhs,
                                           const Distribution &rhs,
                                           DivError *error = nullptr) {
  if (rhs.min <= 0.0 && rhs.max >= 0.0) {
    if (error) {
      *error = DivError::RangeContainsZero;
    }
    return std::nullopt;
  }
  if (rhs.mean == 0.0) {
    if (error) {
      *error = DivError::ZeroMean;
    }
    return std::nullopt;
  }

  Distribution out;
  double muYSq = rhs.mean * rhs.mean;
  out.mean = lhs.mean / rhs.mean;
  out.variance = lhs.variance / muYSq +
                 (lhs.mean * lhs.mean) * rhs.variance / (muYSq * muYSq);

  double c00 = lhs.min / rhs.min;
  double c01 = lhs.min / rhs.max;
  double c10 = lhs.max / rhs.min;
  double c11 = lhs.max / rhs.max;
  out.min = std::min({c00, c01, c10, c11});
  out.max = std::max({c00, c01, c10, c11});
  return out;
}

/// clamp(X, lo, hi) for X ~ N(mu, sigma^2), i.e. the winsorized variable.
/// With alpha = (lo - mu)/sigma and beta = (hi - mu)/sigma, the moments of Y
/// have a closed form in terms of the standard normal CDF Phi and PDF phi:
///   P(X < lo) = Phi(alpha),  P(X > hi) = 1 - Phi(beta)
///   E[X * 1{lo<X<hi}]   = mu*pMid - sigma*(phi(beta) - phi(alpha))
///   E[X^2 * 1{lo<X<hi}] = (mu^2 + sigma^2)*pMid
///                         - sigma^2*(beta*phi(beta) - alpha*phi(alpha))
///                         - 2*mu*sigma*(phi(beta) - phi(alpha))
/// where pMid = Phi(beta) - Phi(alpha). Infinite bounds reduce to one-sided
/// clamping (e.g. ReLU when lo = 0 and hi = +inf). Requires lo <= hi.
inline Distribution clampDist(const Distribution &in, double lowerBound,
                              double upperBound) {
  // clamp is monotone, so the output extrema are the clamped input extrema.
  double outMin = std::min(std::max(in.min, lowerBound), upperBound);
  double outMax = std::min(std::max(in.max, lowerBound), upperBound);

  double mu = in.mean;
  double sigma = std::sqrt(in.variance);

  // A degenerate (constant) input is clamped deterministically. The Gaussian
  // formulas below would produce NaN from (bound - mu) / 0.
  if (!(sigma > 0.0)) {
    double value = std::min(std::max(mu, lowerBound), upperBound);
    return Distribution{
        .min = outMin, .max = outMax, .mean = value, .variance = 0.0};
  }

  auto stdNormalCdf = [](double z) {
    return 0.5 * (1 + std::erf(z / std::sqrt(2)));
  };
  auto stdNormalPdf = [](double z) {
    return (1 / std::sqrt(2 * M_PI)) * std::exp(-0.5 * z * z);
  };

  // Handle infinite bounds explicitly: beta*phi(beta) would evaluate to
  // inf * 0 = NaN numerically, even though its limit is 0.
  double cdfLo = 0.0, pdfLo = 0.0, alphaPdfLo = 0.0;
  if (std::isfinite(lowerBound)) {
    double alpha = (lowerBound - mu) / sigma;
    cdfLo = stdNormalCdf(alpha);
    pdfLo = stdNormalPdf(alpha);
    alphaPdfLo = alpha * pdfLo;
  }

  double cdfHi = 1.0, pdfHi = 0.0, betaPdfHi = 0.0;
  if (std::isfinite(upperBound)) {
    double beta = (upperBound - mu) / sigma;
    cdfHi = stdNormalCdf(beta);
    pdfHi = stdNormalPdf(beta);
    betaPdfHi = beta * pdfHi;
  }

  // Probability of landing below, inside, and above the clamping interval.
  double probLo = cdfLo;
  double probHi = 1.0 - cdfHi;
  double probMid = cdfHi - cdfLo;

  // Partial first and second moments of X over the interval (lo, hi).
  double midMean = mu * probMid - sigma * (pdfHi - pdfLo);
  double midSecondMoment = (mu * mu + sigma * sigma) * probMid -
                           sigma * sigma * (betaPdfHi - alphaPdfLo) -
                           2 * mu * sigma * (pdfHi - pdfLo);

  // Combine with the point masses at the bounds. Infinite bounds carry zero
  // probability, so their (otherwise non-finite) contributions are dropped.
  double clampedMean = midMean;
  double clampedSecondMoment = midSecondMoment;
  if (std::isfinite(lowerBound)) {
    clampedMean += probLo * lowerBound;
    clampedSecondMoment += probLo * lowerBound * lowerBound;
  }
  if (std::isfinite(upperBound)) {
    clampedMean += probHi * upperBound;
    clampedSecondMoment += probHi * upperBound * upperBound;
  }

  // Var[Y] = E[Y^2] - E[Y]^2. Guard against small negative values caused by
  // floating-point cancellation.
  double clampedVariance =
      std::max(0.0, clampedSecondMoment - clampedMean * clampedMean);

  return Distribution{.min = outMin,
                      .max = outMax,
                      .mean = clampedMean,
                      .variance = clampedVariance};
}

/// exp(X). If X ~ Normal(mu, sigma^2), then exp(X) is log-normal with
///   E[exp(X)] = exp(mu + sigma^2/2)
///   Var[exp(X)] = (exp(sigma^2) - 1) * exp(2mu + sigma^2)
inline Distribution expDist(const Distribution &in) {
  Distribution out;
  out.mean = std::exp(in.mean + (in.variance / 2));
  out.variance =
      (std::exp(in.variance) - 1) * (std::exp(2 * in.mean + in.variance));
  out.min = std::exp(in.min);
  out.max = std::exp(in.max);
  return out;
}

/// rsqrt(X) = X^(-1/2). Fails if the input range contains values <= 0.
///
/// rsqrt is monotonically decreasing, so the interval endpoints swap. For the
/// moments, use a second-order Taylor expansion (delta method) of
/// g(X) = X^(-1/2) around the mean mu, with g'(x) = -x^(-3/2)/2 and
/// g''(x) = 3*x^(-5/2)/4:
///   E[g(X)]   ~= g(mu) + g''(mu)/2 * sigma^2
///             =  mu^(-1/2) * (1 + 3*sigma^2 / (8*mu^2))
///   Var[g(X)] ~= g'(mu)^2 * sigma^2 = sigma^2 / (4*mu^3)
/// Caveat: this approximation is only accurate when the coefficient of
/// variation sigma/mu is small; for widely spread inputs it can severely
/// underestimate the moments.
inline std::optional<Distribution> rsqrtDist(const Distribution &in) {
  if (in.min <= 0.0) {
    return std::nullopt;
  }
  double mu = in.mean;
  double sigmaSq = in.variance;
  Distribution out;
  out.min = 1.0 / std::sqrt(in.max);
  out.max = 1.0 / std::sqrt(in.min);
  out.mean = (1.0 / std::sqrt(mu)) * (1.0 + 3.0 * sigmaSq / (8.0 * mu * mu));
  out.variance = sigmaSq / (4.0 * mu * mu * mu);
  return out;
}

/// Sum of \p n independent copies of \p in.
inline Distribution sumOfIidDist(int64_t n, const Distribution &in) {
  return {.min = in.min * n,
          .max = in.max * n,
          .mean = in.mean * n,
          // FIXME: There probably is a more accurate way to estimate this
          .variance = in.variance * n};
}

/// Maximum over copies of \p in.
inline Distribution maxOfIidDist(const Distribution &in) {
  return {.min = in.min,
          .max = in.max,
          // FIXME: This estimation can be improved
          .mean = in.mean + in.variance,
          .variance = in.variance};
}

/// Distribution of a sum of many products given the accumulated \p mean and
/// \p variance of the sum. The range is bounded by a fixed number of standard
/// deviations from the mean instead of interval arithmetic, which would
/// grossly overestimate it.
///
/// FIXME: This heuristic may be incorrect or unsound. We need to test it
/// empirically, or try a more dynamic approach.
inline Distribution accumulationDist(double mean, double variance) {
  constexpr double delta = 6.0;
  Distribution out;
  out.mean = mean;
  out.variance = variance;
  out.min = mean - delta * std::sqrt(variance);
  out.max = mean + delta * std::sqrt(variance);
  return out;
}

/// Sum over \p contractionSize independent products X * Y, as in a matrix
/// multiplication or convolution.
inline Distribution contractionDist(int64_t contractionSize,
                                    const Distribution &lhs,
                                    const Distribution &rhs) {
  // Assume that the mean of each element product is lhs.mean * rhs.mean.
  // Moreover, for each element in the output, we sum contractionSize values.
  double mean = contractionSize * lhs.mean * rhs.mean;

  // Variance of a product of independent variables
  double varTerm = (lhs.variance * rhs.variance) +
                   (lhs.variance * rhs.mean * rhs.mean) +
                   (rhs.variance * lhs.mean * lhs.mean);
  return accumulationDist(mean, contractionSize * varTerm);
}

/// Mixture of \p in (with weight \p ratio) and the constant \p padValue (with
/// weight \p padRatio = 1 - ratio), as produced by padding a tensor.
inline Distribution padMixDist(const Distribution &in, double padValue,
                               double ratio, double padRatio) {
  Distribution out;
  out.mean = (ratio * in.mean) + (padRatio * padValue);
  out.variance = (ratio * in.variance) +
                 ratio * (in.mean - out.mean) * (in.mean - out.mean) +
                 padRatio * (padValue - out.mean) * (padValue - out.mean);
  out.min = std::min(in.min, padValue);
  out.max = std::max(in.max, padValue);
  return out;
}

} // namespace mlir::vishap

#endif // VISHAP_INCLUDE_SUPPORT_DISTRIBUTIONTRANSFER_H
