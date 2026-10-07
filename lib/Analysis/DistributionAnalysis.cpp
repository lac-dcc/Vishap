#include "Analysis/DistributionAnalysis.h"
#include "Analysis/TensorDistribution.h"
#include "Support/Distribution.h"
#include "Support/DistributionTransfer.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <type_traits>
#include <vector>

namespace mlir::vishap {
namespace {
#define DEBUG_TYPE "vishap-analysis"

//===----------------------------------------------------------------------===//
// Constant matching helpers
//===----------------------------------------------------------------------===//

/// Look through shape-preserving ops (broadcast, reshape) and converts to find
/// a constant defining op. Returns nullptr if none is found.
Operation *getConstantDefiningOp(Operation *op) {
  while (op && llvm::isa<stablehlo::BroadcastInDimOp, stablehlo::ReshapeOp,
                         stablehlo::ConvertOp>(op)) {
    op = op->getOperand(0).getDefiningOp();
  }

  if (!op || !llvm::isa<stablehlo::ConstantOp, arith::ConstantOp>(op)) {
    return nullptr;
  }
  return op;
}

/// Get value for \p op, assuming it is a constant. Looks through
/// shape-preserving ops (broadcast, reshape) and converts.
FailureOr<Attribute> getConstantValue(Operation *op) {
  Operation *constOp = getConstantDefiningOp(op);
  if (!constOp) {
    return failure();
  }

  if (auto stablehloConst = llvm::dyn_cast<stablehlo::ConstantOp>(constOp)) {
    return stablehloConst.getValue();
  }
  return llvm::cast<arith::ConstantOp>(constOp).getValue();
}

/// Try to extract the value of a scalar or splat float/integer constant that
/// defines \p value, looking through shape-preserving ops (broadcast, reshape)
/// and converts.
FailureOr<double> getScalarConstantValue(Value value) {
  auto constantValueOrFailure = getConstantValue(value.getDefiningOp());
  if (failed(constantValueOrFailure)) {
    return failure();
  }

  Attribute valueAttr = *constantValueOrFailure;
  if (auto floatAttr = llvm::dyn_cast<FloatAttr>(valueAttr)) {
    return floatAttr.getValueAsDouble();
  }

  auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(valueAttr);
  if (!denseAttr || !denseAttr.isSplat()) {
    return failure();
  }

  if (llvm::isa<FloatType>(denseAttr.getElementType())) {
    return denseAttr.getSplatValue<APFloat>().convertToDouble();
  }

  if (auto intTy = llvm::dyn_cast<IntegerType>(denseAttr.getElementType())) {
    if (intTy.getWidth() > 64) {
      return failure();
    }
    APInt val = denseAttr.getSplatValue<APInt>();
    return static_cast<double>(intTy.isUnsigned() ? val.getZExtValue()
                                                  : val.getSExtValue());
  }

  return failure();
}

/// Look through reshape / broadcast_in_dim defining \p value.
Operation *skipReshapeAndBroadcast(Value value) {
  Operation *op = value.getDefiningOp();
  while (op &&
         llvm::isa<stablehlo::BroadcastInDimOp, stablehlo::ReshapeOp>(op)) {
    op = op->getOperand(0).getDefiningOp();
  }
  return op;
}

/// Extract the elements of the dense float constant that defines \p value as
/// doubles, looking through shape-preserving ops (broadcast, reshape) and
/// converts. Note that broadcasts do not preserve element multiplicity: the
/// values of the pre-broadcast constant are returned.
FailureOr<llvm::SmallVector<double>> getConstantTensorValues(Value value) {
  auto valueAttrOrFailure = getConstantValue(value.getDefiningOp());
  if (failed(valueAttrOrFailure)) {
    return failure();
  }

  auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(*valueAttrOrFailure);
  if (!denseAttr) {
    return failure();
  }

  Type elementType = denseAttr.getElementType();
  llvm::SmallVector<double> values;
  if (elementType.isF32()) {
    llvm::append_range(
        values, llvm::map_range(denseAttr.getValues<float>(), [](float v) {
          return static_cast<double>(v);
        }));
  } else if (elementType.isF64()) {
    llvm::append_range(values, denseAttr.getValues<double>());
  } else {
    return failure();
  }
  return values;
}

/// Result of matching a stablehlo.rsqrt over a constant expression:
/// rsqrt(values + epsilon).
struct ConstantRsqrtInfo {
  /// Per-element values of the constant operand (before the epsilon addition).
  llvm::SmallVector<double> values;
  /// Scalar added to the values before the rsqrt (0 if there is no addition).
  /// In a batch normalization this is the numerical-stability constant
  /// (typically ~1e-5) added to the running variance before taking
  /// 1/sqrt(var + eps), so that channels with near-zero variance do not
  /// produce a division by zero / huge scale factor.
  double epsilon;
};

/// Match a stablehlo.rsqrt whose operand is a constant expression: either a
/// plain constant, or add(const, splat_const) as produced by decomposed batch
/// normalizations (rsqrt(running_var + eps)). Fails if any element of
/// values + epsilon is not strictly positive and finite.
FailureOr<ConstantRsqrtInfo> matchConstantRsqrt(stablehlo::RsqrtOp rsqrtOp) {
  Value operand = rsqrtOp.getOperand();

  ConstantRsqrtInfo info;
  auto directOrFailure = getConstantTensorValues(operand);
  if (succeeded(directOrFailure)) {
    info.values = std::move(*directOrFailure);
    info.epsilon = 0.0;
  } else {
    auto addOp =
        llvm::dyn_cast_or_null<stablehlo::AddOp>(operand.getDefiningOp());
    if (!addOp) {
      return failure();
    }

    // One operand is a dense constant, the other a scalar/splat epsilon. Try
    // both orders (the choice does not matter mathematically if both succeed).
    bool matched = false;
    for (auto [vectorVal, scalarVal] :
         {std::make_pair(addOp.getLhs(), addOp.getRhs()),
          std::make_pair(addOp.getRhs(), addOp.getLhs())}) {
      auto epsOrFailure = getScalarConstantValue(scalarVal);
      auto valuesOrFailure = getConstantTensorValues(vectorVal);
      if (failed(epsOrFailure) || failed(valuesOrFailure)) {
        continue;
      }
      info.values = std::move(*valuesOrFailure);
      info.epsilon = *epsOrFailure;
      matched = true;
      break;
    }
    if (!matched) {
      return failure();
    }
  }

  if (info.values.empty() || llvm::any_of(info.values, [&](double v) {
        double sum = v + info.epsilon;
        return !std::isfinite(sum) || sum <= 0.0;
      })) {
    return failure();
  }

  return info;
}

/// Result of matching the normalization multiply of a decomposed batch
/// normalization.
///
/// The full per-channel vectors are kept (rather than their summary
/// statistics) because the per-tensor transfer function needs channel-wise
/// *pairs* (mean_c, variance_c): the output extrema and the mixture variance
/// E_c[v_c / (v_c + eps)] pair each channel's mean with that same channel's
/// variance, a correspondence that cannot be reconstructed from aggregate
/// statistics of the two tensors.
struct BatchNormNormalizeInfo {
  /// The activation being normalized.
  Value input;
  /// The centered activation, sub(x, bcast(mean_const)).
  Value normalized;
  /// The broadcast scale, bcast(rsqrt(var_const + eps)).
  Value scale;
  /// Per-channel running means stored in the graph.
  llvm::SmallVector<double> means;
  /// Per-channel running variances stored in the graph.
  llvm::SmallVector<double> variances;
  /// Numerical-stability constant (typically ~1e-5) added to the running
  /// variance before the rsqrt, so channels with near-zero variance do not
  /// produce a division by zero / huge scale factor.
  double epsilon;
};

/// Match the normalization multiply of a decomposed batch normalization:
///   mul(sub(x, bcast(mean_const)), bcast(rsqrt(var_const + eps)))
/// in either operand order.
FailureOr<BatchNormNormalizeInfo>
matchBatchNormNormalize(stablehlo::MulOp mulOp) {
  for (auto [subValue, rsqrtValue] :
       {std::make_pair(mulOp.getLhs(), mulOp.getRhs()),
        std::make_pair(mulOp.getRhs(), mulOp.getLhs())}) {
    auto rsqrtOp = llvm::dyn_cast_or_null<stablehlo::RsqrtOp>(
        skipReshapeAndBroadcast(rsqrtValue));
    if (!rsqrtOp) {
      continue;
    }

    auto subOp =
        llvm::dyn_cast_or_null<stablehlo::SubtractOp>(subValue.getDefiningOp());
    if (!subOp) {
      continue;
    }

    auto rsqrtInfoOrFailure = matchConstantRsqrt(rsqrtOp);
    if (failed(rsqrtInfoOrFailure)) {
      continue;
    }

    auto meansOrFailure = getConstantTensorValues(subOp.getRhs());
    if (failed(meansOrFailure) ||
        meansOrFailure->size() != rsqrtInfoOrFailure->values.size()) {
      continue;
    }

    return BatchNormNormalizeInfo{subOp.getLhs(),
                                  subValue,
                                  rsqrtValue,
                                  std::move(*meansOrFailure),
                                  std::move(rsqrtInfoOrFailure->values),
                                  rsqrtInfoOrFailure->epsilon};
  }
  return failure();
}

/// Check if \p divOp corresponds to a softmax/L1-normalize: x /
/// broadcast(sum(x)). Looks through reshape and broadcast on the divisor. On
/// success, returns the number of elements folded into each sum (e.g., the
/// class count for a typical softmax.).
FailureOr<int64_t> matchSoftmax(stablehlo::DivOp divOp) {
  Operation *divisorDef = skipReshapeAndBroadcast(divOp.getRhs());
  auto reduceOp = llvm::dyn_cast_or_null<stablehlo::ReduceOp>(divisorDef);
  if (!reduceOp || reduceOp.getInputs().size() != 1) {
    return failure();
  }

  // Body must be a single add followed by the return terminator.
  auto &bodyOps = reduceOp.getBody().front().getOperations();
  if (bodyOps.size() != 2 || !llvm::isa<stablehlo::AddOp>(bodyOps.front())) {
    return failure();
  }

  if (reduceOp.getInputs().front() != divOp.getLhs()) {
    return failure();
  }

  auto inputShape =
      llvm::cast<ShapedType>(reduceOp.getInputs().front().getType()).getShape();
  int64_t reduceSize = 1;
  for (int64_t dim : reduceOp.getDimensions()) {
    if (inputShape[dim] == ShapedType::kDynamic) {
      return failure();
    }
    reduceSize *= inputShape[dim];
  }
  if (reduceSize <= 0) {
    return failure();
  }
  return reduceSize;
}

//===----------------------------------------------------------------------===//
// Constant distributions
//===----------------------------------------------------------------------===//

/// Compute the distribution of a constant tensor of type \p type whose
/// elements are given by \p range (iterable twice), including per-axis slice
/// distributions when \p perAxis is set and the constant is not a splat.
template <typename FloatT, typename R>
TensorDistribution computeConstantTensorDistribution(R range,
                                                     RankedTensorType type,
                                                     bool perAxis,
                                                     bool isSplat) {
  Distribution summary = computeFloatRangeDistribution<FloatT, R>(range);
  TensorDistribution result = TensorDistribution::uniform(summary, type);
  if (!perAxis || isSplat || !type.hasStaticShape()) {
    return result;
  }

  std::vector<int64_t> shape(type.getShape().begin(), type.getShape().end());
  auto perAxisDists = computePerAxisDistributions<FloatT, R>(range, shape);
  for (auto [axis, slices] : llvm::enumerate(perAxisDists)) {
    result.axes[axis].slices.assign(slices.begin(), slices.end());
  }
  return result;
}

/// Convert a constant \p denseAttr to target float type \p targetType and
/// compute its empirical distribution. Only types with native C++ casts are
/// supported.
template <typename FloatT>
FailureOr<TensorDistribution>
computeConvertedConstantDistribution(DenseElementsAttr denseAttr,
                                     RankedTensorType resultType,
                                     bool perAxis) {
  static_assert(std::is_same_v<FloatT, float> || std::is_same_v<FloatT, double>,
                "FloatT must be float or double");

  Type srcType = denseAttr.getElementType();
  assert(srcType != resultType.getElementType() &&
         "Source and destination types must be different for conversion.");
  bool isSplat = denseAttr.isSplat();

  // f32 <-> f64
  if (srcType.isF32()) {
    auto mapped = llvm::map_range(denseAttr.getValues<float>(), [](float v) {
      return static_cast<FloatT>(v);
    });
    return computeConstantTensorDistribution<FloatT, decltype(mapped)>(
        mapped, resultType, perAxis, isSplat);
  }
  if (srcType.isF64()) {
    auto mapped = llvm::map_range(denseAttr.getValues<double>(), [](double v) {
      return static_cast<FloatT>(v);
    });
    return computeConstantTensorDistribution<FloatT, decltype(mapped)>(
        mapped, resultType, perAxis, isSplat);
  }

  // integer -> float (bitwidth <= 64)
  if (auto intTy = llvm::dyn_cast<IntegerType>(srcType)) {
    if (intTy.getWidth() <= 64) {
      // Signless/signed → signed convert (matches StableHLO: !isUnsigned)
      bool isSigned = !intTy.isUnsigned();
      auto mapped = llvm::map_range(
          denseAttr.getValues<APInt>(), [=](const APInt &val) -> FloatT {
            return static_cast<FloatT>(isSigned ? val.getSExtValue()
                                                : val.getZExtValue());
          });
      return computeConstantTensorDistribution<FloatT, decltype(mapped)>(
          mapped, resultType, perAxis, isSplat);
    }

    // Treat wide integers as unsupported
    return failure();
  }

  // Treat other float types (e.g., f16) as unsupported
  return failure();
}

/// Convert \p arrayAttr to a Distribution struct. Emit an error and return
/// failure if the attribute is not well-formed.
inline FailureOr<Distribution> arrayAttrToDistribution(Location loc,
                                                       Attribute attr) {
  auto arrayAttr = llvm::dyn_cast<ArrayAttr>(attr);
  if (!arrayAttr) {
    return mlir::emitError(loc) << "Invalid attribute. Expected ArrayAttr";
  }

  if (arrayAttr.size() != 4) {
    mlir::emitError(loc)
        << "Expected array attribute of size 4 for distribution info, got "
        << arrayAttr.size();
    return failure();
  }

  if (llvm::any_of(arrayAttr,
                   [](auto attr) { return !llvm::isa<FloatAttr>(attr); })) {
    return mlir::emitError(loc) << "Expected float type for all values in "
                                   "distribution attribute";
  }

  return Distribution{
      .min = llvm::cast<FloatAttr>(arrayAttr[0]).getValueAsDouble(),
      .max = llvm::cast<FloatAttr>(arrayAttr[1]).getValueAsDouble(),
      .mean = llvm::cast<FloatAttr>(arrayAttr[2]).getValueAsDouble(),
      .variance = llvm::cast<FloatAttr>(arrayAttr[3]).getValueAsDouble(),
  };
}

//===----------------------------------------------------------------------===//
// Per-axis lifting helpers
//===----------------------------------------------------------------------===//

/// Shape of \p type if it is a ranked tensor with static shape.
std::optional<llvm::ArrayRef<int64_t>> getStaticShape(Type type) {
  auto rankedType = llvm::dyn_cast<RankedTensorType>(type);
  if (!rankedType || !rankedType.hasStaticShape()) {
    return std::nullopt;
  }
  return rankedType.getShape();
}

int64_t productOf(llvm::ArrayRef<int64_t> values) {
  return std::accumulate(values.begin(), values.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

/// Apply the binary scalar transfer function \p f to the summaries and to
/// every pair of corresponding slices of \p lhs and \p rhs (which must have
/// the same shape, as StableHLO elementwise ops have no implicit broadcast).
/// Returns failure if \p f fails on the summaries; if it fails on a slice, the
/// corresponding axis is left uniform. The result is not finalized.
template <typename Fn>
FailureOr<TensorDistribution> zipSlices(const TensorDistribution &lhs,
                                        const TensorDistribution &rhs, Fn f) {
  std::optional<Distribution> summary = f(lhs.summary, rhs.summary);
  if (!summary) {
    return failure();
  }

  TensorDistribution out;
  out.summary = *summary;
  size_t rank = std::min(lhs.axes.size(), rhs.axes.size());
  out.axes.resize(rank);
  for (unsigned axis = 0; axis < rank; ++axis) {
    if (lhs.isUniform(axis) && rhs.isUniform(axis)) {
      continue;
    }
    if (!lhs.isUniform(axis) && !rhs.isUniform(axis) &&
        lhs.numSlices(axis) != rhs.numSlices(axis)) {
      continue;
    }
    int64_t n = std::max(lhs.numSlices(axis), rhs.numSlices(axis));
    llvm::SmallVector<Distribution> slices;
    slices.reserve(n);
    bool ok = true;
    for (int64_t i = 0; i < n && ok; ++i) {
      std::optional<Distribution> d = f(lhs.slice(axis, i), rhs.slice(axis, i));
      if (!d) {
        ok = false;
        break;
      }
      slices.push_back(*d);
    }
    if (ok) {
      out.axes[axis].slices = std::move(slices);
    }
  }
  return out;
}

/// Unary counterpart of \ref zipSlices.
template <typename Fn>
FailureOr<TensorDistribution> mapSlices(const TensorDistribution &in, Fn f) {
  std::optional<Distribution> summary = f(in.summary);
  if (!summary) {
    return failure();
  }

  TensorDistribution out;
  out.summary = *summary;
  out.axes.resize(in.axes.size());
  for (unsigned axis = 0; axis < in.axes.size(); ++axis) {
    if (in.isUniform(axis)) {
      continue;
    }
    llvm::SmallVector<Distribution> slices;
    slices.reserve(in.numSlices(axis));
    bool ok = true;
    for (const Distribution &slice : in.axes[axis].slices) {
      std::optional<Distribution> d = f(slice);
      if (!d) {
        ok = false;
        break;
      }
      slices.push_back(*d);
    }
    if (ok) {
      out.axes[axis].slices = std::move(slices);
    }
  }
  return out;
}

/// Sum of \p count products of \p x with the slices of \p y along one of the
/// contracting axes \p yContractAxes. Uses the most informative non-uniform
/// contracting axis of \p y; if there is none, falls back to the per-tensor
/// contraction transfer.
Distribution contractionOverAxis(const Distribution &x,
                                 const TensorDistribution &y,
                                 llvm::ArrayRef<int64_t> yContractAxes,
                                 int64_t contractionSize) {
  std::optional<unsigned> best;
  double bestScore = -1.0;
  for (int64_t axis : yContractAxes) {
    if (y.isUniform(axis)) {
      continue;
    }
    double score = y.betweenSliceVariance(axis);
    if (!best || score > bestScore) {
      best = axis;
      bestScore = score;
    }
  }
  if (!best) {
    return contractionDist(contractionSize, x, y.summary);
  }

  int64_t n = y.numSlices(*best);
  double multiplicity = static_cast<double>(contractionSize) / n;
  double mean = 0.0;
  double variance = 0.0;
  for (const Distribution &slice : y.axes[*best].slices) {
    Distribution product = mulDist(x, slice);
    mean += product.mean;
    variance += product.variance;
  }
  return accumulationDist(mean * multiplicity, variance * multiplicity);
}

/// Per-tensor result of a contraction of \p lhs and \p rhs over the paired
/// contracting axes, exploiting per-slice information along the most
/// informative pair. Returns nullopt if no contracting axis carries per-slice
/// information (the caller should use \ref contractionDist on the summaries).
std::optional<Distribution>
pairedContraction(const TensorDistribution &lhs, const TensorDistribution &rhs,
                  llvm::ArrayRef<int64_t> lhsContractAxes,
                  llvm::ArrayRef<int64_t> rhsContractAxes,
                  int64_t contractionSize) {
  std::optional<size_t> best;
  double bestScore = -1.0;
  for (size_t k = 0; k < lhsContractAxes.size(); ++k) {
    int64_t la = lhsContractAxes[k];
    int64_t ra = rhsContractAxes[k];
    if (lhs.isUniform(la) && rhs.isUniform(ra)) {
      continue;
    }
    if (!lhs.isUniform(la) && !rhs.isUniform(ra) &&
        lhs.numSlices(la) != rhs.numSlices(ra)) {
      continue;
    }
    double score =
        std::max(lhs.betweenSliceVariance(la), rhs.betweenSliceVariance(ra));
    if (!best || score > bestScore) {
      best = k;
      bestScore = score;
    }
  }
  if (!best) {
    return std::nullopt;
  }

  int64_t la = lhsContractAxes[*best];
  int64_t ra = rhsContractAxes[*best];
  int64_t n = std::max(lhs.numSlices(la), rhs.numSlices(ra));
  double multiplicity = static_cast<double>(contractionSize) / n;
  double mean = 0.0;
  double variance = 0.0;
  for (int64_t j = 0; j < n; ++j) {
    Distribution product = mulDist(lhs.slice(la, j), rhs.slice(ra, j));
    mean += product.mean;
    variance += product.variance;
  }
  return accumulationDist(mean * multiplicity, variance * multiplicity);
}

/// Translate \p dist by \p shift.
Distribution shiftDist(Distribution dist, double shift) {
  dist.min += shift;
  dist.max += shift;
  dist.mean += shift;
  return dist;
}

/// One group of a reshape: the source axes in `srcDims` are reshaped into the
/// target axes in `dstDims` (same number of elements on both sides).
struct ReshapeGroup {
  llvm::SmallVector<unsigned> srcDims;
  llvm::SmallVector<unsigned> dstDims;
};

/// Decompose a reshape from \p src to \p dst into groups of consecutive axes
/// whose sizes multiply to the same value (greedy). Leading/trailing size-1
/// axes that do not fit a group are left out (they are always uniform).
std::optional<llvm::SmallVector<ReshapeGroup>>
computeReshapeGroups(llvm::ArrayRef<int64_t> src, llvm::ArrayRef<int64_t> dst) {
  if (llvm::any_of(src, [](int64_t d) { return d <= 0; }) ||
      llvm::any_of(dst, [](int64_t d) { return d <= 0; })) {
    return std::nullopt;
  }

  llvm::SmallVector<ReshapeGroup> groups;
  size_t i = 0, j = 0;
  while (i < src.size() && j < dst.size()) {
    ReshapeGroup group;
    int64_t srcProd = src[i];
    int64_t dstProd = dst[j];
    group.srcDims.push_back(i++);
    group.dstDims.push_back(j++);
    while (srcProd != dstProd) {
      if (srcProd < dstProd) {
        if (i >= src.size()) {
          return std::nullopt;
        }
        srcProd *= src[i];
        group.srcDims.push_back(i++);
      } else {
        if (j >= dst.size()) {
          return std::nullopt;
        }
        dstProd *= dst[j];
        group.dstDims.push_back(j++);
      }
    }
    groups.push_back(std::move(group));
  }
  for (; i < src.size(); ++i) {
    if (src[i] != 1) {
      return std::nullopt;
    }
  }
  for (; j < dst.size(); ++j) {
    if (dst[j] != 1) {
      return std::nullopt;
    }
  }
  return groups;
}

enum class ReductionKind { Sum, Max };

/// Identify the reduction computed by a reduce / reduce_window \p body, which
/// must consist of a single Add or Max followed by the return terminator.
FailureOr<ReductionKind> getReductionKind(Operation *op, Region &body) {
  auto &bodyOps = body.front().getOperations();
  if (bodyOps.size() != 2) {
    return op->emitError() << "Unsupported reduction body in " << op->getName();
  }

  auto &reductionOp = bodyOps.front();
  if (llvm::isa<stablehlo::AddOp>(reductionOp)) {
    return ReductionKind::Sum;
  }
  if (llvm::isa<stablehlo::MaxOp>(reductionOp)) {
    return ReductionKind::Max;
  }
  return op->emitError() << "Unsupported reduction in " << op->getName() << ": "
                         << reductionOp.getName();
}

/// Reduction of \p reduceSize copies of \p in.
Distribution reduceDist(ReductionKind kind, int64_t reduceSize,
                        const Distribution &in) {
  switch (kind) {
  case ReductionKind::Sum:
    return sumOfIidDist(reduceSize, in);
  case ReductionKind::Max:
    return maxOfIidDist(in);
  }
  llvm_unreachable("Unknown reduction kind");
}

/// Reduction over the slices of a non-uniform reduced axis (one element drawn
/// from each slice), repeated \p multiplicity times for the other reduced
/// axes.
Distribution reduceOverSlices(ReductionKind kind,
                              llvm::ArrayRef<Distribution> slices,
                              int64_t multiplicity) {
  switch (kind) {
  case ReductionKind::Sum: {
    Distribution total = slices.front();
    for (const Distribution &slice : slices.drop_front()) {
      total = addDist(total, slice);
    }
    return sumOfIidDist(multiplicity, total);
  }
  case ReductionKind::Max: {
    const Distribution *argmax = &slices.front();
    double min = slices.front().min;
    double max = slices.front().max;
    for (const Distribution &slice : slices) {
      min = std::max(min, slice.min);
      max = std::max(max, slice.max);
      if (slice.mean > argmax->mean) {
        argmax = &slice;
      }
    }
    Distribution result = maxOfIidDist(*argmax);
    result.min = min;
    result.max = max;
    return result;
  }
  }
  llvm_unreachable("Unknown reduction kind");
}
} // namespace

/// Convert \p distribution to an array attribute that can be attached to an
/// operation.
std::array<Attribute, 4>
distributionToArrayAttr(Builder &builder, const Distribution &distribution) {
  return {builder.getF64FloatAttr(distribution.min),
          builder.getF64FloatAttr(distribution.max),
          builder.getF64FloatAttr(distribution.mean),
          builder.getF64FloatAttr(distribution.variance)};
}

//===----------------------------------------------------------------------===//
// DistributionAnalysis
//===----------------------------------------------------------------------===//

void DistributionAnalysis::registerDistributions(
    Operation *op, llvm::ArrayRef<TensorDistribution> dists) {
  assert(op->getNumResults() == dists.size() &&
         "Number of distributions does not match number of operation results.");

  for (const auto &[result, dist] : llvm::zip(op->getResults(), dists)) {
    this->distributionMap[result] = dist;
  }
  this->analyzedOperations.insert(op);
}

TensorDistribution DistributionAnalysis::finish(TensorDistribution dist,
                                                const Distribution &fallback,
                                                Type type,
                                                bool recomputeSummary) const {
  dist.normalizeAxes(type);
  if (!options.perAxis) {
    dist.makeUniform(fallback);
    return dist;
  }
  if (recomputeSummary) {
    dist.finalize(fallback);
  } else {
    dist.summary = fallback;
  }
  return dist;
}

TensorDistribution DistributionAnalysis::makeUniform(const Distribution &dist,
                                                     Type type) const {
  return TensorDistribution::uniform(dist, type);
}

LogicalResult DistributionAnalysis::visitConstantLikeOp(Operation *op,
                                                        Attribute valueAttr) {
  assert(op->getNumResults() == 1 &&
         "Constant-like operation should have exactly one result");

  auto type = op->getResult(0).getType();
  auto tensorType = llvm::dyn_cast<RankedTensorType>(type);
  if (!tensorType || !tensorType.getElementType().isFloat()) {
    // FIXME: For now, we only consider float tensors
    LLVM_DEBUG(llvm::dbgs()
               << "[visitConstantLikeOp] Unsupported constant type: " << type
               << "\n");
    return success();
  }

  // FIXME: should we traverse the whole constant, or only take a sample?
  // For large constants, this could be expensive.

  // Extract the constant value and update the distribution map
  if (auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(valueAttr)) {
    TensorDistribution dist;
    switch (tensorType.getElementType().getIntOrFloatBitWidth()) {
    case 32: {
      auto values = denseAttr.getValues<float>();
      dist = computeConstantTensorDistribution<float, decltype(values)>(
          values, tensorType, options.perAxis, denseAttr.isSplat());
      break;
    }
    case 64: {
      auto values = denseAttr.getValues<double>();
      dist = computeConstantTensorDistribution<double, decltype(values)>(
          values, tensorType, options.perAxis, denseAttr.isSplat());
      break;
    }
    default:
      return op->emitError()
             << "Unsupported element type for distribution analysis: "
             << tensorType.getElementType();
    }
    this->registerDistributions(op, {dist});
  }

  return success();
}

LogicalResult DistributionAnalysis::visitAddOp(stablehlo::AddOp addOp) {
  auto lhsDistOrFailure = getTensorDistribution(addOp.getLhs());
  auto rhsDistOrFailure = getTensorDistribution(addOp.getRhs());
  if (failed(lhsDistOrFailure) || failed(rhsDistOrFailure)) {
    // If we don't have distribution info for one of the inputs, we can't
    // compute the distribution for the output.
    return addOp.emitError()
           << "Missing distribution info for at least one of the inputs.";
  }

  // Premise: inputs are independent and are described by a normal distribution
  auto outDist = zipSlices(**lhsDistOrFailure, **rhsDistOrFailure,
                           [](const Distribution &a, const Distribution &b) {
                             return std::optional(addDist(a, b));
                           });
  this->registerDistributions(
      addOp, {finish(*outDist, outDist->summary, addOp.getType())});

  return success();
}

LogicalResult
DistributionAnalysis::visitSubtractOp(stablehlo::SubtractOp subOp) {
  auto lhsDistOrFailure = getTensorDistribution(subOp.getLhs());
  auto rhsDistOrFailure = getTensorDistribution(subOp.getRhs());
  if (failed(lhsDistOrFailure) || failed(rhsDistOrFailure)) {
    return subOp.emitError()
           << "Missing distribution info for at least one of the inputs.";
  }

  // Same premises as addition
  auto outDist = zipSlices(**lhsDistOrFailure, **rhsDistOrFailure,
                           [](const Distribution &a, const Distribution &b) {
                             return std::optional(subDist(a, b));
                           });
  this->registerDistributions(
      subOp, {finish(*outDist, outDist->summary, subOp.getType())});

  return success();
}

LogicalResult DistributionAnalysis::visitMulOp(stablehlo::MulOp mulOp) {
  // Normalization multiply of a decomposed batch normalization:
  //   (x - mean) * rsqrt(var + eps).
  // Instead of applying the abstract product transfer function, propagate the
  // running statistics stored in the graph: per channel c, x is described by
  // mean m_c and variance v_c, so the normalized activation has mean 0 and
  // variance v_c / (v_c + eps) (approximately 1). Min/max are computed per
  // channel from the analyzed range of x.
  auto bnOrFailure = matchBatchNormNormalize(mulOp);
  if (succeeded(bnOrFailure)) {
    auto inputDistOrFailure = getTensorDistribution(bnOrFailure->input);
    auto normalizedDistOrFailure =
        getTensorDistribution(bnOrFailure->normalized);
    auto scaleDistOrFailure = getTensorDistribution(bnOrFailure->scale);
    if (failed(inputDistOrFailure) || failed(normalizedDistOrFailure) ||
        failed(scaleDistOrFailure)) {
      return mulOp.emitError()
             << "Missing distribution info for batch-normalization input.";
    }
    const TensorDistribution &inputDist = **inputDistOrFailure;
    const TensorDistribution &normalizedDist = **normalizedDistOrFailure;
    const TensorDistribution &scaleDist = **scaleDistOrFailure;

    // Per-tensor result: mixture over channels of the running statistics.
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    double varianceSum = 0.0;
    for (auto [m, v] : llvm::zip(bnOrFailure->means, bnOrFailure->variances)) {
      double r = 1.0 / std::sqrt(v + bnOrFailure->epsilon);
      // r > 0, so the channel extrema come from the extrema of x.
      min = std::min(min, (inputDist.summary.min - m) * r);
      max = std::max(max, (inputDist.summary.max - m) * r);
      varianceSum += v * r * r;
    }
    Distribution fallback{.min = min,
                          .max = max,
                          .mean = 0.0,
                          .variance = varianceSum / static_cast<double>(
                                                        bnOrFailure->means.size())};

    // The channel axis is the (unique) axis along which the broadcast scale
    // differs. Along it, each slice c is a point mass r_c = 1/sqrt(v_c + eps)
    // from which the running variance is recovered, and the centered
    // activation slice gives the channel range.
    std::optional<unsigned> channelAxis;
    unsigned nonUniformAxes = 0;
    for (unsigned axis = 0; axis < scaleDist.axes.size(); ++axis) {
      if (!scaleDist.isUniform(axis)) {
        channelAxis = axis;
        ++nonUniformAxes;
      }
    }

    if (nonUniformAxes <= 1) {
      TensorDistribution outDist = makeUniform(fallback, mulOp.getType());
      if (channelAxis && *channelAxis < normalizedDist.axes.size()) {
        llvm::SmallVector<Distribution> slices;
        bool ok = true;
        for (int64_t c = 0; c < scaleDist.numSlices(*channelAxis); ++c) {
          double r = scaleDist.slice(*channelAxis, c).mean;
          if (!(r > 0.0) || !std::isfinite(r)) {
            ok = false;
            break;
          }
          double v = std::max(0.0, 1.0 / (r * r) - bnOrFailure->epsilon);
          const Distribution &centered =
              normalizedDist.slice(*channelAxis, c);
          slices.push_back(Distribution{.min = centered.min * r,
                                        .max = centered.max * r,
                                        .mean = 0.0,
                                        .variance = v * r * r});
        }
        if (ok) {
          outDist.axes[*channelAxis].slices = std::move(slices);
        }
      }
      this->registerDistributions(
          mulOp, {finish(outDist, fallback, mulOp.getType())});
      return success();
    }
    // Ambiguous channel axis: fall through to the generic product.
  }

  auto lhsDistOrFailure = getTensorDistribution(mulOp.getLhs());
  auto rhsDistOrFailure = getTensorDistribution(mulOp.getRhs());
  if (failed(lhsDistOrFailure) || failed(rhsDistOrFailure)) {
    return mulOp.emitError()
           << "Missing distribution info for at least one of the inputs.";
  }

  // Same premises as add
  auto outDist = zipSlices(**lhsDistOrFailure, **rhsDistOrFailure,
                           [](const Distribution &a, const Distribution &b) {
                             return std::optional(mulDist(a, b));
                           });
  this->registerDistributions(
      mulOp, {finish(*outDist, outDist->summary, mulOp.getType())});

  return success();
}

LogicalResult DistributionAnalysis::visitDivOp(stablehlo::DivOp divOp) {
  auto lhsDistOrFailure = getTensorDistribution(divOp.getLhs());
  auto rhsDistOrFailure = getTensorDistribution(divOp.getRhs());
  if (failed(lhsDistOrFailure) || failed(rhsDistOrFailure)) {
    return divOp.emitError()
           << "Missing distribution info for at least one of the inputs.";
  }

  const TensorDistribution &lhsDist = **lhsDistOrFailure;
  const TensorDistribution &rhsDist = **rhsDistOrFailure;

  // Softmax / L1-normalize: x / broadcast(sum(x)) with nonnegative x.
  // Dependence gives tight bounds that independent interval arithmetic cannot:
  // each output is in [0, 1], and the tensor mean is 1/N (N terms per sum).
  // Nothing more is known about the individual slices, so all axes are
  // uniform.
  auto numClassesOrFailure = matchSoftmax(divOp);
  if (succeeded(numClassesOrFailure) && lhsDist.summary.min >= 0.0) {
    double mean = 1.0 / static_cast<double>(*numClassesOrFailure);
    Distribution outputDist{
        .min = 0.0,
        .max = 1.0,
        .mean = mean,
        // Any value in [0, 1] with mean μ has variance at most μ(1 − μ).
        .variance = mean * (1.0 - mean)};
    this->registerDistributions(divOp,
                                {makeUniform(outputDist, divOp.getType())});
    return success();
  }

  // Assume both random variables are independent.
  DivError summaryError = DivError::RangeContainsZero;
  auto outDist = zipSlices(lhsDist, rhsDist,
                           [&](const Distribution &a, const Distribution &b) {
                             DivError error;
                             auto result = divDist(a, b, &error);
                             if (!result) {
                               summaryError = error;
                             }
                             return result;
                           });
  if (failed(outDist)) {
    switch (summaryError) {
    case DivError::RangeContainsZero:
      // If the divisor range includes 0, X/Y is unbounded (or undefined), so
      // we cannot produce finite min/max.
      return divOp.emitError()
             << "Division by a value whose range contains zero is not "
                "supported in distribution analysis";
    case DivError::ZeroMean:
      // Mean estimation divides by the mean, so reject it if it is zero.
      return divOp.emitError()
             << "Cannot handle division with zero divisor mean";
    }
  }

  this->registerDistributions(
      divOp, {finish(*outDist, outDist->summary, divOp.getType())});

  return success();
}

LogicalResult DistributionAnalysis::visitClamp(Operation *op, Value input,
                                               double lowerBound,
                                               double upperBound) {
  auto inputDistOrFailure = getTensorDistribution(input);
  if (failed(inputDistOrFailure)) {
    return op->emitError() << "Missing distribution info for input.";
  }

  if (lowerBound > upperBound) {
    return op->emitError() << "Clamp lower bound (" << lowerBound
                           << ") is greater than upper bound (" << upperBound
                           << ")";
  }

  // Model each slice as X ~ N(mu, sigma^2) and the output as the winsorized
  // variable Y = clamp(X, lo, hi) (see clampDist). Applying the Gaussian model
  // per slice, and mixing afterwards, is far more faithful than a single
  // Gaussian for the whole tensor when slices have different means (e.g.
  // per-channel biases followed by a ReLU).
  auto outDist = mapSlices(**inputDistOrFailure, [&](const Distribution &d) {
    return std::optional(clampDist(d, lowerBound, upperBound));
  });
  this->registerDistributions(
      op, {finish(*outDist, outDist->summary, op->getResult(0).getType())});

  return success();
}

LogicalResult DistributionAnalysis::visitMaxOp(stablehlo::MaxOp maxOp) {
  // A maximum against a constant is a clamp from below (e.g. ReLU when the
  // constant is 0).
  constexpr double inf = std::numeric_limits<double>::infinity();
  auto lhsConstOrFailure = getScalarConstantValue(maxOp.getLhs());
  auto rhsConstOrFailure = getScalarConstantValue(maxOp.getRhs());

  if (succeeded(rhsConstOrFailure)) {
    return visitClamp(maxOp, maxOp.getLhs(), *rhsConstOrFailure, inf);
  }

  if (succeeded(lhsConstOrFailure)) {
    return visitClamp(maxOp, maxOp.getRhs(), *lhsConstOrFailure, inf);
  }

  return maxOp.emitError()
         << "Expected one of the operands of stablehlo.maximum to be a "
            "scalar or splat float constant";
}

LogicalResult DistributionAnalysis::visitClampOp(stablehlo::ClampOp clampOp) {
  auto lowerOrFailure = getScalarConstantValue(clampOp.getMin());
  if (failed(lowerOrFailure)) {
    return clampOp.emitError()
           << "Expected the min operand of stablehlo.clamp to be a scalar or "
              "splat float constant";
  }

  auto upperOrFailure = getScalarConstantValue(clampOp.getMax());
  if (failed(upperOrFailure)) {
    return clampOp.emitError()
           << "Expected the max operand of stablehlo.clamp to be a scalar or "
              "splat float constant";
  }

  return visitClamp(clampOp, clampOp.getOperand(), *lowerOrFailure,
                    *upperOrFailure);
}

LogicalResult DistributionAnalysis::visitUnaryIdentityOp(Operation *op) {
  assert(op->getNumOperands() >= 1 && op->getNumResults() == 1 &&
         "Expected unary operation with at least one operand and exactly one "
         "result");

  auto inputDistOrFailure = getTensorDistribution(op->getOperand(0));
  if (failed(inputDistOrFailure)) {
    return op->emitError() << "Missing distribution info for input.";
  }

  // The per-tensor distribution is preserved (or at least approximated by the
  // operand's), but the per-axis structure is unknown.
  this->registerDistributions(
      op, {makeUniform((*inputDistOrFailure)->summary,
                       op->getResult(0).getType())});

  return success();
}

LogicalResult
DistributionAnalysis::visitBroadcastInDim(stablehlo::BroadcastInDimOp bcastOp) {
  auto inputDistOrFailure = getTensorDistribution(bcastOp.getOperand());
  if (failed(inputDistOrFailure)) {
    return bcastOp.emitError() << "Missing distribution info for input.";
  }
  const TensorDistribution &inputDist = **inputDistOrFailure;

  // Broadcasting replicates every element the same number of times, so the
  // per-tensor distribution is unchanged. Input axis i becomes output axis
  // broadcast_dimensions[i] (unless it is expanded from size 1); new output
  // axes are uniform.
  TensorDistribution outDist =
      makeUniform(inputDist.summary, bcastOp.getType());
  if (auto outShape = getStaticShape(bcastOp.getType())) {
    for (auto [inAxis, outAxis] :
         llvm::enumerate(bcastOp.getBroadcastDimensions())) {
      if (inAxis < inputDist.axes.size() && !inputDist.isUniform(inAxis) &&
          inputDist.numSlices(inAxis) == (*outShape)[outAxis]) {
        outDist.axes[outAxis] = inputDist.axes[inAxis];
      }
    }
  }

  this->registerDistributions(
      bcastOp, {finish(outDist, inputDist.summary, bcastOp.getType(),
                       /*recomputeSummary=*/false)});
  return success();
}

LogicalResult
DistributionAnalysis::visitTranspose(stablehlo::TransposeOp transposeOp) {
  auto inputDistOrFailure = getTensorDistribution(transposeOp.getOperand());
  if (failed(inputDistOrFailure)) {
    return transposeOp.emitError() << "Missing distribution info for input.";
  }
  const TensorDistribution &inputDist = **inputDistOrFailure;

  // A transpose only relabels axes: output axis i is input axis
  // permutation[i].
  TensorDistribution outDist =
      makeUniform(inputDist.summary, transposeOp.getType());
  auto permutation = transposeOp.getPermutation();
  if (permutation.size() == inputDist.axes.size() &&
      permutation.size() == outDist.axes.size()) {
    for (auto [outAxis, inAxis] : llvm::enumerate(permutation)) {
      outDist.axes[outAxis] = inputDist.axes[inAxis];
    }
  }

  this->registerDistributions(
      transposeOp, {finish(outDist, inputDist.summary, transposeOp.getType(),
                           /*recomputeSummary=*/false)});
  return success();
}

LogicalResult
DistributionAnalysis::visitReshape(stablehlo::ReshapeOp reshapeOp) {
  auto inputDistOrFailure = getTensorDistribution(reshapeOp.getOperand());
  if (failed(inputDistOrFailure)) {
    return reshapeOp.emitError() << "Missing distribution info for input.";
  }
  const TensorDistribution &inputDist = **inputDistOrFailure;

  // A reshape does not change the population of values, only the summary is
  // trivially preserved. Per-axis information is mapped through the groups of
  // axes that are split or merged:
  //  - 1 -> 1: the axis is relabeled.
  //  - 1 -> n (split): every output slice is the exact mixture of the input
  //    slices that fold into it.
  //  - n -> 1 (merge): the marginals of the merged axes do not determine the
  //    joint slices. If a single merged axis carries information, its slices
  //    are inherited (exact when the other axes are truly uniform within each
  //    slice); otherwise the output axis is left uniform.
  TensorDistribution outDist =
      makeUniform(inputDist.summary, reshapeOp.getType());
  auto srcShape = getStaticShape(reshapeOp.getOperand().getType());
  auto dstShape = getStaticShape(reshapeOp.getType());
  std::optional<llvm::SmallVector<ReshapeGroup>> groups;
  if (srcShape && dstShape && srcShape->size() == inputDist.axes.size()) {
    groups = computeReshapeGroups(*srcShape, *dstShape);
  }

  for (const ReshapeGroup &group : groups.value_or(llvm::SmallVector<ReshapeGroup>{})) {
    if (group.srcDims.size() == 1 && group.dstDims.size() == 1) {
      outDist.axes[group.dstDims[0]] = inputDist.axes[group.srcDims[0]];
      continue;
    }

    if (group.srcDims.size() == 1) {
      // Split: source axis a of size n into dstDims (row-major).
      unsigned a = group.srcDims[0];
      if (inputDist.isUniform(a)) {
        continue;
      }
      const auto &srcSlices = inputDist.axes[a].slices;
      int64_t n = srcSlices.size();
      for (auto [pos, d] : llvm::enumerate(group.dstDims)) {
        int64_t size = (*dstShape)[d];
        if (size <= 1) {
          continue;
        }
        int64_t stride = 1;
        for (unsigned later : llvm::ArrayRef(group.dstDims).drop_front(pos + 1)) {
          stride *= (*dstShape)[later];
        }
        llvm::SmallVector<Distribution> dstSlices;
        dstSlices.reserve(size);
        for (int64_t t = 0; t < size; ++t) {
          llvm::SmallVector<Distribution> members;
          for (int64_t i = 0; i < n; ++i) {
            if ((i / stride) % size == t) {
              members.push_back(srcSlices[i]);
            }
          }
          dstSlices.push_back(mixtureDistribution(members));
        }
        outDist.axes[d].slices = std::move(dstSlices);
      }
      continue;
    }

    if (group.dstDims.size() == 1) {
      // Merge: srcDims (row-major) into destination axis d.
      unsigned d = group.dstDims[0];
      std::optional<unsigned> informative;
      unsigned nonUniform = 0;
      for (unsigned s : group.srcDims) {
        if (!inputDist.isUniform(s)) {
          informative = s;
          ++nonUniform;
        }
      }
      if (nonUniform != 1) {
        // FIXME: Combine marginals when several merged axes are informative.
        continue;
      }
      int64_t stride = 1;
      bool after = false;
      for (unsigned s : group.srcDims) {
        if (after) {
          stride *= (*srcShape)[s];
        }
        if (s == *informative) {
          after = true;
        }
      }
      const auto &srcSlices = inputDist.axes[*informative].slices;
      int64_t n = srcSlices.size();
      int64_t size = (*dstShape)[d];
      llvm::SmallVector<Distribution> dstSlices;
      dstSlices.reserve(size);
      for (int64_t i = 0; i < size; ++i) {
        dstSlices.push_back(srcSlices[(i / stride) % n]);
      }
      outDist.axes[d].slices = std::move(dstSlices);
      continue;
    }
    // n -> m with n, m > 1: no simple correspondence, leave uniform.
  }

  this->registerDistributions(
      reshapeOp, {finish(outDist, inputDist.summary, reshapeOp.getType(),
                         /*recomputeSummary=*/false)});
  return success();
}

LogicalResult DistributionAnalysis::visitSlice(stablehlo::SliceOp sliceOp) {
  auto inputDistOrFailure = getTensorDistribution(sliceOp.getOperand());
  if (failed(inputDistOrFailure)) {
    return sliceOp.emitError() << "Missing distribution info for input.";
  }
  const TensorDistribution &inputDist = **inputDistOrFailure;

  // Along every axis, the output slices are a strided selection of the input
  // slices. Each kept slice is also restricted along the other axes, so its
  // input distribution is only an approximation (sound for min/max).
  TensorDistribution outDist =
      makeUniform(inputDist.summary, sliceOp.getType());
  auto outShape = getStaticShape(sliceOp.getType());
  if (outShape && outShape->size() == inputDist.axes.size()) {
    auto starts = sliceOp.getStartIndices();
    auto strides = sliceOp.getStrides();
    for (unsigned axis = 0; axis < outShape->size(); ++axis) {
      if (inputDist.isUniform(axis)) {
        continue;
      }
      int64_t n = inputDist.numSlices(axis);
      llvm::SmallVector<Distribution> slices;
      bool ok = true;
      for (int64_t k = 0; k < (*outShape)[axis]; ++k) {
        int64_t index = starts[axis] + k * strides[axis];
        if (index < 0 || index >= n) {
          ok = false;
          break;
        }
        slices.push_back(inputDist.slice(axis, index));
      }
      if (ok) {
        outDist.axes[axis].slices = std::move(slices);
      }
    }
  }

  this->registerDistributions(
      sliceOp, {finish(outDist, inputDist.summary, sliceOp.getType())});
  return success();
}

LogicalResult
DistributionAnalysis::visitDotGeneral(stablehlo::DotGeneralOp dotOp) {
  auto lhsDistOrFailure = getTensorDistribution(dotOp.getLhs());
  auto rhsDistOrFailure = getTensorDistribution(dotOp.getRhs());
  if (failed(lhsDistOrFailure) || failed(rhsDistOrFailure)) {
    // If we don't have distribution info for one of the inputs, we can't
    // compute the distribution for the output.
    return dotOp.emitError()
           << "Missing distribution info for at least one of the inputs.";
  }

  const TensorDistribution &lhsDist = **lhsDistOrFailure;
  const TensorDistribution &rhsDist = **rhsDistOrFailure;

  // The contraction size is the number of elements that are multiplied and
  // accumulated for each element of the output.
  auto lhsShape = llvm::cast<ShapedType>(dotOp.getLhs().getType()).getShape();
  auto rhsShape = llvm::cast<ShapedType>(dotOp.getRhs().getType()).getShape();
  auto dimNumbers = dotOp.getDotDimensionNumbers();
  auto lhsBatch = dimNumbers.getLhsBatchingDimensions();
  auto rhsBatch = dimNumbers.getRhsBatchingDimensions();
  auto lhsContract = dimNumbers.getLhsContractingDimensions();
  auto rhsContract = dimNumbers.getRhsContractingDimensions();
  int64_t contractionSize = 1;
  for (auto dim : lhsContract) {
    if (lhsShape[dim] == ShapedType::kDynamic) {
      return dotOp.emitError()
             << "Dynamic contracting dimensions are not supported";
    }
    contractionSize *= lhsShape[dim];
  }

  // Per-tensor result. Pairing the slices along an informative contracting
  // axis (e.g. per-input-channel activations with per-input-channel weights)
  // gives the exact mean under independence: sum_k mu_lhs[k] * mu_rhs[k].
  Distribution naive =
      contractionDist(contractionSize, lhsDist.summary, rhsDist.summary);
  std::optional<Distribution> paired = pairedContraction(
      lhsDist, rhsDist, lhsContract, rhsContract, contractionSize);
  Distribution fallback = paired.value_or(naive);

  // The output slices below only see one operand's free-axis marginal and the
  // other operand's contracting-axis marginals, so their means miss the
  // pairing term. Under an additive model (element ~ free-axis effect +
  // contracting-axis effect) that term is the same for every slice: the
  // difference between the paired mean and the mean obtained with the
  // operand's summary. Shifting the slices by it makes their mixture agree
  // with the paired mean while keeping the between-slice structure.
  double batchShift = paired ? paired->mean - naive.mean : 0.0;
  double lhsFreeShift =
      paired ? paired->mean - contractionOverAxis(lhsDist.summary, rhsDist,
                                                  rhsContract, contractionSize)
                                  .mean
             : 0.0;
  double rhsFreeShift =
      paired ? paired->mean - contractionOverAxis(rhsDist.summary, lhsDist,
                                                  lhsContract, contractionSize)
                                  .mean
             : 0.0;

  TensorDistribution outDist = makeUniform(fallback, dotOp.getType());

  // Output axes are [batch..., lhs free..., rhs free...].
  auto outShape = getStaticShape(dotOp.getType());
  if (outShape && lhsDist.axes.size() == lhsShape.size() &&
      rhsDist.axes.size() == rhsShape.size()) {
    llvm::SmallVector<int64_t> lhsFree, rhsFree;
    for (int64_t axis = 0; axis < static_cast<int64_t>(lhsShape.size());
         ++axis) {
      if (!llvm::is_contained(lhsBatch, axis) &&
          !llvm::is_contained(lhsContract, axis)) {
        lhsFree.push_back(axis);
      }
    }
    for (int64_t axis = 0; axis < static_cast<int64_t>(rhsShape.size());
         ++axis) {
      if (!llvm::is_contained(rhsBatch, axis) &&
          !llvm::is_contained(rhsContract, axis)) {
        rhsFree.push_back(axis);
      }
    }

    if (lhsBatch.size() + lhsFree.size() + rhsFree.size() == outShape->size()) {
      unsigned outAxis = 0;
      // Batch axes: slice b of the output is the contraction of slice b of
      // both operands.
      for (auto [lb, rb] : llvm::zip(lhsBatch, rhsBatch)) {
        int64_t n = (*outShape)[outAxis];
        if ((!lhsDist.isUniform(lb) || !rhsDist.isUniform(rb)) &&
            (lhsDist.isUniform(lb) || lhsDist.numSlices(lb) == n) &&
            (rhsDist.isUniform(rb) || rhsDist.numSlices(rb) == n)) {
          auto &slices = outDist.axes[outAxis].slices;
          for (int64_t b = 0; b < n; ++b) {
            slices.push_back(shiftDist(
                contractionDist(contractionSize, lhsDist.slice(lb, b),
                                rhsDist.slice(rb, b)),
                batchShift));
          }
        }
        ++outAxis;
      }
      // Free axes: slice m only involves slice m of its operand, contracted
      // against the whole other operand.
      for (int64_t la : lhsFree) {
        int64_t n = (*outShape)[outAxis];
        if (!lhsDist.isUniform(la) && lhsDist.numSlices(la) == n) {
          auto &slices = outDist.axes[outAxis].slices;
          for (int64_t m = 0; m < n; ++m) {
            slices.push_back(shiftDist(
                contractionOverAxis(lhsDist.slice(la, m), rhsDist, rhsContract,
                                    contractionSize),
                lhsFreeShift));
          }
        }
        ++outAxis;
      }
      for (int64_t ra : rhsFree) {
        int64_t n = (*outShape)[outAxis];
        if (!rhsDist.isUniform(ra) && rhsDist.numSlices(ra) == n) {
          auto &slices = outDist.axes[outAxis].slices;
          for (int64_t m = 0; m < n; ++m) {
            slices.push_back(shiftDist(
                contractionOverAxis(rhsDist.slice(ra, m), lhsDist, lhsContract,
                                    contractionSize),
                rhsFreeShift));
          }
        }
        ++outAxis;
      }
    }
  }

  this->registerDistributions(dotOp,
                              {finish(outDist, fallback, dotOp.getType())});

  return success();
}

LogicalResult
DistributionAnalysis::visitConvolution(stablehlo::ConvolutionOp convOp) {
  auto inputDistOrFailure = getTensorDistribution(convOp.getLhs());
  auto kernelDistOrFailure = getTensorDistribution(convOp.getRhs());
  if (failed(inputDistOrFailure) || failed(kernelDistOrFailure)) {
    return convOp.emitError()
           << "Missing distribution info for at least one of the inputs.";
  }
  const TensorDistribution &inputDist = **inputDistOrFailure;
  const TensorDistribution &kernelDist = **kernelDistOrFailure;

  auto inputShape =
      llvm::cast<ShapedType>(convOp.getLhs().getType()).getShape();
  auto kernelShape =
      llvm::cast<ShapedType>(convOp.getRhs().getType()).getShape();

  // The dimension numbers make this layout-agnostic (i.e. not restricted to
  // NCHW/FCHW convolutions).
  auto dimNumbers = convOp.getDimensionNumbers();
  int64_t inputBatchDim = dimNumbers.getInputBatchDimension();
  int64_t inputFeatureDim = dimNumbers.getInputFeatureDimension();
  int64_t kernelInputFeatureDim = dimNumbers.getKernelInputFeatureDimension();
  int64_t kernelOutputFeatureDim =
      dimNumbers.getKernelOutputFeatureDimension();
  int64_t outputBatchDim = dimNumbers.getOutputBatchDimension();
  int64_t outputFeatureDim = dimNumbers.getOutputFeatureDimension();
  int64_t groups = convOp.getFeatureGroupCount();
  int64_t inputChannels = inputShape[inputFeatureDim];
  int64_t channelsPerGroup = inputChannels / groups;

  // This is the size of the number of elements in the "sliding window" of
  // convolution
  int64_t spatialSize = 1;
  for (auto dim : dimNumbers.getKernelSpatialDimensions()) {
    spatialSize *= kernelShape[dim];
  }
  int64_t contractionSize = channelsPerGroup * spatialSize;

  bool perSliceOk = convOp.getBatchGroupCount() == 1 &&
                    inputDist.axes.size() == inputShape.size() &&
                    kernelDist.axes.size() == kernelShape.size() &&
                    llvm::none_of(inputShape, ShapedType::isDynamic) &&
                    llvm::none_of(kernelShape, ShapedType::isDynamic) &&
                    inputChannels % groups == 0;

  // Sum of the products of input channel c with the kernel, repeated for every
  // spatial tap of the window. \p kernelDistForChannels gives the kernel
  // distribution to pair with each channel (per-channel when the kernel is
  // informative along its input-feature axis, otherwise a fixed one).
  auto contractChannels = [&](int64_t firstChannel, int64_t numChannels,
                              auto kernelForChannel) {
    double mean = 0.0;
    double variance = 0.0;
    for (int64_t c = firstChannel; c < firstChannel + numChannels; ++c) {
      Distribution product =
          mulDist(inputDist.slice(inputFeatureDim, c), kernelForChannel(c));
      mean += product.mean;
      variance += product.variance;
    }
    return accumulationDist(mean * spatialSize, variance * spatialSize);
  };

  // Per-tensor result. As in dot_general, pairing the input channels with the
  // matching kernel input-feature slices gives the exact mean under
  // independence; the output slices below are shifted so that their mixture
  // agrees with it (additive model, see visitDotGeneral).
  Distribution naive = contractionDist(contractionSize, inputDist.summary,
                                       kernelDist.summary);
  // Input channels paired with the kernel summary.
  std::optional<Distribution> channelPaired;
  // Input channels paired with the kernel's per-input-channel slices.
  std::optional<Distribution> fullPaired;
  if (perSliceOk && groups == 1 && !inputDist.isUniform(inputFeatureDim)) {
    channelPaired = contractChannels(
        0, inputChannels,
        [&](int64_t) -> const Distribution & { return kernelDist.summary; });
    if (!kernelDist.isUniform(kernelInputFeatureDim) &&
        kernelDist.numSlices(kernelInputFeatureDim) == inputChannels) {
      fullPaired = contractChannels(0, inputChannels, [&](int64_t c) {
        return kernelDist.slice(kernelInputFeatureDim, c);
      });
    }
  }
  std::optional<Distribution> paired = fullPaired ? fullPaired : channelPaired;
  Distribution fallback = paired.value_or(naive);
  double batchShift = paired ? paired->mean - naive.mean : 0.0;
  double featureShift =
      fullPaired ? fullPaired->mean - channelPaired->mean : 0.0;

  TensorDistribution outDist = makeUniform(fallback, convOp.getType());
  auto outShape = getStaticShape(convOp.getType());
  if (perSliceOk && outShape) {
    // Batch: every output image depends only on the same input image.
    if (!inputDist.isUniform(inputBatchDim) &&
        inputDist.numSlices(inputBatchDim) == (*outShape)[outputBatchDim]) {
      auto &slices = outDist.axes[outputBatchDim].slices;
      for (const Distribution &image : inputDist.axes[inputBatchDim].slices) {
        slices.push_back(shiftDist(
            contractionDist(contractionSize, image, kernelDist.summary),
            batchShift));
      }
    }

    // Output features: channel o uses kernel slice o and, for grouped
    // convolutions, only the input channels of its group.
    int64_t outputChannels = (*outShape)[outputFeatureDim];
    if (outputChannels == kernelShape[kernelOutputFeatureDim] &&
        outputChannels % groups == 0 &&
        (!kernelDist.isUniform(kernelOutputFeatureDim) ||
         !inputDist.isUniform(inputFeatureDim))) {
      int64_t outputChannelsPerGroup = outputChannels / groups;
      auto &slices = outDist.axes[outputFeatureDim].slices;
      for (int64_t o = 0; o < outputChannels; ++o) {
        const Distribution &weights = kernelDist.slice(kernelOutputFeatureDim, o);
        if (!inputDist.isUniform(inputFeatureDim)) {
          int64_t group = o / outputChannelsPerGroup;
          slices.push_back(shiftDist(
              contractChannels(
                  group * channelsPerGroup, channelsPerGroup,
                  [&](int64_t) -> const Distribution & { return weights; }),
              groups == 1 ? featureShift : 0.0));
        } else {
          slices.push_back(
              contractionDist(contractionSize, inputDist.summary, weights));
        }
      }
    }
    // Spatial axes: windows mix neighbouring positions; left uniform.
  }

  this->registerDistributions(convOp,
                              {finish(outDist, fallback, convOp.getType())});

  return success();
}

LogicalResult DistributionAnalysis::visitReduceWindow(
    stablehlo::ReduceWindowOp reduceWindowOp) {
  if (reduceWindowOp.getInputs().size() != 1) {
    return reduceWindowOp.emitError()
           << "Expected stablehlo.reduce_window to have exactly 1 input";
  }

  auto input = reduceWindowOp.getInputs().front();
  auto inputDistOrFailure = getTensorDistribution(input);
  if (failed(inputDistOrFailure)) {
    return reduceWindowOp.emitError() << "Missing distribution info for input.";
  }
  const TensorDistribution &inputDist = **inputDistOrFailure;

  auto kindOrFailure =
      getReductionKind(reduceWindowOp, reduceWindowOp.getBody());
  if (failed(kindOrFailure)) {
    return failure();
  }

  auto windowDims = reduceWindowOp.getWindowDimensions();
  int64_t windowSize = productOf(windowDims);
  Distribution fallback = reduceDist(*kindOrFailure, windowSize, inputDist.summary);

  Type resultType = reduceWindowOp->getResult(0).getType();
  TensorDistribution outDist = makeUniform(fallback, resultType);

  // Axes that the window does not move along (size 1, stride 1, no padding or
  // dilation) map slice to slice: every window lies within one slice.
  auto inShape = getStaticShape(input.getType());
  auto outShape = getStaticShape(resultType);
  if (inShape && outShape && inShape->size() == outShape->size() &&
      inShape->size() == inputDist.axes.size() &&
      windowDims.size() == inShape->size()) {
    auto strides = reduceWindowOp.getWindowStrides();
    auto baseDilations = reduceWindowOp.getBaseDilations();
    auto windowDilations = reduceWindowOp.getWindowDilations();
    auto padding = reduceWindowOp.getPadding();
    llvm::SmallVector<int64_t> paddingValues;
    if (padding) {
      llvm::append_range(paddingValues, padding->getValues<int64_t>());
    }

    for (unsigned axis = 0; axis < inShape->size(); ++axis) {
      bool passthrough =
          windowDims[axis] == 1 && (!strides || (*strides)[axis] == 1) &&
          (!baseDilations || (*baseDilations)[axis] == 1) &&
          (!windowDilations || (*windowDilations)[axis] == 1) &&
          (paddingValues.empty() ||
           (paddingValues.size() > 2 * axis + 1 &&
            paddingValues[2 * axis] == 0 && paddingValues[2 * axis + 1] == 0)) &&
          (*inShape)[axis] == (*outShape)[axis];
      if (!passthrough || inputDist.isUniform(axis)) {
        continue;
      }
      auto &slices = outDist.axes[axis].slices;
      for (const Distribution &slice : inputDist.axes[axis].slices) {
        slices.push_back(reduceDist(*kindOrFailure, windowSize, slice));
      }
    }
  }

  this->registerDistributions(reduceWindowOp,
                              {finish(outDist, fallback, resultType)});
  return success();
}

LogicalResult DistributionAnalysis::visitReduce(stablehlo::ReduceOp reduceOp) {
  if (reduceOp.getInputs().size() != 1) {
    return reduceOp.emitError()
           << "Expected stablehlo.reduce to have exactly 1 input";
  }

  auto input = reduceOp.getInputs().front();
  auto inputDistOrFailure = getTensorDistribution(input);
  if (failed(inputDistOrFailure)) {
    return reduceOp.emitError() << "Missing distribution info for input.";
  }
  const TensorDistribution &inputDist = **inputDistOrFailure;

  auto inputShape = llvm::cast<ShapedType>(input.getType()).getShape();
  auto dims = reduceOp.getDimensions();
  int64_t reduceSize = 1;
  for (int64_t dim : dims) {
    if (inputShape[dim] == ShapedType::kDynamic) {
      return reduceOp.emitError()
             << "Dynamic reduced dimensions are not supported";
    }
    reduceSize *= inputShape[dim];
  }

  auto kindOrFailure = getReductionKind(reduceOp, reduceOp.getBody());
  if (failed(kindOrFailure)) {
    return failure();
  }
  ReductionKind kind = *kindOrFailure;

  Distribution fallback = reduceDist(kind, reduceSize, inputDist.summary);
  Type resultType = reduceOp->getResult(0).getType();
  TensorDistribution outDist = makeUniform(fallback, resultType);

  std::optional<Distribution> summaryOverride;
  if (inputDist.axes.size() == inputShape.size()) {
    // Kept axes: each output slice is the reduction of the matching input
    // slice.
    unsigned outAxis = 0;
    for (unsigned axis = 0; axis < inputShape.size(); ++axis) {
      if (llvm::is_contained(dims, static_cast<int64_t>(axis))) {
        continue;
      }
      if (outAxis < outDist.axes.size() && !inputDist.isUniform(axis)) {
        auto &slices = outDist.axes[outAxis].slices;
        for (const Distribution &slice : inputDist.axes[axis].slices) {
          slices.push_back(reduceDist(kind, reduceSize, slice));
        }
      }
      ++outAxis;
    }

    // Reduced axes: if one of them is informative, every output element
    // combines exactly one element of each of its slices, which is more
    // precise than n copies of the summary.
    std::optional<unsigned> best;
    double bestScore = -1.0;
    for (int64_t dim : dims) {
      if (inputDist.isUniform(dim)) {
        continue;
      }
      double score = inputDist.betweenSliceVariance(dim);
      if (!best || score > bestScore) {
        best = dim;
        bestScore = score;
      }
    }
    if (best) {
      int64_t multiplicity = reduceSize / inputDist.numSlices(*best);
      summaryOverride =
          reduceOverSlices(kind, inputDist.axes[*best].slices, multiplicity);
    }
  }

  outDist = finish(outDist, fallback, resultType);
  if (summaryOverride && options.perAxis) {
    outDist.summary = *summaryOverride;
  }
  this->registerDistributions(reduceOp, {outDist});

  return success();
}

LogicalResult DistributionAnalysis::visitPad(stablehlo::PadOp padOp) {
  auto source = padOp.getOperand();
  auto sourceDistOrFailure = getTensorDistribution(source);
  if (failed(sourceDistOrFailure)) {
    return padOp.emitError() << "Missing distribution info for source.";
  }
  const TensorDistribution &sourceDist = **sourceDistOrFailure;

  auto padValueOrFailure = getScalarConstantValue(padOp.getPaddingValue());
  if (failed(padValueOrFailure)) {
    return padOp.emitError()
           << "Expected pad value to be a scalar or splat float constant";
  }
  double padValue = *padValueOrFailure;

  if (!std::isfinite(padValue)) {
    padOp.emitWarning() << "Pad value is NaN/Inf. Treating pad as identity "
                           "for distribution analysis.";
    this->registerDistributions(
        padOp, {finish(sourceDist, sourceDist.summary, padOp.getType(),
                       /*recomputeSummary=*/false)});
    return success();
  }

  auto getTensorSize = [](llvm::ArrayRef<int64_t> shape) -> size_t {
    size_t size = 1;
    for (auto dim : shape) {
      // FIXME: Dynamic shapes are being ignored, but this is not ideal. How
      // should we handle this?
      if (dim != ShapedType::kDynamic) {
        size *= dim;
      }
    }

    return size;
  };

  // Compute size of source tensor
  auto sourceShape = llvm::cast<ShapedType>(source.getType()).getShape();
  size_t tensorSize = getTensorSize(sourceShape);
  // Compute size of padding. This accounts for both edge and interior
  // padding, since both are reflected in the result shape.
  auto outShape =
      llvm::cast<ShapedType>(padOp.getResult().getType()).getShape();
  size_t outSize = getTensorSize(outShape);
  size_t padSize = outSize - tensorSize;

  auto newSize = tensorSize + padSize;
  double originalRatio = static_cast<double>(tensorSize) / newSize;
  double padRatio = static_cast<double>(padSize) / newSize;
  Distribution fallback =
      padMixDist(sourceDist.summary, padValue, originalRatio, padRatio);

  TensorDistribution outDist = makeUniform(fallback, padOp.getType());

  // Along an axis, padded indices are point masses at the pad value, and the
  // original indices are the source slices mixed with the padding introduced
  // along the *other* axes.
  auto low = padOp.getEdgePaddingLow();
  auto high = padOp.getEdgePaddingHigh();
  auto interior = padOp.getInteriorPadding();
  bool staticShapes = llvm::none_of(sourceShape, ShapedType::isDynamic) &&
                      llvm::none_of(outShape, ShapedType::isDynamic);
  bool nonNegative = llvm::all_of(low, [](int64_t p) { return p >= 0; }) &&
                     llvm::all_of(high, [](int64_t p) { return p >= 0; });
  if (staticShapes && nonNegative &&
      sourceDist.axes.size() == sourceShape.size()) {
    for (unsigned axis = 0; axis < sourceShape.size(); ++axis) {
      int64_t srcDim = sourceShape[axis];
      int64_t outDim = outShape[axis];
      bool axisPadded = low[axis] > 0 || high[axis] > 0 ||
                        (interior[axis] > 0 && srcDim > 1);
      if (outDim <= 1 || (!axisPadded && sourceDist.isUniform(axis))) {
        continue;
      }

      double srcOthers = 1.0;
      double outOthers = 1.0;
      for (unsigned other = 0; other < sourceShape.size(); ++other) {
        if (other != axis) {
          srcOthers *= static_cast<double>(sourceShape[other]);
          outOthers *= static_cast<double>(outShape[other]);
        }
      }
      double sliceOriginalRatio = srcOthers / outOthers;
      double slicePadRatio = (outOthers - srcOthers) / outOthers;

      llvm::SmallVector<Distribution> slices(outDim, pointMassDist(padValue));
      for (int64_t i = 0; i < srcDim; ++i) {
        int64_t pos = low[axis] + i * (interior[axis] + 1);
        if (pos >= 0 && pos < outDim) {
          slices[pos] = padMixDist(sourceDist.slice(axis, i), padValue,
                                   sliceOriginalRatio, slicePadRatio);
        }
      }
      outDist.axes[axis].slices = std::move(slices);
    }
  }

  // The per-tensor mixture above is exact given the source summary, so there
  // is nothing to gain from recomputing it from the slices.
  this->registerDistributions(
      padOp, {finish(outDist, fallback, padOp.getType(),
                     /*recomputeSummary=*/false)});

  return success();
}

LogicalResult
DistributionAnalysis::visitConcatenate(stablehlo::ConcatenateOp concatOp) {
  auto axis = concatOp.getDimension();

  llvm::SmallVector<uint64_t> dims;
  auto inputs = concatOp.getInputs();
  uint64_t newDimSize = 0;
  // Compute size of concatenated dimension
  for (auto input : inputs) {
    // FIXME: What to do with dynamic shapes?
    dims.push_back(llvm::cast<ShapedType>(input.getType()).getDimSize(axis));
    newDimSize += dims.back();
  }

  llvm::SmallVector<const TensorDistribution *> dists;
  llvm::SmallVector<Distribution> summaries;
  llvm::SmallVector<double> weights;
  for (const auto [input, dim] : llvm::zip(inputs, dims)) {
    auto inputDistOrFailure = getTensorDistribution(input);
    if (failed(inputDistOrFailure)) {
      return concatOp.emitError()
             << "Missing distribution info for at least one of the inputs.";
    }
    dists.push_back(*inputDistOrFailure);
    summaries.push_back((*inputDistOrFailure)->summary);
    weights.push_back(static_cast<double>(dim) / newDimSize);
  }

  // Per-tensor result: weighted mixture of the inputs.
  Distribution fallback = mixtureDistribution(summaries, weights);
  TensorDistribution outDist = makeUniform(fallback, concatOp.getType());

  auto outShape = getStaticShape(concatOp.getType());
  bool ranksMatch = outShape && llvm::all_of(dists, [&](const auto *d) {
                      return d->axes.size() == outShape->size();
                    });
  if (ranksMatch && !llvm::is_contained(dims, ShapedType::kDynamic)) {
    // Concatenation axis: the slices of the inputs are simply appended.
    if (newDimSize > 1) {
      auto &slices = outDist.axes[axis].slices;
      for (auto [dist, dim] : llvm::zip(dists, dims)) {
        if (dist->isUniform(axis)) {
          slices.append(dim, dist->summary);
        } else {
          slices.append(dist->axes[axis].slices);
        }
      }
    }
    // Other axes: slice i of the output is the mixture of slice i of every
    // input, weighted by their extents along the concatenation axis.
    for (unsigned other = 0; other < outShape->size(); ++other) {
      if (other == axis || llvm::all_of(dists, [&](const auto *d) {
            return d->isUniform(other);
          })) {
        continue;
      }
      int64_t n = (*outShape)[other];
      if (llvm::any_of(dists, [&](const auto *d) {
            return !d->isUniform(other) && d->numSlices(other) != n;
          })) {
        continue;
      }
      auto &slices = outDist.axes[other].slices;
      for (int64_t i = 0; i < n; ++i) {
        llvm::SmallVector<Distribution> members;
        for (const auto *dist : dists) {
          members.push_back(dist->slice(other, i));
        }
        slices.push_back(mixtureDistribution(members, weights));
      }
    }
  }

  // The weighted mixture of the input summaries is exact, so keep it rather
  // than recomputing it from the slices.
  this->registerDistributions(
      concatOp, {finish(outDist, fallback, concatOp.getType(),
                        /*recomputeSummary=*/false)});

  return success();
}

LogicalResult DistributionAnalysis::visitExpOp(stablehlo::ExpOp expOp) {
  auto inputDistOrFailure = getTensorDistribution(expOp.getOperand());
  if (failed(inputDistOrFailure)) {
    return expOp.emitError()
           << "Missing distribution info for exponential operand.";
  }

  auto outDist = mapSlices(**inputDistOrFailure, [](const Distribution &d) {
    return std::optional(expDist(d));
  });
  this->registerDistributions(
      expOp, {finish(*outDist, outDist->summary, expOp.getType())});

  return success();
}

LogicalResult DistributionAnalysis::visitRsqrtOp(stablehlo::RsqrtOp rsqrtOp) {
  auto inputDistOrFailure = getTensorDistribution(rsqrtOp.getOperand());
  if (failed(inputDistOrFailure)) {
    return rsqrtOp.emitError() << "Missing distribution info for input.";
  }

  auto outDist = mapSlices(**inputDistOrFailure, [](const Distribution &d) {
    return rsqrtDist(d);
  });
  if (failed(outDist)) {
    // rsqrt is only real-valued and bounded for strictly positive inputs.
    return rsqrtOp.emitError()
           << "stablehlo.rsqrt over a range that contains values <= 0 is not "
              "supported in distribution analysis";
  }

  this->registerDistributions(
      rsqrtOp, {finish(*outDist, outDist->summary, rsqrtOp.getType())});
  return success();
}

LogicalResult
DistributionAnalysis::visitConvert(stablehlo::ConvertOp convertOp) {
  auto result = convertOp.getResult();
  auto resultType = llvm::dyn_cast<RankedTensorType>(result.getType());
  if (!resultType) {
    return success();
  }
  auto resultElementType = resultType.getElementType();

  if (!resultElementType.isFloat()) {
    // Do nothing if target type is not float
    return success();
  }

  // If the operand already has distribution info, simply re-use it.
  Value operand = convertOp.getOperand();
  auto operandDistIt = this->distributionMap.find(operand);
  if (operandDistIt != this->distributionMap.end()) {
    this->registerDistributions(
        convertOp, {finish(operandDistIt->second,
                           operandDistIt->second.summary, resultType,
                           /*recomputeSummary=*/false)});
    return success();
  }

  Operation *constOp = getConstantDefiningOp(operand.getDefiningOp());
  if (!constOp) {
    // FIXME: For now, we will only handle convert operations that deal with
    // constants.
    convertOp.emitError() << "Cannot handle convert op operand.";
    return failure();
  }

  auto denseAttr =
      llvm::dyn_cast<DenseElementsAttr>(*getConstantValue(constOp));
  if (!denseAttr) {
    return convertOp.emitError()
           << "Expected dense elements attribute for convert operand.";
  }

  auto targetFloatType = llvm::cast<FloatType>(resultElementType);
  FailureOr<TensorDistribution> distOrFailure;
  switch (targetFloatType.getWidth()) {
  case 32:
    distOrFailure = computeConvertedConstantDistribution<float>(
        denseAttr, resultType, options.perAxis);
    break;
  case 64:
    distOrFailure = computeConvertedConstantDistribution<double>(
        denseAttr, resultType, options.perAxis);
    break;
  default:
    return convertOp.emitError()
           << "Unsupported element type for distribution analysis: "
           << resultElementType;
  }

  if (failed(distOrFailure)) {
    return convertOp.emitError() << "Unsupported convert operand element type: "
                                 << denseAttr.getElementType();
  }

  this->registerDistributions(convertOp, {*distOrFailure});

  // Non-float constants are skipped by visitConstantLikeOp. Retroactively
  // attach the converted distribution so the constant is annotated too. The
  // constant may have a different shape than the convert result (reshapes /
  // broadcasts in between), so only the summary is kept in that case.
  if (this->distributionMap.find(constOp->getResult(0)) ==
      this->distributionMap.end()) {
    this->registerDistributions(
        constOp, {finish(*distOrFailure, distOrFailure->summary,
                         constOp->getResult(0).getType(),
                         /*recomputeSummary=*/false)});
  }

  return success();
}

LogicalResult DistributionAnalysis::visitPreloadedOperation(Operation *op) {
  // If the operation already has distribution annotations, collect them and
  // add to the distribution map.

  auto distAttr = op->getAttrOfType<ArrayAttr>(kDistributionAttrName);
  if (!distAttr || distAttr.size() != op->getNumResults()) {
    return op->emitError() << "Invalid " << kDistributionAttrName
                           << " attribute";
  }

  for (size_t i = 0; i < distAttr.size(); i++) {
    auto distOrFailure = arrayAttrToDistribution(op->getLoc(), distAttr[i]);
    if (failed(distOrFailure)) {
      return failure();
    }

    this->distributionMap[op->getResult(i)] =
        makeUniform(*distOrFailure, op->getResult(i).getType());
  }

  return success();
}

LogicalResult DistributionAnalysis::visitOperation(Operation *op) {
  if (op->hasAttr(kDistributionAttrName)) {
    return visitPreloadedOperation(op);
  }

  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<stablehlo::ConstantOp>([&](stablehlo::ConstantOp constOp) {
        return visitConstantLikeOp(constOp, constOp.getValue());
      })
      .Case<arith::ConstantOp>([&](arith::ConstantOp constOp) {
        return visitConstantLikeOp(constOp, constOp.getValue());
      })
      .Case<stablehlo::AddOp>(
          [&](stablehlo::AddOp addOp) { return visitAddOp(addOp); })
      .Case<stablehlo::SubtractOp>(
          [&](stablehlo::SubtractOp subOp) { return visitSubtractOp(subOp); })
      .Case<stablehlo::MulOp>(
          [&](stablehlo::MulOp mulOp) { return visitMulOp(mulOp); })
      .Case<stablehlo::DivOp>(
          [&](stablehlo::DivOp divOp) { return visitDivOp(divOp); })
      .Case<stablehlo::MaxOp>(
          [&](stablehlo::MaxOp maxOp) { return visitMaxOp(maxOp); })
      .Case<stablehlo::ClampOp>(
          [&](stablehlo::ClampOp clampOp) { return visitClampOp(clampOp); })
      .Case<stablehlo::BroadcastInDimOp>(
          [&](stablehlo::BroadcastInDimOp bcastOp) {
            return visitBroadcastInDim(bcastOp);
          })
      .Case<stablehlo::TransposeOp>([&](stablehlo::TransposeOp transposeOp) {
        return visitTranspose(transposeOp);
      })
      .Case<stablehlo::ReshapeOp>([&](stablehlo::ReshapeOp reshapeOp) {
        return visitReshape(reshapeOp);
      })
      .Case<stablehlo::SliceOp>(
          [&](stablehlo::SliceOp sliceOp) { return visitSlice(sliceOp); })
      .Case<stablehlo::GatherOp>([&](Operation *identityOp) {
        // FIXME: Track per-axis information through gathers.
        return visitUnaryIdentityOp(identityOp);
      })
      .Case<stablehlo::DotGeneralOp>(
          [&](stablehlo::DotGeneralOp dotOp) { return visitDotGeneral(dotOp); })
      .Case<stablehlo::ConvolutionOp>([&](stablehlo::ConvolutionOp convOp) {
        return visitConvolution(convOp);
      })
      .Case<stablehlo::ReduceWindowOp>(
          [&](stablehlo::ReduceWindowOp reduceWindowOp) {
            return visitReduceWindow(reduceWindowOp);
          })
      .Case<stablehlo::ReduceOp>(
          [&](stablehlo::ReduceOp reduceOp) { return visitReduce(reduceOp); })
      .Case<stablehlo::PadOp>(
          [&](stablehlo::PadOp padOp) { return visitPad(padOp); })
      .Case<stablehlo::ConcatenateOp>([&](stablehlo::ConcatenateOp concatOp) {
        return visitConcatenate(concatOp);
      })
      .Case<stablehlo::ExpOp>(
          [&](stablehlo::ExpOp expOp) { return visitExpOp(expOp); })
      .Case<stablehlo::RsqrtOp>(
          [&](stablehlo::RsqrtOp rsqrtOp) { return visitRsqrtOp(rsqrtOp); })
      .Case<stablehlo::ConvertOp>([&](stablehlo::ConvertOp convertOp) {
        return visitConvert(convertOp);
      })
      .Default([&](Operation *op) {
        // FIXME: the default behavior should be for unsupported ops to act as
        // identities?
        LLVM_DEBUG(llvm::dbgs() << "[visitOperation] Unsupported operation: "
                                << op->getName() << "\n");
        return success();
      });
}

FailureOr<const Distribution *>
DistributionAnalysis::getDistribution(Value value) {
  auto distOrFailure = getTensorDistribution(value);
  if (failed(distOrFailure)) {
    return failure();
  }
  return &(*distOrFailure)->summary;
}

FailureOr<const TensorDistribution *>
DistributionAnalysis::getTensorDistribution(Value value) {
  auto it = distributionMap.find(value);
  if (it == distributionMap.end()) {
    return mlir::emitError(value.getLoc())
           << "Missing distribution info for value: " << value;
  }

  return &it->second;
}

LogicalResult DistributionAnalysis::getDistributionForArgs(func::FuncOp func) {
  auto distAttr = func->getAttrOfType<ArrayAttr>(kArgsDistributionAttrName);
  if (!distAttr) {
    // No distribution info for function arguments, nothing to do.
    return success();
  }

  if (func.getNumArguments() != distAttr.size()) {
    return mlir::emitError(func.getLoc())
           << "Size of args distribution attribute (" << distAttr.size()
           << ") must match the number of "
              "function arguments ("
           << func.getNumArguments() << ")";
  }

  for (size_t i = 0; i < distAttr.size(); i++) {
    auto distOrFailure = arrayAttrToDistribution(func.getLoc(), distAttr[i]);
    if (failed(distOrFailure)) {
      return failure();
    }

    this->distributionMap[func.getArgument(i)] =
        makeUniform(*distOrFailure, func.getArgument(i).getType());
  }

  return success();
}

LogicalResult DistributionAnalysis::run(func::FuncOp func) {
  if (failed(getDistributionForArgs(func))) {
    return mlir::emitError(func.getLoc())
           << "Failed to get distribution info for function arguments";
  }

  for (auto &op : func.getBody().front().getOperations()) {
    if (failed(visitOperation(&op))) {
      return mlir::emitError(func.getLoc())
             << "Failed to analyze distribution for operation: "
             << op.getName();
    }
  }

  return success();
}
} // namespace mlir::vishap
