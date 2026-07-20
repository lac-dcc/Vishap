#include "Analysis/DistributionAnalysis.h"
#include "Support/Distribution.h"
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

namespace mlir::vishap {
namespace {
#define DEBUG_TYPE "vishap-analysis"

/// Try to extract the value of a scalar or splat float constant that defines
/// \p value, looking through shape-preserving ops (broadcast, reshape).
FailureOr<double> getScalarConstantValue(Value value) {
  Operation *def = value.getDefiningOp();
  while (def &&
         llvm::isa<stablehlo::BroadcastInDimOp, stablehlo::ReshapeOp>(def)) {
    def = def->getOperand(0).getDefiningOp();
  }

  if (!def) {
    return failure();
  }

  Attribute valueAttr;
  if (auto stablehloConst = llvm::dyn_cast<stablehlo::ConstantOp>(def)) {
    valueAttr = stablehloConst.getValue();
  } else if (auto arithConst = llvm::dyn_cast<arith::ConstantOp>(def)) {
    valueAttr = arithConst.getValue();
  } else {
    return failure();
  }

  if (auto floatAttr = llvm::dyn_cast<FloatAttr>(valueAttr)) {
    return floatAttr.getValueAsDouble();
  }

  auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(valueAttr);
  if (!denseAttr || !denseAttr.isSplat() ||
      !llvm::isa<FloatType>(denseAttr.getElementType())) {
    return failure();
  }

  return denseAttr.getSplatValue<APFloat>().convertToDouble();
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
    switch (tensorType.getElementType().getIntOrFloatBitWidth()) {
    case 32: {
      auto values = denseAttr.getValues<float>();
      this->distributionMap[op->getResult(0)] =
          computeFloatRangeDistribution<float, decltype(values)>(values);
      break;
    }
    case 64: {
      auto values = denseAttr.getValues<double>();
      this->distributionMap[op->getResult(0)] =
          computeFloatRangeDistribution<double, decltype(values)>(values);
      break;
    }
    default:
      return op->emitError()
             << "Unsupported element type for distribution analysis: "
             << tensorType.getElementType();
    }
  }

  this->analyzedOperations.insert(op);
  return success();
}

LogicalResult DistributionAnalysis::visitAddOp(stablehlo::AddOp addOp) {
  auto lhsDistOrFailure = getDistribution(addOp.getLhs());
  auto rhsDistOrFailure = getDistribution(addOp.getRhs());
  if (failed(lhsDistOrFailure) || failed(rhsDistOrFailure)) {
    // If we don't have distribution info for one of the inputs, we can't
    // compute the distribution for the output.
    return addOp.emitError()
           << "Missing distribution info for at least one of the inputs.";
  }

  const auto *lhsDist = *lhsDistOrFailure;
  const auto *rhsDist = *rhsDistOrFailure;

  Distribution outputDist;
  outputDist.min = lhsDist->min + rhsDist->min;
  outputDist.max = lhsDist->max + rhsDist->max;
  // Premise: inputs are independent and are described by a normal distribution
  outputDist.mean = lhsDist->mean + rhsDist->mean;
  outputDist.variance = lhsDist->variance + rhsDist->variance;

  this->distributionMap[addOp.getResult()] = outputDist;
  this->analyzedOperations.insert(addOp);

  return success();
}

LogicalResult DistributionAnalysis::visitClamp(Operation *op, Value input,
                                               double clampValue) {
  auto inputDistOrFailure = getDistribution(input);
  if (failed(inputDistOrFailure)) {
    return op->emitError() << "Missing distribution info for input.";
  }

  const auto *inputDist = *inputDistOrFailure;

  double sigma = std::sqrt(inputDist->variance);
  double alpha = (clampValue - inputDist->mean) / sigma;

  // 1. Calculate probabilities (CDF and PDF)
  double cdf = 0.5 * (1 + std::erf(alpha / std::sqrt(2)));
  double pdf = (1 / std::sqrt(2 * M_PI)) * std::exp(-0.5 * alpha * alpha);

  double clampProb = cdf;
  double tailProb = 1 - cdf;

  // 2. Calculate tail statistics
  // Inverse Millis ratio
  double lam = pdf / tailProb;
  double tailMean = inputDist->mean + (sigma * lam);
  double delta = lam * (lam - alpha);
  double tailVariance = inputDist->variance * (1 - delta);

  // 3. Combine moments
  // New mean E[X]
  double rectifiedMean = (clampProb * clampValue) + (tailProb * tailMean);

  // Second moment E[X^2]
  // For the tail part, E[X^2] = Var + Mean^2
  double tailSecondMoment = tailVariance + (tailMean * tailMean);
  // For the clamped part, E[X^2] = C^2
  double rectifiedSecondMoment =
      (clampProb * (clampValue * clampValue)) + (tailProb * tailSecondMoment);

  // Variance Var[X] = E[X^2] - (E[X])^2
  double rectifiedVariance =
      rectifiedSecondMoment - (rectifiedMean * rectifiedMean);

  Distribution outputDist = {.min = std::max(inputDist->min, clampValue),
                             .max = std::max(inputDist->max, clampValue),
                             .mean = rectifiedMean,
                             .variance = rectifiedVariance};

  this->distributionMap[op->getResult(0)] = outputDist;
  this->analyzedOperations.insert(op);

  return success();
}

LogicalResult DistributionAnalysis::visitMaxOp(stablehlo::MaxOp maxOp) {
  // A maximum against a constant is a clamp from below (e.g. ReLU when the
  // constant is 0).
  auto lhsConstOrFailure = getScalarConstantValue(maxOp.getLhs());
  auto rhsConstOrFailure = getScalarConstantValue(maxOp.getRhs());

  if (succeeded(rhsConstOrFailure)) {
    return visitClamp(maxOp, maxOp.getLhs(), *rhsConstOrFailure);
  }

  if (succeeded(lhsConstOrFailure)) {
    return visitClamp(maxOp, maxOp.getRhs(), *lhsConstOrFailure);
  }

  return maxOp.emitError()
         << "Expected one of the operands of stablehlo.maximum to be a "
            "scalar or splat float constant";
}

LogicalResult DistributionAnalysis::visitDivOp(stablehlo::DivOp divOp) {
  auto lhsDistOrFailure = getDistribution(divOp.getLhs());
  auto rhsDistOrFailure = getDistribution(divOp.getRhs());
  if (failed(lhsDistOrFailure) || failed(rhsDistOrFailure)) {
    return divOp.emitError()
           << "Missing distribution info for at least one of the inputs.";
  }

  const auto *lhsDist = *lhsDistOrFailure;
  const auto *rhsDist = *rhsDistOrFailure;

  // FIXME: this may be too restrictive
  // The divisor support must not contain zero, otherwise the ratio range is
  // unbounded / undefined.
  if (rhsDist->min <= 0.0 && rhsDist->max >= 0.0) {
    return divOp.emitError()
           << "Division by a value whose range contains zero is not "
              "supported in distribution analysis";
  }

  // Mean/variance assume independent operands (same premise as Add). When the
  // divisor is degenerate these reduce to exact scaling by a constant.
  // Min/max are the range of the ratio over the product of the input intervals
  // (extrema at the corners when the divisor has constant sign).
  double muY = rhsDist->mean;
  double muYSq = muY * muY;
  Distribution outputDist;
  outputDist.mean = lhsDist->mean / muY;
  outputDist.variance =
      lhsDist->variance / muYSq +
      (lhsDist->mean * lhsDist->mean) * rhsDist->variance / (muYSq * muYSq);

  double c00 = lhsDist->min / rhsDist->min;
  double c01 = lhsDist->min / rhsDist->max;
  double c10 = lhsDist->max / rhsDist->min;
  double c11 = lhsDist->max / rhsDist->max;
  outputDist.min = std::min(std::min(c00, c01), std::min(c10, c11));
  outputDist.max = std::max(std::max(c00, c01), std::max(c10, c11));

  this->distributionMap[divOp.getResult()] = outputDist;
  this->analyzedOperations.insert(divOp);

  return success();
}

LogicalResult DistributionAnalysis::visitUnaryIdentityOp(Operation *op) {
  assert(op->getNumOperands() >= 1 && op->getNumResults() == 1 &&
         "Expected unary operation with at least one operand and exactly one "
         "result");

  auto inputDistOrFailure = getDistribution(op->getOperand(0));
  if (failed(inputDistOrFailure)) {
    return op->emitError() << "Missing distribution info for input.";
  }

  this->distributionMap[op->getResult(0)] = *(*inputDistOrFailure);
  this->analyzedOperations.insert(op);

  return success();
}

/// Propagate distribution for a matrix multiplication-like operation, given the
/// distribution of the inputs and the size of the contraction dimension.
Distribution getMatrixMultiplicationDistribution(long contractionSize,
                                                 const Distribution *dist1,
                                                 const Distribution *dist2) {
  Distribution outputDist;
  // Assume that the mean of each element product is dist1.mean * dist2.mean.
  // Moreover, for each element in the output, we sum contractionSize values.
  outputDist.mean = contractionSize * dist1->mean * dist2->mean;

  // Variance of a product of independent variables
  double varTerm = (dist1->variance * dist2->variance) +
                   (dist1->variance * dist2->mean * dist2->mean) +
                   (dist2->variance * dist1->mean * dist1->mean);
  // Final estimate
  outputDist.variance = contractionSize * varTerm;

  // FIXME: We try to avoid gross overestimations by using a fixed number of
  // standard deviations from the mean. This heuristic may be incorrect or
  // unsound. We need to test it empirically, or try a more dynamic approach.
  constexpr double delta = 6.0;
  outputDist.min = outputDist.mean - delta * std::sqrt(outputDist.variance);
  outputDist.max = outputDist.mean + delta * std::sqrt(outputDist.variance);

  return outputDist;
}

LogicalResult
DistributionAnalysis::visitDotGeneral(stablehlo::DotGeneralOp dotOp) {
  auto lhsDistOrFailure = getDistribution(dotOp.getLhs());
  auto rhsDistOrFailure = getDistribution(dotOp.getRhs());
  if (failed(lhsDistOrFailure) || failed(rhsDistOrFailure)) {
    // If we don't have distribution info for one of the inputs, we can't
    // compute the distribution for the output.
    return dotOp.emitError()
           << "Missing distribution info for at least one of the inputs.";
  }

  const auto *lhsDist = *lhsDistOrFailure;
  const auto *rhsDist = *rhsDistOrFailure;

  // The contraction size is the number of elements that are multiplied and
  // accumulated for each element of the output.
  auto lhsShape = llvm::cast<ShapedType>(dotOp.getLhs().getType()).getShape();
  auto dimNumbers = dotOp.getDotDimensionNumbers();
  int64_t contractionSize = 1;
  for (auto dim : dimNumbers.getLhsContractingDimensions()) {
    if (lhsShape[dim] == ShapedType::kDynamic) {
      return dotOp.emitError()
             << "Dynamic contracting dimensions are not supported";
    }
    contractionSize *= lhsShape[dim];
  }

  Distribution outputDist =
      getMatrixMultiplicationDistribution(contractionSize, lhsDist, rhsDist);

  this->distributionMap[dotOp.getResult()] = outputDist;
  this->analyzedOperations.insert(dotOp);

  return success();
}

LogicalResult
DistributionAnalysis::visitConvolution(stablehlo::ConvolutionOp convOp) {
  auto inputDistOrFailure = getDistribution(convOp.getLhs());
  auto kernelDistOrFailure = getDistribution(convOp.getRhs());
  if (failed(inputDistOrFailure) || failed(kernelDistOrFailure)) {
    return convOp.emitError()
           << "Missing distribution info for at least one of the inputs.";
  }
  const auto *inputDist = *inputDistOrFailure;
  const auto *kernelDist = *kernelDistOrFailure;

  auto inputShape =
      llvm::cast<ShapedType>(convOp.getLhs().getType()).getShape();
  auto kernelShape =
      llvm::cast<ShapedType>(convOp.getRhs().getType()).getShape();

  // The dimension numbers make this layout-agnostic (i.e. not restricted to
  // NCHW/FCHW convolutions).
  auto dimNumbers = convOp.getDimensionNumbers();
  auto inputChannels = inputShape[dimNumbers.getInputFeatureDimension()] /
                       convOp.getFeatureGroupCount();

  // This is the size of the number of elements in the "sliding window" of
  // convolution
  int64_t contractionSize = inputChannels;
  for (auto dim : dimNumbers.getKernelSpatialDimensions()) {
    contractionSize *= kernelShape[dim];
  }

  Distribution outputDist = getMatrixMultiplicationDistribution(
      contractionSize, inputDist, kernelDist);

  this->distributionMap[convOp.getResult()] = outputDist;
  this->analyzedOperations.insert(convOp);

  return success();
}

LogicalResult DistributionAnalysis::visitReductionLike(Operation *op,
                                                       Value input,
                                                       Region &body,
                                                       int64_t reduceSize) {
  auto inputDistOrFailure = getDistribution(input);
  if (failed(inputDistOrFailure)) {
    return op->emitError() << "Missing distribution info for input.";
  }
  const auto *inputDist = *inputDistOrFailure;

  // The body must consist of a single reduction op followed by the implicit
  // stablehlo.return terminator.
  auto &bodyOps = body.front().getOperations();
  if (bodyOps.size() != 2) {
    return op->emitError() << "Unsupported reduction body in " << op->getName();
  }

  auto &reductionOp = bodyOps.front();
  Distribution outputDist;
  if (llvm::isa<stablehlo::AddOp>(reductionOp)) {
    // Sum reduction.
    outputDist = {.min = inputDist->min * reduceSize,
                  .max = inputDist->max * reduceSize,
                  .mean = inputDist->mean * reduceSize,
                  // FIXME: There probably is a more accurate way to estimate
                  // this
                  .variance = inputDist->variance * reduceSize};
  } else if (llvm::isa<stablehlo::MaxOp>(reductionOp)) {
    // Max reduction.
    outputDist = {.min = inputDist->min,
                  .max = inputDist->max,
                  // FIMXE: This estimation can be improved
                  .mean = inputDist->mean + inputDist->variance,
                  .variance = inputDist->variance};
  } else {
    return op->emitError() << "Unsupported reduction in " << op->getName()
                           << ": " << reductionOp.getName();
  }

  this->distributionMap[op->getResult(0)] = outputDist;
  this->analyzedOperations.insert(op);

  return success();
}

LogicalResult DistributionAnalysis::visitReduceWindow(
    stablehlo::ReduceWindowOp reduceWindowOp) {
  if (reduceWindowOp.getInputs().size() != 1) {
    return reduceWindowOp.emitError()
           << "Expected stablehlo.reduce_window to have exactly 1 input";
  }

  auto input = reduceWindowOp.getInputs().front();
  int64_t windowSize = llvm::accumulate(reduceWindowOp.getWindowDimensions(),
                                        /*Init=*/1, std::multiplies<int64_t>());
  return visitReductionLike(reduceWindowOp, input, reduceWindowOp.getBody(),
                            windowSize);
}

LogicalResult DistributionAnalysis::visitReduce(stablehlo::ReduceOp reduceOp) {
  if (reduceOp.getInputs().size() != 1) {
    return reduceOp.emitError()
           << "Expected stablehlo.reduce to have exactly 1 input";
  }

  auto input = reduceOp.getInputs().front();
  auto inputShape = llvm::cast<ShapedType>(input.getType()).getShape();
  int64_t reduceSize = 1;
  for (int64_t dim : reduceOp.getDimensions()) {
    if (inputShape[dim] == ShapedType::kDynamic) {
      return reduceOp.emitError()
             << "Dynamic reduced dimensions are not supported";
    }
    reduceSize *= inputShape[dim];
  }

  return visitReductionLike(reduceOp, input, reduceOp.getBody(), reduceSize);
}

LogicalResult DistributionAnalysis::visitPad(stablehlo::PadOp padOp) {
  auto source = padOp.getOperand();
  auto sourceDistOrFailure = getDistribution(source);
  if (failed(sourceDistOrFailure)) {
    return padOp.emitError() << "Missing distribution info for source.";
  }
  const auto *sourceDist = *sourceDistOrFailure;

  auto padValueOrFailure = getScalarConstantValue(padOp.getPaddingValue());
  if (failed(padValueOrFailure)) {
    return padOp.emitError()
           << "Expected pad value to be a scalar or splat float constant";
  }
  double padValue = *padValueOrFailure;

  if (!std::isfinite(padValue)) {
    padOp->emitWarning() << "Pad value is NaN/Inf. Treating pad as identity "
                            "for distribution analysis.";
    this->distributionMap[padOp.getResult()] = (*sourceDist);
    this->analyzedOperations.insert(padOp);
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

  Distribution outputDist;
  outputDist.mean = (originalRatio * sourceDist->mean) + (padRatio * padValue);
  outputDist.variance =
      (originalRatio * sourceDist->variance) +
      originalRatio * (sourceDist->mean - outputDist.mean) *
          (sourceDist->mean - outputDist.mean) +
      padRatio * (padValue - outputDist.mean) * (padValue - outputDist.mean);
  outputDist.min = std::min(sourceDist->min, padValue);
  outputDist.max = std::max(sourceDist->max, padValue);

  this->distributionMap[padOp.getResult()] = outputDist;
  this->analyzedOperations.insert(padOp);

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

  llvm::SmallVector<const Distribution *> dists;
  double min = std::numeric_limits<double>::infinity();
  double max = std::numeric_limits<double>::lowest();
  double mean = 0.;
  for (const auto [input, dim] : llvm::zip(inputs, dims)) {
    auto inputDistOrFailure = getDistribution(input);
    if (failed(inputDistOrFailure)) {
      return concatOp.emitError()
             << "Missing distribution info for at least one of the inputs.";
    }

    const auto *inputDist = *inputDistOrFailure;
    dists.push_back(inputDist);
    min = std::min(min, inputDist->min);
    max = std::max(max, inputDist->max);
    auto weight = static_cast<double>(dim) / newDimSize;
    mean += weight * inputDist->mean;
  }

  double variance = 0.;
  for (const auto [input, dim, dist] : llvm::zip(inputs, dims, dists)) {
    auto weight = static_cast<double>(dim) / newDimSize;
    // Intra-tensor variance + inter-tensor variance
    variance +=
        weight * (dist->variance + (dist->mean - mean) * (dist->mean - mean));
  }

  Distribution outputDist = {
      .min = min, .max = max, .mean = mean, .variance = variance};

  this->distributionMap[concatOp.getResult()] = outputDist;
  this->analyzedOperations.insert(concatOp);

  return success();
}

LogicalResult DistributionAnalysis::visitOperation(Operation *op) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<stablehlo::ConstantOp>([&](stablehlo::ConstantOp constOp) {
        return visitConstantLikeOp(constOp, constOp.getValue());
      })
      .Case<arith::ConstantOp>([&](arith::ConstantOp constOp) {
        return visitConstantLikeOp(constOp, constOp.getValue());
      })
      .Case<stablehlo::AddOp>(
          [&](stablehlo::AddOp addOp) { return visitAddOp(addOp); })
      .Case<stablehlo::MaxOp>(
          [&](stablehlo::MaxOp maxOp) { return visitMaxOp(maxOp); })
      .Case<stablehlo::DivOp>(
          [&](stablehlo::DivOp divOp) { return visitDivOp(divOp); })
      .Case<stablehlo::BroadcastInDimOp, stablehlo::TransposeOp,
            stablehlo::ReshapeOp>([&](Operation *identityOp) {
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
  auto it = distributionMap.find(value);
  if (it == distributionMap.end()) {
    return mlir::emitError(value.getLoc())
           << "Missing distribution info for value: " << value;
  }

  return &it->second;
}

/// Convert \p arrayAttr to a Distribution struct. Emit an error and return
/// failure if the attribute is not well-formed.
FailureOr<Distribution> arrayAttrToDistribution(Location loc,
                                                ArrayAttr arrayAttr) {
  if (arrayAttr.size() != 4) {
    mlir::emitError(loc)
        << "Expected array attribute of size 4 for distribution info, got "
        << arrayAttr.size();
    return failure();
  }

  for (auto attr : arrayAttr) {
    if (!llvm::isa<FloatAttr>(attr)) {
      return mlir::emitError(loc) << "Expected float type for all values in "
                                     "distribution attribute";
    }
  }

  return Distribution{
      .min = llvm::cast<FloatAttr>(arrayAttr[0]).getValueAsDouble(),
      .max = llvm::cast<FloatAttr>(arrayAttr[1]).getValueAsDouble(),
      .mean = llvm::cast<FloatAttr>(arrayAttr[2]).getValueAsDouble(),
      .variance = llvm::cast<FloatAttr>(arrayAttr[3]).getValueAsDouble(),
  };
}

LogicalResult DistributionAnalysis::getDistributionForArgs(func::FuncOp func) {
  auto distAttr = func->getAttrOfType<ArrayAttr>(kArgsDistributionAttrName);
  if (!distAttr) {
    // No distribution info for function arguments, nothing to do.
    return success();
  }

  if (func.getNumArguments() != distAttr.size()) {
    return func.emitError()
           << "Size of args distribution attribute (" << distAttr.size()
           << ") must match the number of "
              "function arguments ("
           << func.getNumArguments() << ")";
  }

  for (size_t i = 0; i < distAttr.size(); i++) {
    auto operandDistAttr = llvm::cast<ArrayAttr>(distAttr[i]);
    auto distOrFailure =
        arrayAttrToDistribution(func->getLoc(), operandDistAttr);
    if (failed(distOrFailure)) {
      return failure();
    }

    this->distributionMap[func.getArgument(i)] = *distOrFailure;
  }

  return success();
}

LogicalResult DistributionAnalysis::run(func::FuncOp func) {
  if (failed(getDistributionForArgs(func))) {
    return func.emitError()
           << "Failed to get distribution info for function arguments";
  }

  for (auto &op : func.getBody().front().getOperations()) {
    if (failed(visitOperation(&op))) {
      return func.emitError()
             << "Failed to analyze distribution for operation: "
             << op.getName();
    }
  }

  return success();
}
} // namespace mlir::vishap
