#include "Analysis/DistributionAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cmath>
#include <limits>
#include <numeric>

namespace mlir::vishap {
namespace {
#define DEBUG_TYPE "vishap-analysis"

template <typename T>
Distribution computeDistribution(DenseElementsAttr denseAttr) {
  double max = std::numeric_limits<double>::lowest();
  double min = std::numeric_limits<double>::infinity();
  double sum = 0.0;
  auto values = denseAttr.getValues<T>();
  for (auto element : values) {
    max = max > element ? max : static_cast<double>(element);
    min = min < element ? min : static_cast<double>(element);
    // FIXME: this could overflow for large values
    sum += element;
  }

  double mean = sum / values.size();
  double variance = 0.0;
  // FIXME: should we traverse the whole constant, or only take a sample?
  // For large constants, this could be expensive.
  for (auto element : values) {
    double elementDouble = static_cast<double>(element);
    variance += (elementDouble - mean) * (elementDouble - mean);
  }
  variance /= values.size();

  LLVM_DEBUG(llvm::dbgs() << "Computed distribution: "
                          << "min=" << min << ", max=" << max << ", mean="
                          << mean << ", variance=" << variance << "\n");

  return {min, max, mean, variance};
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

LogicalResult DistributionAnalysis::visitConstantOp(arith::ConstantOp constOp) {
  auto type = constOp.getType();
  auto tensorType = llvm::dyn_cast<RankedTensorType>(type);
  if (!tensorType || !tensorType.getElementType().isFloat()) {
    // FIXME: For now, we only consider float tensors
    LLVM_DEBUG(llvm::dbgs() << "Unsupported constant type: " << type << "\n");
    return success();
  }

  // Extract the constant value and update the distribution map
  auto valueAttr = constOp.getValue();
  if (auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(valueAttr)) {
    Distribution distribution;
    switch (tensorType.getElementType().getIntOrFloatBitWidth()) {
    case 32: {
      this->distributionMap[constOp.getResult()] =
          computeDistribution<float>(denseAttr);
      break;
    }
    case 64: {
      this->distributionMap[constOp.getResult()] =
          computeDistribution<double>(denseAttr);
      break;
    }
    default:
      return constOp.emitError()
             << "Unsupported element type for distribution analysis: "
             << tensorType.getElementType();
    }
  }

  this->analyzedOperations.insert(constOp);
  return success();
}

LogicalResult DistributionAnalysis::visitAddOp(linalg::AddOp addOp) {
  auto inputs = addOp.getInputs();
  assert(inputs.size() == 2 && "Expected linalg.add to have exactly 2 inputs");

  auto lhsDistOrFailure = getDistribution(inputs[0]);
  auto rhsDistOrFailure = getDistribution(inputs[1]);
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

  // Sanity check
  assert(addOp->getNumResults() == 1 &&
         "Expected linalg.add to have exactly one result");

  this->distributionMap[addOp.getResult(0)] = outputDist;
  this->analyzedOperations.insert(addOp);

  return success();
}

LogicalResult DistributionAnalysis::visitClamp(linalg::GenericOp clamp,
                                               double clampValue) {
  auto inputs = clamp.getInputs();
  if (inputs.size() != 1) {
    return clamp.emitError() << "Expected clamp to have exactly 1 input";
  }

  auto outputs = clamp.getOutputs();
  if (outputs.size() != 1) {
    return clamp.emitError() << "Expected clamp to have exactly 1 output";
  }

  auto inputDistOrFailure = getDistribution(inputs[0]);
  if (failed(inputDistOrFailure)) {
    return clamp.emitError() << "Missing distribution info for input.";
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

  this->distributionMap[clamp.getResult(0)] = outputDist;
  this->analyzedOperations.insert(clamp);

  return success();
}

LogicalResult DistributionAnalysis::visitDivFOp(linalg::GenericOp divOp,
                                                Value divisor) {
  auto inputs = divOp.getInputs();
  if (inputs.size() != 1) {
    return divOp.emitError() << "Expected division op to have exactly 1 input";
  }

  auto outputs = divOp.getOutputs();
  if (outputs.size() != 1) {
    return divOp.emitError() << "Expected division op to have exactly 1 output";
  }

  auto inputDistOrFailure = getDistribution(inputs[0]);
  if (failed(inputDistOrFailure)) {
    return divOp.emitError() << "Missing distribution info for input.";
  }

  auto constOp = divisor.getDefiningOp<arith::ConstantOp>();
  if (!constOp || !constOp.getType().isFloat()) {
    return divOp.emitError()
           << "Expected divisor to be defined by a scalar float constant op";
  }

  double divisorValue =
      llvm::cast<FloatAttr>(constOp.getValue()).getValueAsDouble();
  if (divisorValue == 0.0) {
    return divOp.emitError()
           << "Division by zero is not supported in distribution analysis";
  }

  const auto *inputDist = *inputDistOrFailure;
  Distribution outputDist = {.min = inputDist->min / divisorValue,
                             .max = inputDist->max / divisorValue,
                             .mean = inputDist->mean / divisorValue,
                             .variance = inputDist->variance /
                                         (divisorValue * divisorValue)};

  this->distributionMap[divOp.getResult(0)] = outputDist;
  this->analyzedOperations.insert(divOp);

  return success();
}

LogicalResult DistributionAnalysis::visitUnaryIdentityOp(linalg::LinalgOp op) {
  if (!op.isSingleInputOutput()) {
    return op.emitError() << "Expected unary operation with exactly "
                             "one input and one output";
  }

  auto inputDistOrFailure = getDistribution(op->getOperand(0));
  if (failed(inputDistOrFailure)) {
    return op.emitError() << "Missing distribution info for input.";
  }

  this->distributionMap[op->getResult(0)] = *(*inputDistOrFailure);
  this->analyzedOperations.insert(op);

  return success();
}

LogicalResult
DistributionAnalysis::visitGenericOp(linalg::GenericOp genericOp) {
  auto body = genericOp.getBody();
  auto &ops = body->getOperations();

  // FIXME: this is a hack to identify linalg.generic ops that implement other
  // ops. Ideally, the analysis should work at a higher abstraction level,
  // where it would be trivial to identify computational graph operations, but
  // this is not the case with linalg.

  // FIXME: should we check which constant is used for clamping? For now, assume
  // it is 0.
  auto it = ops.begin();
  if (ops.size() == 3 && llvm::isa<arith::CmpFOp>(*it) &&
      llvm::isa<arith::SelectOp>(*std::next(it))) {
    // ReLU/Clamp pattern
    return visitClamp(genericOp, /* clampValue= */ 0.0);
  }

  if (ops.size() == 2) {
    if (auto divOp = llvm::dyn_cast<arith::DivFOp>(*it)) {
      // Div op pattern
      auto divisor = divOp.getRhs();
      return visitDivFOp(genericOp, divisor);
    }
  }

  return genericOp.emitError() << "Unsupported linalg.generic pattern";
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

LogicalResult DistributionAnalysis::visitMatmul(linalg::MatmulOp matmulOp) {
  auto inputs = matmulOp.getInputs();
  assert(inputs.size() == 2 &&
         "Expected linalg.matmul to have exactly 2 inputs");

  auto lhsDistOrFailure = getDistribution(inputs[0]);
  auto rhsDistOrFailure = getDistribution(inputs[1]);
  if (failed(lhsDistOrFailure) || failed(rhsDistOrFailure)) {
    // If we don't have distribution info for one of the inputs, we can't
    // compute the distribution for the output.
    return matmulOp.emitError()
           << "Missing distribution info for at least one of the inputs.";
  }

  const auto *lhsDist = *lhsDistOrFailure;
  const auto *rhsDist = *rhsDistOrFailure;
  auto innerDim =
      llvm::cast<ShapedType>(inputs[1].getType()).getShape().front();

  Distribution outputDist =
      getMatrixMultiplicationDistribution(innerDim, lhsDist, rhsDist);

  this->distributionMap[matmulOp.getResult(0)] = outputDist;
  this->analyzedOperations.insert(matmulOp);

  return success();
}

LogicalResult
DistributionAnalysis::visitConv2D(linalg::Conv2DNchwFchwOp convOp) {
  auto inputs = convOp.getInputs();
  assert(inputs.size() == 2 &&
         "Expected linalg.conv_2d_nchw_fchw to have exactly 2 inputs");

  auto inputDistOrFailure = getDistribution(inputs[0]);
  auto kernelDistOrFailure = getDistribution(inputs[1]);
  if (failed(inputDistOrFailure) || failed(kernelDistOrFailure)) {
    return convOp.emitError()
           << "Missing distribution info for at least one of the inputs.";
  }
  const auto *inputDist = *inputDistOrFailure;
  const auto *kernelDist = *kernelDistOrFailure;

  auto inputShape = llvm::cast<ShapedType>(inputs[0].getType()).getShape();
  auto kernelShape = llvm::cast<ShapedType>(inputs[1].getType()).getShape();

  auto inputChannels = inputShape[1];
  auto kernelHeight = kernelShape[2];
  auto kernelWidth = kernelShape[3];

  // This is the size of the number of elements in the "sliding window" of
  // convolution
  auto contractionSize = inputChannels * kernelHeight * kernelWidth;

  Distribution outputDist = getMatrixMultiplicationDistribution(
      contractionSize, inputDist, kernelDist);

  this->distributionMap[convOp.getResult(0)] = outputDist;
  this->analyzedOperations.insert(convOp);

  return success();
}

LogicalResult DistributionAnalysis::visitFill(linalg::FillOp fillOp) {
  auto inputs = fillOp.getInputs();
  assert(inputs.size() == 1 && "Expected linalg.fill to have exactly 1 input");

  auto constOp = inputs[0].getDefiningOp<arith::ConstantOp>();
  if (!inputs[0].getType().isFloat() || !constOp) {
    return fillOp.emitError()
           << "Expected linalg.fill input to be a floating point scalar";
  }

  double fillValue =
      llvm::cast<FloatAttr>(constOp.getValue()).getValueAsDouble();
  Distribution outputDist = {
      .min = fillValue, .max = fillValue, .mean = fillValue, .variance = 0.0};

  this->distributionMap[fillOp.getResult(0)] = outputDist;
  this->analyzedOperations.insert(fillOp);

  return success();
}

LogicalResult DistributionAnalysis::visitPad(tensor::PadOp padOp) {
  auto source = padOp.getSource();
  auto sourceDistOrFailure = getDistribution(source);
  if (failed(sourceDistOrFailure)) {
    return padOp.emitError() << "Missing distribution info for source.";
  }
  const auto *sourceDist = *sourceDistOrFailure;

  // Get value used for padding
  auto &ops = padOp.getBody()->getOperations();
  if (ops.size() != 1) {
    return padOp.emitError()
           << "Expected pad body to have exactly 1 operation (tensor.yield)";
  }
  auto constOp = llvm::cast<tensor::YieldOp>(ops.back())
                     .getValue()
                     .getDefiningOp<arith::ConstantOp>();
  if (!constOp || !constOp.getType().isFloat()) {
    return padOp.emitError() << "Expected pad value to be defined by a "
                                "floating point scalar constant";
  }
  double padValue =
      llvm::cast<FloatAttr>(constOp.getValue()).getValueAsDouble();

  // Compute size of source tensor
  auto sourceShape = source.getType().getShape();
  size_t tensorSize = std::accumulate(sourceShape.begin(), sourceShape.end(), 1,
                                      std::multiplies<size_t>());

  // Compute size of padding
  auto outShape = padOp.getResult().getType().getShape();
  size_t outSize = std::accumulate(outShape.begin(), outShape.end(), 1,
                                   std::multiplies<size_t>());
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
DistributionAnalysis::visitSumPooling(linalg::PoolingNchwSumOp poolingOp) {
  auto inputs = poolingOp.getInputs();
  assert(inputs.size() == 2 &&
         "Expected linalg.pooling_nchw_sum to have exactly 2 inputs");

  auto input = inputs[0];
  auto inputDistOrFailure = getDistribution(input);
  if (failed(inputDistOrFailure)) {
    return poolingOp.emitError() << "Missing distribution info for input.";
  }

  auto kernel = inputs[1];
  auto kernelShape = llvm::cast<ShapedType>(kernel.getType()).getShape();
  auto poolingWindowSize = kernelShape[0] * kernelShape[1];

  const auto *inputDist = *inputDistOrFailure;
  Distribution outputDist = {
      .min = inputDist->min * poolingWindowSize,
      .max = inputDist->max * poolingWindowSize,
      .mean = inputDist->mean * poolingWindowSize,
      // FIXME: There probably is a more accurate way to estimate this
      .variance = inputDist->variance * poolingWindowSize};

  this->distributionMap[poolingOp->getResult(0)] = outputDist;
  this->analyzedOperations.insert(poolingOp);

  return success();
}

LogicalResult DistributionAnalysis::visitConcat(tensor::ConcatOp concatOp) {
  auto axis = concatOp.getDim();

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

LogicalResult DistributionAnalysis::visitCollapseShape(
    tensor::CollapseShapeOp collapseShapeOp) {
  auto input = collapseShapeOp.getSrc();
  auto inputDistOrFailure = getDistribution(input);
  if (failed(inputDistOrFailure)) {
    return collapseShapeOp.emitError()
           << "Missing distribution info for source.";
  }

  // FIXME: Assume tensor.collapse_shape is an identity. Does this always hold?
  this->distributionMap[collapseShapeOp->getResult(0)] = *(*inputDistOrFailure);
  this->analyzedOperations.insert(collapseShapeOp);

  return success();
}

LogicalResult DistributionAnalysis::visitOperation(Operation *op) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<arith::ConstantOp>(
          [&](arith::ConstantOp constOp) { return visitConstantOp(constOp); })
      .Case<linalg::AddOp>(
          [&](linalg::AddOp addOp) { return visitAddOp(addOp); })
      .Case<linalg::GenericOp>([&](linalg::GenericOp genericOp) {
        return visitGenericOp(genericOp);
      })
      .Case<linalg::BroadcastOp>([&](linalg::BroadcastOp broadcastOp) {
        return visitUnaryIdentityOp(broadcastOp);
      })
      .Case<linalg::TransposeOp>([&](linalg::TransposeOp transposeOp) {
        return visitUnaryIdentityOp(transposeOp);
      })
      .Case<linalg::MatmulOp>(
          [&](linalg::MatmulOp matmulOp) { return visitMatmul(matmulOp); })
      .Case<linalg::Conv2DNchwFchwOp>(
          [&](linalg::Conv2DNchwFchwOp convOp) { return visitConv2D(convOp); })
      .Case<linalg::FillOp>(
          [&](linalg::FillOp fillOp) { return visitFill(fillOp); })
      .Case<tensor::PadOp>([&](tensor::PadOp padOp) { return visitPad(padOp); })
      .Case<linalg::PoolingNchwSumOp>([&](linalg::PoolingNchwSumOp poolingOp) {
        return visitSumPooling(poolingOp);
      })
      .Case<tensor::ConcatOp>(
          [&](tensor::ConcatOp concatOp) { return visitConcat(concatOp); })
      .Case<tensor::CollapseShapeOp>(
          [&](tensor::CollapseShapeOp collapseShapeOp) {
            return visitCollapseShape(collapseShapeOp);
          })
      .Default([&](Operation *op) {
        // FIXME: the default behavior should be for unsupported ops to act as
        // identities?
        LLVM_DEBUG(llvm::dbgs()
                   << "Unsupported operation: " << op->getName() << "\n");
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
