#include "Analysis/DistributionAnalysis.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <cmath>

namespace mlir::vishap {
namespace {
#define DEBUG_TYPE "vishap-analysis"

template <typename T>
Distribution computeDistribution(DenseElementsAttr denseAttr) {
  double max = std::numeric_limits<double>::min();
  double min = std::numeric_limits<double>::max();
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
    return llvm::success();
  }

  // Extract the constant value and update the distribution map
  auto valueAttr = constOp.getValue();
  if (auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(valueAttr)) {
    Distribution distribution;
    switch (tensorType.getElementType().getIntOrFloatBitWidth()) {
    case 32: {
      distributionMap[constOp.getResult()] =
          computeDistribution<float>(denseAttr);
      break;
    }
    case 64: {
      distributionMap[constOp.getResult()] =
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

  const auto *dist1 = getDistribution(inputs[0]);
  const auto *dist2 = getDistribution(inputs[1]);
  if (!dist1 || !dist2) {
    // If we don't have distribution info for one of the inputs, we can't
    // compute the distribution for the output.
    return addOp.emitError()
           << "Missing distribution info for at least one of the inputs.";
  }

  Distribution outputDist;
  outputDist.min = dist1->min + dist2->min;
  outputDist.max = dist1->max + dist2->max;
  // Premise: inputs are independent and are described by a normal distribution
  outputDist.mean = dist1->mean + dist2->mean;
  outputDist.variance = dist1->variance + dist2->variance;

  // Sanity check
  assert(addOp->getNumResults() == 1 &&
         "Expected linalg.add to have exactly one result");

  distributionMap[addOp.getResult(0)] = outputDist;
  this->analyzedOperations.insert(addOp);

  return success();
}

LogicalResult DistributionAnalysis::visitClamp(linalg::GenericOp clamp,
                                               double clampValue) {
  auto inputs = clamp.getInputs();
  if (inputs.size() != 1) {
    return clamp->emitError() << "Expected clamp to have exactly 1 input";
  }

  auto outputs = clamp.getOutputs();
  if (outputs.size() != 1) {
    return clamp->emitError() << "Expected clamp to have exactly 1 output";
  }

  const auto *inputDist = getDistribution(inputs[0]);

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

  Distribution outputDist;
  outputDist.min = std::max(inputDist->min, clampValue);
  outputDist.max = std::max(inputDist->max, clampValue);
  outputDist.mean = rectifiedMean;
  outputDist.variance = rectifiedVariance;

  this->distributionMap[clamp.getResult(0)] = outputDist;
  this->analyzedOperations.insert(clamp);

  return success();
}

LogicalResult DistributionAnalysis::visitUnaryIdentityOp(linalg::LinalgOp op) {
  if (!op.isSingleInputOutput()) {
    return op.emitError() << "Expected unary operation with exactly "
                             "one input and one output";
  }

  const auto *inputDist = getDistribution(op->getOperand(0));
  if (!inputDist) {
    return op.emitError() << "Missing distribution info for input.";
  }

  distributionMap[op->getResult(0)] = *inputDist;
  this->analyzedOperations.insert(op);

  return success();
}

LogicalResult
DistributionAnalysis::visitGenericOp(linalg::GenericOp genericOp) {
  auto body = genericOp.getBody();

  auto &ops = body->getOperations();
  if (ops.size() != 3) {
    LLVM_DEBUG(llvm::dbgs() << "Generic op does not match expected pattern\n");
    return success();
  }

  // FIXME: this is a hack to identify linalg.generic ops that implement other
  // ops. Ideally, the analysis should work at a higher abstraction level,
  // where it would be trivial to identify computational graph operations, but
  // this is not the case with linalg.

  // FIXME: should we check which constant is used for clamping? For now, assume
  // it is 0.
  auto it = ops.begin();
  if (llvm::isa<arith::CmpFOp>(*it) &&
      llvm::isa<arith::SelectOp>(*std::next(it))) {
    return visitClamp(genericOp, /* clampValue= */ 0.0);
  }

  return success();
}

LogicalResult DistributionAnalysis::visitMatmul(linalg::MatmulOp matmulOp) {
  auto inputs = matmulOp.getInputs();
  assert(inputs.size() == 2 &&
         "Expected linalg.matmul to have exactly 2 inputs");

  const auto *dist1 = getDistribution(inputs[0]);
  const auto *dist2 = getDistribution(inputs[1]);
  if (!dist1 || !dist2) {
    // If we don't have distribution info for one of the inputs, we can't
    // compute the distribution for the output.
    return matmulOp.emitError()
           << "Missing distribution info for at least one of the inputs.";
  }

  auto innerDim =
      llvm::cast<ShapedType>(inputs[1].getType()).getShape().front();

  Distribution outputDist;
  // Assume that the mean of each element product is dist1->mean * dist2->mean.
  // Moreover, for each element in the output, we sum innerDim values.
  outputDist.mean = innerDim * dist1->mean * dist2->mean;

  // Variance of a product of independent variables
  double varTerm = (dist1->variance * dist2->variance) +
                   (dist1->variance * dist2->mean * dist2->mean) +
                   (dist2->variance * dist1->mean * dist1->mean);
  // Final estimate
  outputDist.variance = innerDim * varTerm;

  // FIXME: We try to avoid gross overestimations by using a fixed number of
  // standard deviations from the mean. This is heuristic may be incorrect or
  // unsound. We need to test it empirically, or try a more dynamic approach.
  constexpr double delta = 6.0;
  outputDist.min = outputDist.mean - delta * std::sqrt(outputDist.variance);
  outputDist.max = outputDist.mean + delta * std::sqrt(outputDist.variance);

  distributionMap[matmulOp.getResult(0)] = outputDist;
  this->analyzedOperations.insert(matmulOp);

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
      .Default([&](Operation *) {
        // FIXME: the default behavior should be for unsupported ops to act as
        // identities.
        LLVM_DEBUG(llvm::dbgs()
                   << "Unsupported operation: " << op->getName() << "\n");
        return success();
      });
}

LogicalResult DistributionAnalysis::run(func::FuncOp func) {
  for (auto &op : func.getBody().front().getOperations()) {
    if (failed(visitOperation(&op))) {
      return func->emitError()
             << "Failed to analyze distribution for operation: "
             << op.getName();
    }
  }

  return success();
}
} // namespace mlir::vishap
