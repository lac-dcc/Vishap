#include "Analysis/DistributionAnalysis.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

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

  this->analyzedOperations.insert(constOp.getOperation());
  return success();
}

LogicalResult DistributionAnalysis::visitAddOp(linalg::AddOp addOp) {
  auto inputs = addOp.getInputs();
  assert(inputs.size() == 2 && "Expected linalg.add to have exactly 2 inputs");

  const Distribution *dist1 = getDistribution(inputs[0]);
  const Distribution *dist2 = getDistribution(inputs[1]);
  if (!dist1 || !dist2) {
    // If we don't have distribution info for one of the inputs, we can't
    // compute the distribution for the output.
    return addOp.emitWarning()
           << "Missing distribution info for one of the inputs, skipping "
              "distribution analysis for this operation";
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
  this->analyzedOperations.insert(addOp.getOperation());

  return success();
}

LogicalResult DistributionAnalysis::visitOperation(Operation *op) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<arith::ConstantOp>(
          [&](arith::ConstantOp constOp) { return visitConstantOp(constOp); })
      .Case<linalg::AddOp>(
          [&](linalg::AddOp addOp) { return visitAddOp(addOp); })
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
