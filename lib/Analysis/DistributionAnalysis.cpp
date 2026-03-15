#include "Analysis/DistributionAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace vishap;

#define DEBUG_TYPE "vishap-analysis"

namespace {
template <typename T>
Distribution computeDistribution(DenseElementsAttr denseAttr) {
  double max = std::numeric_limits<double>::min();
  double min = std::numeric_limits<double>::max();
  double sum = 0.0;
  auto values = denseAttr.getValues<T>();
  for (auto element : values) {
    max = max > element ? max : static_cast<double>(element);
    min = min < element ? min : static_cast<double>(element);
    sum += element;
  }

  double mean = sum / values.size();
  double variance = 0.0;
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

  return success();
}

LogicalResult DistributionAnalysis::visitOperation(Operation *op) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<arith::ConstantOp>(
          [&](arith::ConstantOp constOp) { return visitConstantOp(constOp); })
      .Default([&](Operation *) {
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
