#include "Analysis/DistributionAnalysis.h"
#include "Dialect/Probe/IR/Probe.h"
#include "Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include <cstdint>

namespace mlir::vishap {
#define GEN_PASS_DEF_ADDPROBECALLSPASS
#include "Transforms/Passes.h.inc"

namespace {
bool isConstantLikeOp(Operation *op) {
  return llvm::isa<stablehlo::ConstantOp, arith::ConstantOp>(op);
}

bool isPureViewOp(Operation *op) {
  return llvm::isa<stablehlo::BroadcastInDimOp, stablehlo::ReshapeOp,
                   stablehlo::TransposeOp>(op);
}

bool isConstantDerivedOp(Operation *op) {
  while (op && (isPureViewOp(op) || llvm::isa<stablehlo::ConvertOp>(op))) {
    op = op->getOperand(0).getDefiningOp();
  }
  return op && isConstantLikeOp(op);
}

bool hasFloatRankedTensorResult(Operation *op) {
  for (Value result : op->getResults()) {
    auto tensorType = llvm::dyn_cast<RankedTensorType>(result.getType());
    if (tensorType && tensorType.getElementType().isFloat())
      return true;
  }
  return false;
}

class AddProbeCallsPass
    : public impl::AddProbeCallsPassBase<AddProbeCallsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (funcOp.getBody().empty()) {
      return;
    }

    std::vector<Operation *> worklist;
    funcOp.walk([&](Operation *op) {
      // Skip ops inside regions (reduce_window / reduce / etc.).
      if (op->getParentOp() != funcOp)
        return;
      // If requested, add probe calls for all available float ops.
      if (allFloatOps) {
        // Never probe constants, for they are unchanged for different
        // calibration inputs.
        if (isConstantLikeOp(op) || !hasFloatRankedTensorResult(op))
          return;
        if (skipRedundantOps &&
            (isPureViewOp(op) ||
             (llvm::isa<stablehlo::ConvertOp>(op) && isConstantDerivedOp(op))))
          return;
        worklist.push_back(op);
        return;
      }
      // Default to only consider ops with vishap distribution attribute
      if (op->hasAttr(kDistributionAttrName))
        worklist.push_back(op);
    });

    auto *ctx = &getContext();
    OpBuilder builder(ctx);
    bool addedAnyObserve = false;

    if (observeFuncArgs) {
      builder.setInsertionPointToStart(&funcOp.getBody().front());
      // Use -1 ID for the function
      auto funcId = builder.getI32IntegerAttr(-1);
      for (auto [argIdx, arg] : llvm::enumerate(funcOp.getArguments())) {
        auto tensorType = llvm::dyn_cast<RankedTensorType>(arg.getType());
        if (!tensorType || !tensorType.getElementType().isFloat())
          continue;
        probe::ObserveOp::create(
            builder, arg.getLoc(), arg, funcId,
            builder.getI32IntegerAttr(static_cast<int32_t>(argIdx)));
        addedAnyObserve = true;
      }
    }

    for (auto [opID, op] : llvm::enumerate(worklist)) {
      for (auto [resultID, result] : llvm::enumerate(op->getResults())) {
        auto tensorType = llvm::dyn_cast<RankedTensorType>(result.getType());
        if (!tensorType || !tensorType.getElementType().isFloat())
          continue;

        // Insert after the operation.
        builder.setInsertionPointAfter(op);

        auto loc = op->getLoc();
        probe::ObserveOp::create(builder, loc, result,
                                 static_cast<uint32_t>(opID),
                                 static_cast<uint32_t>(resultID));
        addedAnyObserve = true;
      }
    }

    if (!addedAnyObserve) {
      // No need to report if no probe.observe ops were added
      return;
    }

    worklist.clear();
    // Insert probe.report before all returns, forwarding the report file path
    // to the runtime report function.
    auto reportArgs = builder.getArrayAttr({builder.getStringAttr(reportFile)});
    funcOp.walk([&](func::ReturnOp returnOp) { worklist.push_back(returnOp); });
    for (auto returnOp : worklist) {
      builder.setInsertionPoint(returnOp);
      probe::ReportOp::create(builder, returnOp->getLoc(), reportArgs);
    }
  }
};
} // namespace
} // namespace mlir::vishap
