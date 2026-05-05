#include "Analysis/DistributionAnalysis.h"
#include "Dialect/Probe/IR/Probe.h"
#include "Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include <cstdint>

namespace mlir::vishap {
#define GEN_PASS_DEF_ADDPROBECALLSPASS
#include "Transforms/Passes.h.inc"

namespace {
class AddProbeCallsPass
    : public impl::AddProbeCallsPassBase<AddProbeCallsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    auto funcOp = dyn_cast<func::FuncOp>(getOperation());
    if (!funcOp)
      return;

    std::vector<Operation *> worklist;
    funcOp.walk([&](Operation *op) {
      // Only consider ops with vishap distribution attribute
      if (op->hasAttr(kDistributionAttrName)) {
        worklist.push_back(op);
      }
    });

    auto *ctx = &getContext();
    OpBuilder builder(ctx);
    for (auto [opID, op] : llvm::enumerate(worklist)) {
      for (auto [resultID, result] : llvm::enumerate(op->getResults())) {
        auto tensorType = llvm::dyn_cast<RankedTensorType>(result.getType());
        if (!tensorType || !tensorType.getElementType().isFloat())
          continue;

        // Insert after the operation.
        builder.setInsertionPointAfter(op);

        auto loc = op->getLoc();
        builder.create<probe::ObserveOp>(loc, result,
                                         static_cast<uint32_t>(opID),
                                         static_cast<uint32_t>(resultID));
      }
    }
  }
};
} // namespace
} // namespace mlir::vishap
