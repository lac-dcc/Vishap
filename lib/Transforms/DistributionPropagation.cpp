#include "Analysis/DistributionAnalysis.h"
#include "Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::vishap {
#define GEN_PASS_DEF_DISTRIBUTIONPROPAGATIONPASS
#include "Transforms/Passes.h.inc"

namespace {
class DistributionPropagationPass
    : public impl::DistributionPropagationPassBase<
          DistributionPropagationPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    auto func = getOperation();

    DistributionAnalysis analysis;
    if (failed(analysis.run(func))) {
      signalPassFailure();
    }

    auto ctx = &getContext();
    Builder builder(ctx);
    // Traverse analyzed operations and annotate with distribution attribute
    for (auto *op : analysis.getAnalyzedOperations()) {
      llvm::SmallVector<Attribute, 3> distributions;
      for (auto result : op->getResults()) {
        if (auto dist = analysis.getDistribution(result)) {
          auto distArray = distributionToArrayAttr(builder, *dist);
          distributions.push_back(ArrayAttr::get(ctx, distArray));
        } else {
          distributions.push_back(UnitAttr::get(ctx));
        }
      }

      op->setAttr(kDistributionAttrName, ArrayAttr::get(ctx, distributions));
    }
  }
};
} // namespace
} // namespace mlir::vishap
