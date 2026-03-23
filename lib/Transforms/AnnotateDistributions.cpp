#include "Analysis/DistributionAnalysis.h"
#include "Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::vishap {
#define GEN_PASS_DEF_ANNOTATEDISTRIBUTIONSPASS
#include "Transforms/Passes.h.inc"

namespace {
class AnnotateDistributionsPass
    : public impl::AnnotateDistributionsPassBase<
          AnnotateDistributionsPass> {
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
        auto distOrFailure = analysis.getDistribution(result);
        if (failed(distOrFailure)) {
          signalPassFailure();
          return;
        }

        auto distArray = distributionToArrayAttr(builder, *(*distOrFailure));
        distributions.push_back(ArrayAttr::get(ctx, distArray));
      }

      op->setAttr(kDistributionAttrName, ArrayAttr::get(ctx, distributions));
    }
  }
};
} // namespace
} // namespace mlir::vishap
