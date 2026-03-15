// TODO: How do we do this in xnnc???
// TODO: What should be in CMakeLists.txt?
#include "Analysis/DistributionAnalysis.h"
#include "Transforms/Passes.h"

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
  }
};
} // namespace
} // namespace mlir::vishap
