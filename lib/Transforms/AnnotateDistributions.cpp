#include "Analysis/DistributionAnalysis.h"
#include "Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"

namespace mlir::vishap {
#define GEN_PASS_DEF_ANNOTATEDISTRIBUTIONSPASS
#include "Transforms/Passes.h.inc"

namespace {
class AnnotateDistributionsPass
    : public impl::AnnotateDistributionsPassBase<AnnotateDistributionsPass> {
private:
  LogicalResult parseInputDists(func::FuncOp func, Builder &builder) {
    if (inputDists.empty() || func->hasAttr(kArgsDistributionAttrName)) {
      // Do not parse if empty, and do not override existing attribute.
      return success();
    }

    constexpr char inputSep[] = ";";
    constexpr char sep = ',';

    auto ctx = builder.getContext();
    llvm::SmallVector<Attribute> funcAttrs;
    for (llvm::StringRef inputDist : llvm::split(inputDists, inputSep)) {
      if (inputDist.empty()) {
        continue;
      }

      llvm::SmallVector<Attribute, 4> inputAttrs;
      for (auto valueStr : llvm::split(inputDist, sep)) {
        double value;
        if (!valueStr.getAsDouble(value)) {
          inputAttrs.push_back(builder.getF64FloatAttr(value));
        } else {
          return func.emitError() << "Failed to parse distribution value: \""
                                  << valueStr << "\"";
        }
      }

      if (inputAttrs.size() != 4) {
        return func.emitError() << "Expected 4 values in distribution, but got "
                                << inputAttrs.size() << "(" << inputDist << ")";
      }

      funcAttrs.push_back(ArrayAttr::get(ctx, inputAttrs));
    }

    if (func.getNumArguments() != funcAttrs.size()) {
      return func.emitError()
             << "Expected " << func.getNumArguments()
             << " distribution(s), but got " << funcAttrs.size();
    }

    func->setAttr(kArgsDistributionAttrName, ArrayAttr::get(ctx, funcAttrs));

    return success();
  }

public:
  using Base::Base;

  void runOnOperation() override {
    auto func = getOperation();
    auto ctx = &getContext();
    Builder builder(ctx);

    if (failed(parseInputDists(func, builder))) {
      return signalPassFailure();
    }

    DistributionAnalysis analysis;
    if (failed(analysis.run(func))) {
      return signalPassFailure();
    }

    // Traverse analyzed operations and annotate with distribution attribute
    for (auto *op : analysis.getAnalyzedOperations()) {
      llvm::SmallVector<Attribute, 3> distributions;
      for (auto result : op->getResults()) {
        auto distOrFailure = analysis.getDistribution(result);
        if (failed(distOrFailure)) {
          return signalPassFailure();
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
