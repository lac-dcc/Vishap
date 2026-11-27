#include "Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::vishap {
#define GEN_PASS_DEF_SEQUENTIALIZENESTEDPARALLELLOOPSPASS
#include "Transforms/Passes.h.inc"

namespace {
class SequentializeNestedParallelLoopsPass
    : public impl::SequentializeNestedParallelLoopsPassBase<
          SequentializeNestedParallelLoopsPass> {
public:
  void runOnOperation() override {
    // Collect nested scf.parallel loops first: rewriting while walking would
    // invalidate the traversal.
    llvm::SmallVector<scf::ParallelOp> nested;
    getOperation().walk([&](scf::ParallelOp op) {
      if (op->getParentOfType<scf::ParallelOp>())
        nested.push_back(op);
    });

    IRRewriter rewriter(&getContext());
    for (scf::ParallelOp op : nested) {
      // Loops with results (reductions) are not supported by the upstream
      // rewrite. Leave them parallel.
      if (failed(scf::parallelForToNestedFors(rewriter, op)))
        continue;
    }
  }
};
} // namespace
} // namespace mlir::vishap
