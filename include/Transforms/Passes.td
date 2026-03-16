#ifndef VISHAP_TRANSFORMS_PASSES
#define VISHAP_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def DistributionPropagationPass
    : Pass<"distribution-propagation", "mlir::func::FuncOp"> {
  let summary = "Propagate distribution information through the IR";
}

#endif // VISHAP_TRANSFORMS_PASSES
