#ifndef VISHAP_TRANSFORMS_PASSES
#define VISHAP_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def AnnotateDistributionsPass
    : Pass<"annotate-distributions", "mlir::func::FuncOp"> {
  let summary = "Annotate IR with distribution information from propagation analysis";
}

#endif // VISHAP_TRANSFORMS_PASSES
