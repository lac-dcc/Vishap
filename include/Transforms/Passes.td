#ifndef VISHAP_TRANSFORMS_PASSES
#define VISHAP_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def AnnotateDistributionsPass
    : Pass<"annotate-distributions", "mlir::func::FuncOp"> {
  let summary = "Annotate IR with distribution information from propagation analysis";
  let options = [
    Option<
      "inputDists",
      "input-distributions",
      "std::string",
      /*default=*/"",
      "Semicolon-separated list of distributions, each of which is a comma-separated list of 4 numeric values."
    >
  ];
}

#endif // VISHAP_TRANSFORMS_PASSES
