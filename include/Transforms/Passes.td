#ifndef VISHAP_TRANSFORMS_PASSES_TD
#define VISHAP_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def AnnotateDistributionsPass
    : Pass<"annotate-distributions", "mlir::func::FuncOp"> {
  let summary = "Annotate IR with distribution information from propagation analysis";
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
  let options = [
    Option<
      "inputDists",
      "input-distributions",
      "std::string",
      /*default=*/"",
      "Semicolon-separated list of distributions, each of which is a "
      "comma-separated list of 4 numeric values."
    >
  ];
}

def AddProbeCallsPass
    : Pass<"add-probe-calls", "mlir::func::FuncOp"> {
  let summary = "Add calls for observing tensor values at runtime";
  let dependentDialects = ["mlir::probe::ProbeDialect"];
  let options = [
    Option<
      "allFloatOps",
      "all-float-ops",
      "bool",
      /*default=*/"false",
      "Add probe calls for all operations with floating-point ranked-tensor "
      "results, not only those with vishap.distribution. Constant ops and ops "
      "nested in regions are skipped."
    >
  ];
}

#endif // VISHAP_TRANSFORMS_PASSES_TD
