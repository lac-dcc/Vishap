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
    >,
    Option<
      "perAxis",
      "per-axis",
      "bool",
      /*default=*/"true",
      "Track one distribution per index of every tensor axis (per-axis "
      "marginals) in addition to the per-tensor summary. When false, the "
      "analysis degenerates to the per-tensor domain."
    >,
    Option<
      "annotateAxes",
      "annotate-axes",
      "bool",
      /*default=*/"false",
      "Also attach the per-axis slice distributions as the "
      "vishap.axis_distribution attribute (one list per result, one list per "
      "axis, empty for axes without per-slice information). Debug output."
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
