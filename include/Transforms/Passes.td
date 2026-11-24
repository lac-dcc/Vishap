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
      "Semicolon-separated list of distributions, each of which is either a "
      "comma-separated list of 4 numeric values or 'unit' for non-float args."
    >,
    Option<
      "perAxis",
      "per-axis",
      "bool",
      /*default=*/"true",
      "Track one distribution per index of every tensor axis in addition to "
      "the per-tensor summary. When false, the analysis degenerates to the "
      "per-tensor domain."
    >,
    Option<
      "annotateAxes",
      "annotate-axes",
      "bool",
      /*default=*/"false",
      "Also attach the per-axis slice distributions as the "
      "vishap.axis_distribution attribute (one list per result, one list per "
      "axis, empty for axes without per-slice information). Used for "
      "debuggging."
    >,
    Option<
      "constantsOnly",
      "constants-only",
      "bool",
      /*default=*/"false",
      "Only compute and annotate distributions for constants and "
      "constant-derived operations, skipping function arguments and dynamic "
      "operations."
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
    >,
    Option<
      "reportFile",
      "report-file",
      "std::string",
      /*default=*/"\"vishap_probe_report.csv\"",
      "Path of the CSV report file written by the probe runtime. Forwarded "
      "to the runtime report function via the reportArgs attribute of "
      "probe.report."
    >,
    Option<
      "observeFuncArgs",
      "observe-func-args",
      "bool",
      /*default=*/"false",
      "Also insert probe.observe calls for floating-point ranked-tensor "
      "function arguments (using opID = -1 and resultID = argument index)."
    >,
    Option<
      "skipRedundantOps",
      "skip-redundant-ops",
      "bool",
      /*default=*/"false",
      "When all-float-ops is enabled, skip probing pure view operations "
      "(broadcast_in_dim, reshape, transpose) and constant-derived convert "
      "operations."
    >
  ];
}

def SequentializeNestedParallelLoopsPass
    : Pass<"sequentialize-nested-parallel-loops", "mlir::func::FuncOp"> {
  let summary = "Turn scf.parallel loops nested in other scf.parallel into scf.for";
  let description = [{
    Keeps the outermost `scf.parallel` of every loop nest and rewrites each
    `scf.parallel` that is (transitively) nested inside another one into
    sequential nested `scf.for` loops. Run it before `convert-scf-to-openmp`,
    which would otherwise turn every nested `scf.parallel` into a nested
    `omp.parallel` region. Those nested regions are executed by a single thread
    but still pay a runtime fork/join for each outer iteration, and they hide
    the inner loop from LLVM's vectorizer.
  }];
  let dependentDialects = ["mlir::scf::SCFDialect"];
}

#endif // VISHAP_TRANSFORMS_PASSES_TD
