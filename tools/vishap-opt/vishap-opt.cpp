#include "Dialect/Probe/IR/Probe.h"
#include "Dialect/Probe/Transforms/Passes.h"
#include "Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::probe::ProbeDialect>();
  mlir::registerAllDialects(registry);

  mlir::vishap::registerAddProbeCallsPass();
  mlir::vishap::registerAnnotateDistributionsPass();
  mlir::probe::registerLowerToFuncCallsPass();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Vishap - tensor values distribution propagation\n",
      registry));
}
