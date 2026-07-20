#include "Dialect/Probe/IR/Probe.h"
#include "Dialect/Probe/Transforms/Passes.h"
#include "Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::probe::ProbeDialect>();
  mlir::registerAllDialects(registry);
  // Registers the stablehlo, chlo and vhlo dialects.
  mlir::stablehlo::registerAllDialects(registry);

  mlir::vishap::registerVishapPasses();
  mlir::probe::registerProbePasses();
  mlir::stablehlo::registerPasses();
  mlir::stablehlo::registerOptimizationPasses();
  mlir::stablehlo::registerStablehloLinalgTransformsPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Vishap - tensor values distribution propagation\n",
      registry));
}
