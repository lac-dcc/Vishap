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
  mlir::registerAllDialects(registry);

  mlir::vishap::registerDistributionPropagationPass();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Vishap - tensor values distribution propagation\n",
      registry));
}
