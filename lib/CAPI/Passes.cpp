#include "Vishap-c/Passes.h"

#include "Dialect/Probe/Transforms/Passes.h"
#include "Transforms/Passes.h"

void vishapRegisterAllPasses(void) {
  mlir::vishap::registerVishapPasses();
  mlir::probe::registerProbePasses();
}
