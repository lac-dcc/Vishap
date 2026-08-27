#include "Vishap-c/Dialects.h"
#include "Vishap-c/Passes.h"

#include "stablehlo/integrations/c/ChloDialect.h"
#include "stablehlo/integrations/c/StablehloDialect.h"
#include "stablehlo/integrations/c/StablehloPasses.h"
#include "stablehlo/integrations/c/VhloDialect.h"

#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

NB_MODULE(_vishap, m) {
  m.doc() = "Vishap Python bindings";

  // Pass registration is process-global. Do it once at import time so that
  // pass pipelines can be parsed by name from Python.
  vishapRegisterAllPasses();
  mlirRegisterAllStablehloPasses();

  m.def(
      "register_dialects",
      [](MlirContext context, bool load) {
        MlirDialectHandle handles[] = {
            mlirGetDialectHandle__stablehlo__(),
            mlirGetDialectHandle__chlo__(),
            mlirGetDialectHandle__vhlo__(),
            mlirGetDialectHandle__probe__(),
        };
        for (MlirDialectHandle handle : handles) {
          mlirDialectHandleRegisterDialect(handle, context);
          if (load)
            mlirDialectHandleLoadDialect(handle, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true,
      "Register (and optionally load) the StableHLO, CHLO, VHLO, and Probe "
      "dialects in the given context.");
}
