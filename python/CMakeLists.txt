include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `vishap`
# top level package.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(VishapPythonSources)

# onnx_importer.py is written to be relocated. Its relative imports
# (``..ir``, ``..dialects.func``) resolve against the vishap package.
declare_mlir_python_sources(VishapPythonSources.Importers
  ROOT_DIR "${CMAKE_SOURCE_DIR}/external/torch-mlir/python/torch_mlir"
  ADD_TO_PARENT VishapPythonSources
  SOURCES
    extras/onnx_importer.py
)

declare_mlir_python_extension(VishapPythonSources.NanobindExtension
  MODULE_NAME _vishap
  ADD_TO_PARENT VishapPythonSources
  SOURCES
    VishapExtension.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPITransforms
    VishapCAPI
    StablehloCAPI
    ChloCAPI
    VhloCAPI
    TorchMLIRCAPI
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(VishapPythonCAPI
  INSTALL_COMPONENT VishapPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    VishapPythonSources
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
    MLIRPythonSources.Dialects.func
    MLIRPythonSources.ExecutionEngine
    MLIRPythonExtension.RegisterEverything
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(VishapPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES
    VishapPythonSources
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
    MLIRPythonSources.Dialects.func
    MLIRPythonSources.ExecutionEngine
    MLIRPythonExtension.RegisterEverything
  COMMON_CAPI_LINK_LIBS
    VishapPythonCAPI
)
