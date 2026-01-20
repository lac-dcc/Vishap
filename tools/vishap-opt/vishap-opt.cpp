/**
 * @file vishap-opt.cpp
 * @brief Standalone driver for VISHAP TOSA passes.
 */

#include "vishap/Passes/TosaValueProfilerPass.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

using namespace mlir;
using namespace vishap;

/// Print an error message and exit.
[[noreturn]] static void die(llvm::StringRef msg) {
  llvm::errs() << msg << "\n";
  std::exit(1);
}

/// Input file CLI option.
static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional,
    llvm::cl::desc("<input mlir file>"),
    llvm::cl::init("-"));

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "VISHAP: TOSA value profiling and sparse analysis\n");

  // 1. Dialect registration: TOSA and Func are essential here.
  DialectRegistry registry;
  registry.insert<
      tosa::TosaDialect,
      func::FuncDialect
  >();

  MLIRContext context(registry);
  // Ensure the dialects are loaded for the parser.
  context.loadAllAvailableDialects();

  // 2. Parse input MLIR file.
  llvm::SourceMgr sourceMgr;
  auto buffer = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!buffer)
    die("Error: could not open input file " + inputFilename);

  sourceMgr.AddNewSourceBuffer(std::move(*buffer), llvm::SMLoc());
  SourceMgrDiagnosticHandler diagHandler(sourceMgr, &context);

  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module)
    die("Error: failed to parse MLIR module");

  // 3. Build pass pipeline.
  PassManager pm(&context);

  // We add our new value profiler pass to the pipeline.
  // This will analyze tensors like E and its slices.
  pm.addNestedPass<func::FuncOp>(createValueProfilerPass());

  // 4. Run the pipeline.
  if (failed(pm.run(*module))) {
    llvm::errs() << "Error: Pass execution failed\n";
    return 1;
  }

  // 5. Print the module (optional, but standard for -opt tools).
  module->print(llvm::outs());

  return 0;
}
