#include "vishap/Passes/TosaValueProfilerPass.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace vishap {

struct TosaValueProfilerPass 
    : public mlir::PassWrapper<TosaValueProfilerPass, mlir::OperationPass<mlir::func::FuncOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TosaValueProfilerPass)

  llvm::StringRef getArgument() const final { return "tosa-value-profiler"; }
  llvm::StringRef getDescription() const final { return "Profiles min/max values per slice of TOSA tensors."; }

  void runOnOperation() override {
    mlir::func::FuncOp f = getOperation();

    f.walk([&](mlir::tosa::ConstOp constOp) {
      // 1. Corrected to getValuesAttr()
      auto attr = constOp.getValuesAttr();
      
      if (auto denseAttr = llvm::dyn_cast_if_present<mlir::DenseElementsAttr>(attr)) {
        auto type = llvm::cast<mlir::ShapedType>(denseAttr.getType());
        
        if (!type.hasRank() || type.getRank() < 2) return;

        llvm::outs() << "Profiling Constant Tensor (Analyzing slices based on Dim 0)\n";
        
        int64_t numSlices = type.getDimSize(0); 
        int64_t elementsPerSlice = type.getNumElements() / numSlices;
        
        if (type.getElementType().isF32()) {
          // 2. Added 'template' keyword to fix the primary-expression error
          auto values = denseAttr.getValues<float>();

          for (int64_t i = 0; i < numSlices; ++i) {
            float minVal = std::numeric_limits<float>::max();
            float maxVal = std::numeric_limits<float>::lowest();

            for (int64_t j = 0; j < elementsPerSlice; ++j) {
              float val = *(values.begin() + (i * elementsPerSlice) + j);
              minVal = std::min(minVal, val);
              maxVal = std::max(maxVal, val);
            }
            llvm::outs() << "  Slice " << i << ": Min=" << minVal << ", Max=" << maxVal << "\n";
          }
        }
      }
    });
  }
};

std::unique_ptr<mlir::Pass> createValueProfilerPass() {
  return std::make_unique<TosaValueProfilerPass>();
}

} // namespace vishap
