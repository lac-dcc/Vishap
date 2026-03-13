#!/bin/bash
# run_conv2d.sh — Compile and run the conv2d benchmark using either mlir or iree.
# Usage:
#   ./run_conv2d.sh mlir
#   ./run_conv2d.sh iree

# --- Paths ---
MLIR_SOURCE="conv2d_llvm.mlir"
IREE_SOURCE="conv2d_iree.mlir"
MLIR_OUTPUT="conv2d_lowered.mlir"
IREE_OUTPUT="conv2d.vmfb"
MLIR_RUNNER_UTILS=/home/fernando/Program/llvm-project/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS=/home/fernando/Program/llvm-project/build/lib/libmlir_c_runner_utils.so

# --- Helpers ---
usage() {
  echo "Usage: $0 [mlir|iree]"
  exit 1
}

run_mlir() {
  echo "=== Compiling with mlir-opt ==="
  mlir-opt "$MLIR_SOURCE" \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --convert-linalg-to-loops \
    --lower-affine \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    --finalize-memref-to-llvm \
    --reconcile-unrealized-casts \
    -o "$MLIR_OUTPUT"

  if [ $? -ne 0 ]; then
    echo "Error: mlir-opt compilation failed."
    exit 1
  fi

  echo "=== Running with mlir-runner ==="
  mlir-runner "$MLIR_OUTPUT" \
    --entry-point-result=void \
    --shared-libs="$MLIR_RUNNER_UTILS" \
    --shared-libs="$MLIR_C_RUNNER_UTILS"

  if [ $? -ne 0 ]; then
    echo "Error: mlir-runner execution failed."
    exit 1
  fi
}

run_iree() {
  echo "=== Compiling with iree-compile ==="
  iree-compile "$IREE_SOURCE" \
    --iree-hal-target-backends=llvm-cpu \
    -o "$IREE_OUTPUT"

  if [ $? -ne 0 ]; then
    echo "Error: iree-compile compilation failed."
    exit 1
  fi

  echo "=== Running with iree-run-module ==="
  iree-run-module \
    --module="$IREE_OUTPUT" \
    --function=main

  if [ $? -ne 0 ]; then
    echo "Error: iree-run-module execution failed."
    exit 1
  fi
}

# --- Main ---
if [ $# -ne 1 ]; then
  usage
fi

case "$1" in
  mlir) run_mlir ;;
  iree) run_iree ;;
  *)    usage ;;
esac
