#!/bin/bash
# run_pipeline.sh — Compile and evaluate baseline vs optimized using mlir-opt + mlir-runner.
#
# The pipeline is:
#   1. Lower and run the baseline program, measuring execution time
#   2. Apply optimization passes to the source
#   3. Lower and run the optimized program, measuring execution time
#   4. Report results side by side
#
# Usage:
#   ./run_pipeline.sh

set -e

# --- Paths ---
MLIR_OPT=mlir-opt
MLIR_RUNNER=mlir-runner
RUNNER_UTILS=/home/fernando/Program/llvm-project/build/lib/libmlir_runner_utils.so
C_RUNNER_UTILS=/home/fernando/Program/llvm-project/build/lib/libmlir_c_runner_utils.so

SOURCE="conv2d.mlir"
BASELINE_LOWERED="baseline_lowered.mlir"
OPTIMIZED_LOWERED="optimized_lowered.mlir"
OPTIMIZED_SOURCE="conv2d_opt_pass.mlir"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# --- Baseline lowering pipeline ---
# Uses scf loops (no affine), so no tiling is applied.
lower_baseline() {
  local input=$1
  local output=$2
  $MLIR_OPT "$input" \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --convert-linalg-to-loops \
    --lower-affine \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-vector-to-llvm \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    --finalize-memref-to-llvm \
    --reconcile-unrealized-casts \
    -o "$output"
}

# --- Optimized lowering pipeline ---
# Uses affine loops so that --affine-loop-tile can tile the loop nest
# into cache-friendly chunks before lowering further.
lower_optimized() {
  local input=$1
  local output=$2
  $MLIR_OPT "$input" \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --convert-linalg-to-affine-loops \
    --affine-loop-tile="tile-size=8" \
    --lower-affine \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-vector-to-llvm \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    --finalize-memref-to-llvm \
    --reconcile-unrealized-casts \
    -o "$output"
}

# --- Helper: time a command in ms ---
time_command() {
  local var=$1
  shift
  local start end
  start=$(date +%s%N)
  "$@"
  end=$(date +%s%N)
  eval "$var=$(( (end - start) / 1000000 ))"
}

# --- Check required files ---
echo -e "${CYAN}=== Checking required files ===${RESET}"
if [ ! -f "$SOURCE" ]; then
  echo -e "${RED}Error: '$SOURCE' not found.${RESET}"
  exit 1
fi
echo "OK: $SOURCE found."

# --- Step 1: Lower baseline ---
echo ""
echo -e "${CYAN}=== Step 1: Lowering baseline ===${RESET}"
echo "Pipeline: bufferize -> linalg-to-loops -> lower-affine -> llvm"
time_command BASELINE_LOWER_MS lower_baseline "$SOURCE" "$BASELINE_LOWERED"
echo "Written: $BASELINE_LOWERED (lower time: ${BASELINE_LOWER_MS}ms)"

# --- Step 2: Run baseline ---
echo ""
echo -e "${CYAN}=== Step 2: Running baseline ===${RESET}"
echo -e "${BOLD}Baseline output:${RESET}"
time_command BASELINE_RUN_MS \
  $MLIR_RUNNER "$BASELINE_LOWERED" \
  --entry-point-result=void \
  --shared-libs="$RUNNER_UTILS" \
  --shared-libs="$C_RUNNER_UTILS"
echo "(run time: ${BASELINE_RUN_MS}ms)"

# --- Step 3: Apply optimization pass ---
# The source is passed through unchanged here — the optimization is applied
# in the lowering pipeline itself (lower_optimized), where we use
# --convert-linalg-to-affine-loops followed by --affine-loop-tile.
#
# --affine-loop-tile="tile-size=8" breaks every loop in the nest into
# tiles of size 8, producing a tiled loop nest that is cache-friendlier:
#
#   Before:  for i in [0..126): for j in [0..126): ...
#   After:   for ii in [0..126) step 8:
#              for jj in [0..126) step 8:
#                for i in [ii, ii+8): for j in [jj, jj+8): ...
#
# This keeps the working set of each inner tile small enough to stay
# in L1/L2 cache, reducing memory traffic significantly for large tensors.
#
# Replace --affine-loop-tile with your own pass in lower_optimized()
# when you are ready to test your custom allocator.
echo ""
echo -e "${CYAN}=== Step 3: Applying optimization ===${RESET}"
echo "Optimization: --affine-loop-tile=tile-size=8 (applied during lowering)"
echo "Strategy: linalg -> affine loops -> tile -> lower to llvm"
echo "(swap --affine-loop-tile for your pass in lower_optimized() when ready)"

# --- Step 4: Lower optimized ---
echo ""
echo -e "${CYAN}=== Step 4: Lowering optimized ===${RESET}"
echo "Pipeline: bufferize -> linalg-to-affine-loops -> tile -> lower-affine -> llvm"
time_command OPTIMIZED_LOWER_MS lower_optimized "$SOURCE" "$OPTIMIZED_LOWERED"
echo "Written: $OPTIMIZED_LOWERED (lower time: ${OPTIMIZED_LOWER_MS}ms)"

# --- Step 5: Run optimized ---
echo ""
echo -e "${CYAN}=== Step 5: Running optimized ===${RESET}"
echo -e "${BOLD}Optimized output:${RESET}"
time_command OPTIMIZED_RUN_MS \
  $MLIR_RUNNER "$OPTIMIZED_LOWERED" \
  --entry-point-result=void \
  --shared-libs="$RUNNER_UTILS" \
  --shared-libs="$C_RUNNER_UTILS"
echo "(run time: ${OPTIMIZED_RUN_MS}ms)"

# --- Step 6: Report ---
echo ""
echo -e "${CYAN}========================================${RESET}"
echo -e "${BOLD}           RESULTS SUMMARY              ${RESET}"
echo -e "${CYAN}========================================${RESET}"
printf "%-28s %10s %10s\n" "" "Baseline" "Optimized"
printf "%-28s %10s %10s\n" "----------------------------" "--------" "---------"
printf "%-28s %9sms %9sms\n" "Lowering time"  "$BASELINE_LOWER_MS"  "$OPTIMIZED_LOWER_MS"
printf "%-28s %9sms %9sms\n" "Run time"       "$BASELINE_RUN_MS"    "$OPTIMIZED_RUN_MS"

if [ "$BASELINE_RUN_MS" -gt 0 ] && [ "$OPTIMIZED_RUN_MS" -gt 0 ]; then
  SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $BASELINE_RUN_MS / $OPTIMIZED_RUN_MS}")
  echo ""
  if awk "BEGIN {exit !($SPEEDUP > 1.0)}"; then
    echo -e "${GREEN}Speedup: ${SPEEDUP}x  (optimized is faster)${RESET}"
  elif awk "BEGIN {exit !($SPEEDUP < 1.0)}"; then
    echo -e "${RED}Speedup: ${SPEEDUP}x  (optimized is slower — revisit your pass!)${RESET}"
  else
    echo -e "Speedup: ${SPEEDUP}x  (no change)"
  fi
else
  echo ""
  echo "Note: run time too fast to measure precisely. Consider a larger benchmark."
fi
echo -e "${CYAN}========================================${RESET}"

# --- Cleanup ---
echo ""
echo "Cleaning up intermediate files..."
rm -f "$BASELINE_LOWERED" "$OPTIMIZED_LOWERED" "$OPTIMIZED_SOURCE"
echo "Done."
