#!/bin/bash

######################### Config #########################

set -e
source "$(dirname "${BASH_SOURCE[0]}")/env_check.sh"

ROOT_DIR="$(dirname $(dirname "${BASH_SOURCE[0]}"))"
TMP_DIR="$ROOT_DIR/tmp"
MODELS_TMP_DIR="$TMP_DIR/models"

######################### Arg parsing #########################

# Default values
EMIT_BYTECODE=false
FUNCTION_NAME="main"
ONNX_MODELS_DIR=""
INPUT_DESC=""
OUTPUT_DESC=""

# Usage function
usage() {
  cat <<EOF
Usage: $(basename "$0") -d <onnx-models-dir> [OPTIONS]

Required arguments:
  -d, --onnx-models-dir <dir>   Directory containing ONNX model files

Optional arguments:
  -b, --emit-bytecode           Save intermediate MLIR models as bytecode instead of text
  -f, --function <name>         Name of entry function in models (default: main)
  -i, --input <desc>            Comma separated list of input tensor shapes and types (e.g., 1x2x3xf32,4x5xi8)
  -o, --output <desc>           Comma separated list of output tensor shapes and types (e.g., 1x2xui8,3x4x5xf32)
  -h, --help                    Display this help message

Example:
  $(basename "$0") -d ./models -f my_function -b
EOF
  exit "${1:-0}"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--onnx-models-dir)
      ONNX_MODELS_DIR="$2"
      shift 2
      ;;
    -b|--emit-bytecode)
      EMIT_BYTECODE=true
      shift
      ;;
    -f|--function)
      FUNCTION_NAME="$2"
      shift 2
      ;;
    -i|--input)
      INPUT_DESC="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT_DESC="$2"
      shift 2
      ;;
    -h|--help)
      usage 0
      ;;
    *)
      echo "Error: Unknown option $1" >&2
      usage 1
      ;;
  esac
done

# Validate required arguments
if [[ -z "$ONNX_MODELS_DIR" ]]; then
  echo "Error: --onnx-models-dir is required" >&2
  usage 1
fi

if [[ -z "$INPUT_DESC" ]]; then
  echo "Error: --input is required" >&2
  usage 1
fi

if [[ -z "$OUTPUT_DESC" ]]; then
  echo "Error: --output is required" >&2
  usage 1
fi

if [[ ! -d "$ONNX_MODELS_DIR" ]]; then
  echo "Error: ONNX models directory does not exist: $ONNX_MODELS_DIR" >&2
  exit 1
fi

######################### Run experiment #########################

mkdir -p $TMP_DIR
mkdir -p $MODELS_TMP_DIR

# Determine output extension based on bytecode flag
OUTPUT_EXT="mlir"
if [[ "$EMIT_BYTECODE" == true ]]; then
  OUTPUT_EXT="mlirbc"
  ONNX_2_MLIR_EXTRA_ARGS="--emit-bytecode"
fi

NUM_ONNX_MODELS=$(ls -1 "$ONNX_MODELS_DIR"/*.onnx 2>/dev/null | wc -l)
NUM_EXISTING_MODELS=$(ls -1 "$MODELS_TMP_DIR"/*.{mlir,mlirbc} 2>/dev/null | wc -l)

echo "*********************************************************************************"
echo "Convert ONNX models to MLIR"
echo "*********************************************************************************"
if [[ $NUM_EXISTING_MODELS -eq $NUM_ONNX_MODELS ]]; then
  echo "All ONNX models have already been converted to MLIR. Skipping conversion step."
else
  ONNX_2_MLIR="$ROOT_DIR/scripts/onnx2mlir.py"
  for onnx_file in "$ONNX_MODELS_DIR"/*.onnx; do
    if [[ -f "$onnx_file" ]]; then
      echo "Converting $onnx_file to MLIR"
      python $ONNX_2_MLIR "$onnx_file" -o "$MODELS_TMP_DIR/$(basename "${onnx_file%.*}.$OUTPUT_EXT")" $ONNX_2_MLIR_EXTRA_ARGS
    fi
  done
fi

echo
echo "*********************************************************************************"
echo "Run models and compare results"
echo "*********************************************************************************"
RUN_SCRIPT="$ROOT_DIR/scripts/run.py"
for mlir_file in "$MODELS_TMP_DIR"/*.{mlir,mlirbc}; do
  if [[ -f "$mlir_file" ]]; then
    echo "*************************$mlir_file*************************"
    python $RUN_SCRIPT "$mlir_file" --input=$INPUT_DESC --output=$OUTPUT_DESC --function=$FUNCTION_NAME
    echo
  fi
done

