#!/bin/bash

######################### Config #########################

set -e
source "$(dirname "${BASH_SOURCE[0]}")/env_check.sh"

ROOT_DIR="$(dirname $(dirname "${BASH_SOURCE[0]}"))"
TMP_DIR="$ROOT_DIR/tmp"

######################### Arg parsing #########################

# Default values
EMIT_BYTECODE=false
MODELS_DIR=""
OUTPUT_DIR=""

# Usage function
usage() {
  cat <<EOF
Usage: $(basename "$0") -d <models-dir> -o <outputs-dir> [OPTIONS]

Required arguments:
  -d, --models-dir <dir>      Directory containing ONNX model files
  -o, --output-dir <dir>      Directory to place output MLIR files

Optional arguments:
  -b, --emit-bytecode         Save intermediate MLIR models as bytecode instead of text
  -h, --help                  Display this help message
EOF
  exit "${1:-0}"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--models-dir)
      MODELS_DIR="$2"
      shift 2
      ;;
    -o|--output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -b|--emit-bytecode)
      EMIT_BYTECODE=true
      shift
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
if [[ -z "$MODELS_DIR" ]]; then
  echo "Error: --models-dir is required" >&2
  usage 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "Error: --output-dir is required" >&2
  usage 1
fi

if [[ ! -d "$MODELS_DIR" ]]; then
  echo "Error: ONNX models directory does not exist: $MODELS_DIR" >&2
  exit 1
fi

######################### Convert models #########################

mkdir -p $TMP_DIR
mkdir -p $OUTPUT_DIR

# Determine output extension based on bytecode flag
OUTPUT_EXT="mlir"
if [[ "$EMIT_BYTECODE" == true ]]; then
  OUTPUT_EXT="mlirbc"
  ONNX_2_MLIR_EXTRA_ARGS="--emit-bytecode"
fi

echo "*********************************************************************************"
echo "Convert ONNX models to MLIR"
echo "*********************************************************************************"
ONNX_2_MLIR="$ROOT_DIR/scripts/onnx2mlir.py"
for onnx_file in "$MODELS_DIR"/*.onnx; do
  if [[ -f "$onnx_file" ]]; then
    echo "Converting $onnx_file to MLIR"
    python $ONNX_2_MLIR "$onnx_file" -o "$OUTPUT_DIR/$(basename "${onnx_file%.*}.$OUTPUT_EXT")" $ONNX_2_MLIR_EXTRA_ARGS
  fi
done
