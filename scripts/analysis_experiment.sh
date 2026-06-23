#!/bin/bash

######################### Config #########################

set -e
source "$(dirname "${BASH_SOURCE[0]}")/env_check.sh"

ROOT_DIR="$(dirname $(dirname "${BASH_SOURCE[0]}"))"

######################### Arg parsing #########################

# Default values
FUNCTION_NAME="main"
INPUT_DIR=""
INPUT_DESC=""
OUTPUT_DESC=""

# Usage function
usage() {
  cat <<EOF
Usage: $(basename "$0") -d <input-dir> [OPTIONS]

Required arguments:
  -d, --input-dir <dir>       Directory containing input MLIR files

Optional arguments:
  -f, --function <name>       Name of entry function in models (default: main)
  -i, --input <desc>          Comma separated list of input tensor shapes and types (e.g., 1x2x3xf32,4x5xi8)
  -o, --output <desc>         Comma separated list of output tensor shapes and types (e.g., 1x2xui8,3x4x5xf32)
  -h, --help                  Display this help message
EOF
  exit "${1:-0}"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--input-dir)
      INPUT_DIR="$2"
      shift 2
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
if [[ -z "$INPUT_DESC" ]]; then
  echo "Error: --input is required" >&2
  usage 1
fi

if [[ -z "$OUTPUT_DESC" ]]; then
  echo "Error: --output is required" >&2
  usage 1
fi

if [[ -z "$INPUT_DIR" ]]; then
  echo "Error: --input-dir is required" >&2
  usage 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Error: MLIR models directory does not exist: $INPUT_DIR" >&2
  exit 1
fi

######################### Run experiment #########################
echo
echo "*********************************************************************************"
echo "Run models and compare results"
echo "*********************************************************************************"
RUN_SCRIPT="$ROOT_DIR/scripts/run.py"
for mlir_file in "$INPUT_DIR"/*.{mlir,mlirbc}; do
  if [[ -f "$mlir_file" ]]; then
    echo "*************************$mlir_file*************************"
    python $RUN_SCRIPT "$mlir_file" --input=$INPUT_DESC --output=$OUTPUT_DESC --function=$FUNCTION_NAME
    echo
  fi
done

