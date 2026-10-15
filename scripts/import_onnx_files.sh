#!/bin/bash

######################### Config #########################

set -e
source "$(dirname "${BASH_SOURCE[0]}")/env_check.sh"

root_dir="$(dirname $(dirname "${BASH_SOURCE[0]}"))"
tmp_dir="$root_dir/tmp"

######################### Arg parsing #########################

# Default values
emit_bytecode=false
models_dir=""
output_dir=""

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
      models_dir="$2"
      shift 2
      ;;
    -o|--output-dir)
      output_dir="$2"
      shift 2
      ;;
    -b|--emit-bytecode)
      emit_bytecode=true
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
if [[ -z "$models_dir" ]]; then
  echo "Error: --models-dir is required" >&2
  usage 1
fi

if [[ -z "$output_dir" ]]; then
  echo "Error: --output-dir is required" >&2
  usage 1
fi

if [[ ! -d "$models_dir" ]]; then
  echo "Error: ONNX models directory does not exist: $models_dir" >&2
  exit 1
fi

######################### Convert models #########################

mkdir -p $tmp_dir
mkdir -p $output_dir

# Determine output extension based on bytecode flag
output_ext="mlir"
extra_args=""
if [[ "$emit_bytecode" == true ]]; then
  output_ext="mlirbc"
  extra_args="--emit-bytecode"
fi

echo "*********************************************************************************"
echo "Convert ONNX models to MLIR"
echo "*********************************************************************************"
onnx2mlir="$root_dir/scripts/onnx2mlir.py"
for onnx_file in "$models_dir"/*.onnx; do
  if [[ -f "$onnx_file" ]]; then
    echo "Converting $onnx_file to MLIR"
    python $onnx2mlir "$onnx_file" -o "$output_dir/$(basename "${onnx_file%.*}.$output_ext")" $extra_args
  fi
done
