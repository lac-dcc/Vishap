#!/bin/bash

######################### Config #########################

set -e
source "$(dirname "${BASH_SOURCE[0]}")/env_check.sh"

root_dir="$(dirname $(dirname "${BASH_SOURCE[0]}"))"

######################### Arg parsing #########################

# Default values
function_name="main"
input_dir=""
input_desc=""
output_desc=""

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
      input_dir="$2"
      shift 2
      ;;
    -f|--function)
      function_name="$2"
      shift 2
      ;;
    -i|--input)
      input_desc="$2"
      shift 2
      ;;
    -o|--output)
      output_desc="$2"
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
if [[ -z "$input_desc" ]]; then
  echo "Error: --input is required" >&2
  usage 1
fi

if [[ -z "$output_desc" ]]; then
  echo "Error: --output is required" >&2
  usage 1
fi

if [[ -z "$input_dir" ]]; then
  echo "Error: --input-dir is required" >&2
  usage 1
fi

if [[ ! -d "$input_dir" ]]; then
  echo "Error: MLIR models directory does not exist: $input_dir" >&2
  exit 1
fi

######################### Run experiment #########################
echo
echo "*********************************************************************************"
echo "Run models and compare results"
echo "*********************************************************************************"
run_script="$root_dir/scripts/run.py"
for mlir_file in "$input_dir"/*.{mlir,mlirbc}; do
  if [[ -f "$mlir_file" ]]; then
    echo "*************************$mlir_file*************************"
    python $run_script "$mlir_file" --input=$input_desc --output=$output_desc --function=$function_name
    echo
  fi
done

