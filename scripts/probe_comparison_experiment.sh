#!/bin/bash

######################### Config #########################

set -e
source "$(dirname "${BASH_SOURCE[0]}")/env_check.sh"

root_dir="$(dirname $(dirname "${BASH_SOURCE[0]}"))"

######################### Arg parsing #########################

# Default values
input_dir=""
images_dir="$root_dir/tmp/val200/split1"
num_inputs=""

# Usage function
usage() {
  cat <<EOF
Usage: $(basename "$0") -d <input-dir> [OPTIONS]

Required arguments:
  -d, --input-dir <dir>       Directory containing input MLIR files

Optional arguments:
  -I, --images <dir>          Directory of validation images (default: tmp/val200/split1)
  -n, --num-inputs <int>      Compare only the first N images, in sorted order (default: all)
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
    -I|--images)
      images_dir="$2"
      shift 2
      ;;
    -n|--num-inputs)
      num_inputs="$2"
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
if [[ -z "$input_dir" ]]; then
  echo "Error: --input-dir is required" >&2
  usage 1
fi

if [[ ! -d "$input_dir" ]]; then
  echo "Error: MLIR models directory does not exist: $input_dir" >&2
  exit 1
fi

if [[ ! -d "$images_dir" ]]; then
  echo "Error: validation image directory does not exist: $images_dir" >&2
  exit 1
fi

if [[ -n "$num_inputs" && "$num_inputs" -lt 1 ]]; then
  echo "Error: --num-inputs must be >= 1" >&2
  exit 1
fi

######################### Run experiment #########################
echo
echo "*********************************************************************************"
echo "Compare distribution estimates against probe stats"
echo "*********************************************************************************"
echo "Validation images: $images_dir"
compare_script="$root_dir/scripts/probe_comparison.py"
num_inputs_args=()
if [[ -n "$num_inputs" ]]; then
  num_inputs_args=(--num-inputs "$num_inputs")
fi

for mlir_file in "$input_dir"/*.{mlir,mlirbc}; do
  if [[ -f "$mlir_file" ]]; then
    echo "*************************$mlir_file*************************"
    python "$compare_script" "$mlir_file" \
      --input-dir="$images_dir" \
      "${num_inputs_args[@]}"
    echo
  fi
done
