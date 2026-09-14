#!/bin/bash

######################### Config #########################

set -e
source "$(dirname "${BASH_SOURCE[0]}")/env_check.sh"

root_dir="$(dirname $(dirname "${BASH_SOURCE[0]}"))"

######################### Arg parsing #########################

# Default values
input_dir=""

# Usage function
usage() {
  cat <<EOF
Usage: $(basename "$0") -d <input-dir> [OPTIONS]

Required arguments:
  -d, --input-dir <dir>       Directory containing input ONNX files

Optional arguments:
  -s, --seed <int>            RNG seed for quantization (default: 42)
  -h, --help                  Display this help message

For each model, discovers mixed float ONNX op types and quantizes once per
type: that type is analyzed with Vishap, all other non-constant float ops
are preloaded from ONNX Runtime.
EOF
  exit "${1:-0}"
}

seed=42

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--input-dir)
      input_dir="$2"
      shift 2
      ;;
    -s|--seed)
      seed="$2"
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
  echo "Error: ONNX models directory does not exist: $input_dir" >&2
  exit 1
fi

out_dir="$root_dir/tmp/op_type_quant_experiments/"
mkdir -p $out_dir

# Disjoint ImageNet-V2 subsets: calib for quantization, val for compare only.
calib_dir="$root_dir/tmp/val200/split0"
val_dir="$root_dir/tmp/val200/split1"
if [[ ! -d "$calib_dir" ]]; then
  echo "Error: calibration directory does not exist: $calib_dir" >&2
  exit 1
fi
if [[ ! -d "$val_dir" ]]; then
  echo "Error: validation directory does not exist: $val_dir" >&2
  exit 1
fi
# Single calib image for mixed op-type runs (must not appear in val_dir).
input_image=$(find "$calib_dir" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort | head -1)
if [[ -z "$input_image" ]]; then
  echo "Error: no images found under $calib_dir" >&2
  exit 1
fi
echo "Using calibration image (mixed op-type): $input_image"
echo "Using validation set for compare: $val_dir"

quantize_script="$root_dir/scripts/ort_quantize.py"
date_string=$(date +%Y%m%d%H%M%S)

sanitize_op_type() {
  # Match ort_quantize._sanitize_op_type_label: replace / and : with _.
  echo "${1//\//_}" | tr ':' '_'
}

######################### Run experiment #########################
for onnx_file in "$input_dir"/*.onnx; do
  if [[ -f "$onnx_file" ]]; then
    echo
    echo "*********************************************************************************"
    echo "Discover op types in $onnx_file"
    echo "*********************************************************************************"
    filename=$(basename "${onnx_file%.*}")
    types_file="$out_dir/${filename}_op_types.txt"
    echo python $quantize_script "$onnx_file" --list-op-types
    python $quantize_script "$onnx_file" --list-op-types > "$types_file"
    if [[ ! -s "$types_file" ]]; then
      echo "Warning: no mixed float op types found in $onnx_file" >&2
      continue
    fi
    echo "Op types:"
    cat "$types_file"

    while IFS= read -r op_type || [[ -n "$op_type" ]]; do
      [[ -z "$op_type" ]] && continue
      label="mixed_op_$(sanitize_op_type "$op_type")"
      echo
      echo "*********************************************************************************"
      echo "Quantize $onnx_file with Vishap on ${op_type}"
      echo "*********************************************************************************"
      out_file="$out_dir/${filename}_${label}.onnx"
      echo python $quantize_script "$onnx_file" --output-model "$out_file" \
        --method mixed --vishap-op-type "$op_type" --input-file "${input_image}" \
        --seed "${seed}" --perf-out "op_type_perf_info_${date_string}.csv"
      python $quantize_script "$onnx_file" --output-model "$out_file" \
        --method mixed --vishap-op-type "$op_type" --input-file "${input_image}" \
        --seed "${seed}" --perf-out "op_type_perf_info_${date_string}.csv"
    done < "$types_file"
  fi
done


compare_script="$root_dir/scripts/compare_models.py"
echo
echo "*********************************************************************************"
echo "Compare quantized models with float"
echo "*********************************************************************************"
for onnx_file in "$input_dir"/*.onnx; do
  if [[ -f "$onnx_file" ]]; then
    echo "*************************$onnx_file*************************"
    filename=$(basename "${onnx_file%.*}")
    types_file="$out_dir/${filename}_op_types.txt"
    if [[ ! -s "$types_file" ]]; then
      continue
    fi
    while IFS= read -r op_type || [[ -n "$op_type" ]]; do
      [[ -z "$op_type" ]] && continue
      label="mixed_op_$(sanitize_op_type "$op_type")"
      quant_file="$out_dir/${filename}_${label}.onnx"
      echo python $compare_script "$onnx_file" "$quant_file" --type "$label" --input "$val_dir" -o "op_type_quant_results_${date_string}.csv"
      python $compare_script "$onnx_file" "$quant_file" --type "$label" --input "$val_dir" -o "op_type_quant_results_${date_string}.csv"
    done < "$types_file"
    echo
  fi
done
