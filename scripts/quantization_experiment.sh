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
  -s, --seed <int>            RNG seed for quantization (default: 0)
  -h, --help                  Display this help message
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

out_dir="$root_dir/tmp/quant_experiemnts/"
mkdir -p $out_dir

# Hardcoded ImageNet-V2 validation subset used for calibration / comparison
image_dir="$root_dir/tmp/val100"
if [[ ! -d "$image_dir" ]]; then
  echo "Error: image directory does not exist: $image_dir" >&2
  exit 1
fi
# Single image shared by the "vishap" method and by validation/compare.
input_image=$(find "$image_dir" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort | head -1)
if [[ -z "$input_image" ]]; then
  echo "Error: no images found under $image_dir" >&2
  exit 1
fi
echo "Using validation/vishap image: $input_image"
num_images=$(find "$image_dir" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)

quantize_script="$root_dir/scripts/ort_quantize.py"

# Each entry: label|extra ort_quantize.py args (after model path)
# label is used for output filenames, compare --type, and section banners.
quant_methods=(
  "vishap|--method vishap --input-file ${input_image} --seed ${seed}"
  "vishap_agg|--method vishap --input-dir ${image_dir} --num-inputs ${num_images} --seed ${seed}"
  "mixed_0.25|--method mixed --input-file ${input_image} --preload-ratio 0.25 --seed ${seed}"
  "mixed_0.50|--method mixed --input-file ${input_image} --preload-ratio 0.5 --seed ${seed}"
  "mixed_0.75|--method mixed --input-file ${input_image} --preload-ratio 0.75 --seed ${seed}"
  "calibration|--method calibration --input-dir ${image_dir} --seed ${seed}"
  "random|--method random --seed ${seed}"
)
date_string=$(date +%Y%m%d%H%M%S)

######################### Run experiment #########################
for method_spec in "${quant_methods[@]}"; do
  IFS='|' read -r label quant_args <<< "$method_spec"
  echo
  echo "*********************************************************************************"
  echo "Quantize models with ${label}"
  echo "*********************************************************************************"
  for onnx_file in "$input_dir"/*.onnx; do
    if [[ -f "$onnx_file" ]]; then
      echo "*************************$onnx_file*************************"
      filename=$(basename "${onnx_file%.*}")
      out_file="$out_dir/${filename}_${label}.onnx"
      echo python $quantize_script "$onnx_file" --output-model "$out_file" $quant_args --perf-out "perf_info_${date_string}.csv"
      python $quantize_script "$onnx_file" --output-model "$out_file" $quant_args --perf-out "perf_info_${date_string}.csv"
      echo
    fi
  done
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
    for method_spec in "${quant_methods[@]}"; do
      IFS='|' read -r label _ <<< "$method_spec"
      quant_file="$out_dir/${filename}_${label}.onnx"
      echo python $compare_script "$onnx_file" "$quant_file" --type "$label" --input "$input_image" -o "quant_results_${date_string}.csv"
      python $compare_script "$onnx_file" "$quant_file" --type "$label" --input "$input_image" -o "quant_results_${date_string}.csv"
    done
    echo
  fi
done
