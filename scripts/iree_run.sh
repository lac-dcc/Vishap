# #!/bin/bash

if [ -z "$1" ]; then
    >&2 echo "No input file specified"
    exit 1;
fi

set -e

ROOT_DIR="$(dirname $(dirname "${BASH_SOURCE[0]}"))"
TMP_DIR="$ROOT_DIR/tmp"

mkdir -p $TMP_DIR

INPUT="$1"
BASENAME=$(basename "$INPUT" | cut -d. -f1)

iree-import-tflite $INPUT -o $TMP_DIR/$BASENAME.tosa.iree.mlir
iree-opt $BASENAME.tosa.iree.mlir -o $TMP_DIR/$BASENAME.tosa.iree.mlir
iree-compile \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=host \
    $TMP_DIR/$BASENAME.tosa.iree.mlir \
    -o $TMP_DIR/$BASENAME.vmfb
iree-run-module \
    --device=local-task \
    --module=$TMP_DIR/$BASENAME.vmfb \
    --function=main \
    --input="1x224x224x3xf32=@$ROOT_DIR/tests/inputs/coffee_pot.bin" \
    --output=@$TMP_DIR/$BASENAME\_out.npy

CLASS_INDEX=$(python -c "import numpy as np; print(np.load('$TMP_DIR/${BASENAME}_out.npy').argmax())")
CLASS_NAME=$(sed -n "$((CLASS_INDEX + 1)){p;q}" $ROOT_DIR/assets/text/imagenet1k_classes.txt)

echo "Inferred class: $CLASS_INDEX - $CLASS_NAME"

