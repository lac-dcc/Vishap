#!/bin/bash

set -e
source "$(dirname "${BASH_SOURCE[0]}")/env_check.sh"

root_dir="$(pwd -P)"
ln -sfn "${root_dir}/external/stablehlo" "${root_dir}/external/torch-mlir/externals/stablehlo"

mkdir -p build
build_dir="$(cd ./build && pwd -P)"

CC=$LLVM_BUILD_DIR/bin/clang
CXX=$LLVM_BUILD_DIR/bin/clang++

BUILD_TYPE="${BUILD_TYPE:-RelWithAsserts}"

extra_cmake_args=()
case "$BUILD_TYPE" in
  Release)
    cmake_build_type="Release"
    extra_cmake_args=(-DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG" -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG")
    ;;
  RelWithAsserts)
    cmake_build_type="Release"
    extra_cmake_args=(-DCMAKE_C_FLAGS_RELEASE="-O3 -UNDEBUG" -DCMAKE_CXX_FLAGS_RELEASE="-O3 -UNDEBUG")
    ;;
  Debug)
    cmake_build_type="Debug"
    ;;
  *)
    >&2 echo "Error: Invalid BUILD_TYPE '$BUILD_TYPE' (expected Release, RelWithAsserts, or Debug)"
    exit 1
    ;;
esac

cmake -S . -B $build_dir -G "Ninja" \
    -DLLVM_DIR=${LLVM_BUILD_DIR}/lib/cmake/llvm   \
    -DMLIR_DIR=${LLVM_BUILD_DIR}/lib/cmake/mlir   \
    -DCMAKE_BUILD_TYPE=$cmake_build_type          \
    "${extra_cmake_args[@]}"                      \
    -DCMAKE_C_COMPILER=$CC                        \
    -DCMAKE_CXX_COMPILER=$CXX                     \
    -DPython3_EXECUTABLE=$(which python3)         \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .
