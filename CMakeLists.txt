cmake_minimum_required(VERSION 3.15)
# C is required because StableHLO's embedded build includes HandleLLVMOptions,
# which runs C compiler checks.
project(Vishap LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------------------
# MLIR / LLVM configuration
# ------------------------------------------------------------------------------

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# ------------------------------------------------------------------------------
# Includes / definitions
# ------------------------------------------------------------------------------

include(AddLLVM)
include(AddMLIR)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/external/mlir-probe/include
  ${PROJECT_SOURCE_DIR}/external/stablehlo
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_BINARY_DIR}/external/mlir-probe/include
  ${CMAKE_BINARY_DIR}/external/stablehlo
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
add_definitions(${LLVM_DEFINITIONS})

# ------------------------------------------------------------------------------
# Project configuration
# ------------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_compile_options(-Wall)

# ------------------------------------------------------------------------------
# Subdirectories
# ------------------------------------------------------------------------------

add_subdirectory(external/mlir-probe)

set(STABLEHLO_BUILD_EMBEDDED ON CACHE BOOL "" FORCE)
set(STABLEHLO_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "" FORCE)
add_subdirectory(external/stablehlo)

add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(include)
add_subdirectory(lit_tests)
