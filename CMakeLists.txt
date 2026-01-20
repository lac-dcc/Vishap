cmake_minimum_required(VERSION 3.15)
project(Vishap LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------------------
# MLIR / LLVM configuration
# ------------------------------------------------------------------------------

# IMPORTANT: disable LSP *before* find_package(MLIR)
set(MLIR_ENABLE_LSP OFF CACHE BOOL "Disable MLIR LSP support" FORCE)

find_package(LLVM REQUIRED CONFIG
  PATHS /home/fernando/Program/llvm-project/build/lib/cmake/llvm
)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")

find_package(MLIR REQUIRED CONFIG
  PATHS /home/fernando/Program/llvm-project/build/lib/cmake/mlir
)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# ------------------------------------------------------------------------------
# Includes / definitions
# ------------------------------------------------------------------------------

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)


add_definitions(${LLVM_DEFINITIONS})

# ------------------------------------------------------------------------------
# Subdirectories
# ------------------------------------------------------------------------------

add_subdirectory(lib)
add_subdirectory(tools)
