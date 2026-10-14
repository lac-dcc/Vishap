# Embed torch-mlir's C++ dialects and conversion passes into Vishap.
# Avoid using torch-mlir root CMakeLists.txt given that it calls project(),
# forces a second Python package, and configures its own StableHLO.

set(TORCH_MLIR_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/torch-mlir")
set(TORCH_MLIR_BINARY_DIR "${CMAKE_BINARY_DIR}/external/torch-mlir")

set(TORCH_MLIR_ENABLE_STABLEHLO ON)
set(TORCH_MLIR_USE_EXTERNAL_STABLEHLO ON)
set(TORCH_MLIR_ENABLE_TOSA OFF)
set(TORCH_MLIR_ENABLE_REFBACKEND OFF)
set(TORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS OFF)

# Tablegen and C++ sources gate StableHLO conversion on this macro.
set(TORCH_MLIR_TABLEGEN_FLAGS "-DTORCH_MLIR_ENABLE_STABLEHLO")
add_compile_definitions(TORCH_MLIR_ENABLE_STABLEHLO)

if(NOT MLIR_INCLUDE_DIR)
  list(GET MLIR_INCLUDE_DIRS 0 MLIR_INCLUDE_DIR)
endif()
if(NOT MLIR_GENERATED_INCLUDE_DIR)
  list(GET MLIR_INCLUDE_DIRS 1 MLIR_GENERATED_INCLUDE_DIR)
endif()

include_directories(
  ${TORCH_MLIR_SOURCE_DIR}/include
  ${TORCH_MLIR_BINARY_DIR}/include
)

# Helpers referenced by torch-mlir's own CMakeLists files.
macro(torch_mlir_enable_werror)
endmacro()

function(torch_mlir_target_includes target)
  set(_dirs
    $<BUILD_INTERFACE:${MLIR_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${MLIR_GENERATED_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${TORCH_MLIR_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${TORCH_MLIR_BINARY_DIR}/include>
  )
  target_include_directories(${target} PUBLIC ${_dirs})
  if(TARGET obj.${target})
    target_include_directories(obj.${target} PRIVATE ${_dirs})
  endif()
endfunction()

message(STATUS "Embedding torch-mlir C++ libraries from ${TORCH_MLIR_SOURCE_DIR}")

add_subdirectory(${TORCH_MLIR_SOURCE_DIR}/include ${TORCH_MLIR_BINARY_DIR}/include)
add_subdirectory(${TORCH_MLIR_SOURCE_DIR}/lib ${TORCH_MLIR_BINARY_DIR}/lib)
