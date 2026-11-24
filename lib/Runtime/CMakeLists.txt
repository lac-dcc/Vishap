add_library(VishapRuntime SHARED probe.cpp)

add_library(VishapCalibrator SHARED calibrator.cpp)
target_compile_options(VishapCalibrator PRIVATE -O3)
if(OpenMP_CXX_FOUND)
  target_link_libraries(VishapCalibrator PRIVATE OpenMP::OpenMP_CXX)
endif()

