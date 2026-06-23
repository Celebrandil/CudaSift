cmake_minimum_required(VERSION 3.18)

# ---- Build options ----
option(BUILD_TESTS "Build test and benchmark programs" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(USE_MANAGED_MEM "Use CUDA managed memory" OFF)
option(VERBOSE_OUTPUT "Enable verbose timing output" ON)
option(USE_HIP "Build with HIP for AMD GPUs" OFF)

# Enable the device language conditionally so the NVIDIA path is unchanged
# while the AMD path uses HIP. Everything else stays in plain CUDA spelling;
# cuda_to_hip.h (force-included for HIP below) does the symbol translation.
if(USE_HIP)
  project(CudaSift VERSION 3.0.0 LANGUAGES CXX HIP)
else()
  project(CudaSift VERSION 3.0.0 LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(USE_HIP)
  set(DEVICE_LANG HIP)
  set(CMAKE_HIP_STANDARD 17)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
else()
  set(DEVICE_LANG CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # ---- CUDA architecture: Ada Lovelace (RTX 4060 Ti) ----
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

# ---- Find OpenCV ----
find_package(OpenCV REQUIRED)

# ---- Compile definitions ----
add_compile_definitions(
  $<$<BOOL:${VERBOSE_OUTPUT}>:VERBOSE>
  $<$<BOOL:${USE_MANAGED_MEM}>:MANAGEDMEM>
)

if(WIN32)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# ---- Device-language optimization flags ----
if(USE_HIP)
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -ffast-math")
  if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
  endif()
else()
  # ---- CUDA flags for Ada Lovelace optimization ----
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
  if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
  endif()
endif()

# The .cu translation units compile as the selected device language. Under
# USE_HIP this routes them through the HIP toolchain; .cu defaults to CUDA
# (which is not enabled in the HIP build) so the language must be set
# explicitly. cudaSiftD.cu is #include'd by cudaSiftH.cu and is not a TU.
set(CUDASIFT_DEVICE_SOURCES cudaImage.cu cudaSiftH.cu matching.cu)
set_source_files_properties(${CUDASIFT_DEVICE_SOURCES} PROPERTIES LANGUAGE ${DEVICE_LANG})

# ---- CudaSift library ----
add_library(cudasift_lib STATIC
  cudaImage.cu
  cudaImage.h
  cudaSiftH.cu
  cudaSiftH.h
  cudaSiftD.h
  cudaSift.h
  cudautils.h
  matching.cu
  geomFuncs.cpp
)

target_include_directories(cudasift_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cudasift_lib PUBLIC ${OpenCV_LIBS})
set_target_properties(cudasift_lib PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON
)

# ---- Main demo executable ----
set_source_files_properties(mainSift.cpp PROPERTIES LANGUAGE ${DEVICE_LANG})
add_executable(cudasift mainSift.cpp)
target_link_libraries(cudasift PRIVATE cudasift_lib)

# ---- Examples ----
if(BUILD_EXAMPLES)
  set_source_files_properties(
    examples/demo_extract.cpp
    examples/demo_match.cpp
    examples/demo_video.cpp
    PROPERTIES LANGUAGE ${DEVICE_LANG}
  )
  add_executable(demo_extract examples/demo_extract.cpp)
  target_link_libraries(demo_extract PRIVATE cudasift_lib)

  add_executable(demo_match examples/demo_match.cpp)
  target_link_libraries(demo_match PRIVATE cudasift_lib)

  add_executable(demo_video examples/demo_video.cpp)
  target_link_libraries(demo_video PRIVATE cudasift_lib)
endif()

# ---- Tests & Benchmarks ----
if(BUILD_TESTS)
  set_source_files_properties(
    tests/benchmark.cpp
    tests/test_extract.cpp
    tests/test_match.cpp
    tests/test_homography.cpp
    PROPERTIES LANGUAGE ${DEVICE_LANG}
  )
  add_executable(benchmark tests/benchmark.cpp)
  target_link_libraries(benchmark PRIVATE cudasift_lib)

  add_executable(test_extract tests/test_extract.cpp)
  target_link_libraries(test_extract PRIVATE cudasift_lib)

  add_executable(test_match tests/test_match.cpp)
  target_link_libraries(test_match PRIVATE cudasift_lib)

  add_executable(test_homography tests/test_homography.cpp)
  target_link_libraries(test_homography PRIVATE cudasift_lib)
endif()

# ---- HIP target configuration ----
if(USE_HIP)
  set(CUDASIFT_TARGETS cudasift_lib cudasift)
  if(BUILD_EXAMPLES)
    list(APPEND CUDASIFT_TARGETS demo_extract demo_match demo_video)
  endif()
  if(BUILD_TESTS)
    list(APPEND CUDASIFT_TARGETS benchmark test_extract test_match test_homography)
  endif()
  # project(... LANGUAGES ... HIP) above auto-detects the host GPU arch (and
  # errors on a no-GPU host); pass -DCMAKE_HIP_ARCHITECTURES to override.
  foreach(tgt ${CUDASIFT_TARGETS})
    set_target_properties(${tgt} PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    # USE_HIP selects the HIP branch of cuda_to_hip.h. __HIP_PLATFORM_AMD__ is
    # only defined once the HIP runtime header is included, so it cannot gate
    # the include of that very header; the explicit define breaks the cycle.
    target_compile_definitions(${tgt} PRIVATE USE_HIP)
    # Force-include the compat header into every HIP translation unit so no
    # per-file #include edits are needed.
    target_compile_options(${tgt} PRIVATE
      $<$<COMPILE_LANGUAGE:HIP>:-include${CMAKE_CURRENT_SOURCE_DIR}/cuda_to_hip.h>)
    # hip_compat/ provides cuda_runtime.h and cuda.h shims for the handful of
    # sources that include them directly; HIP build only.
    target_include_directories(${tgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat)
  endforeach()
endif()

# ---- Install ----
install(TARGETS cudasift cudasift_lib
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
)
install(FILES cudaSift.h cudaImage.h cudautils.h DESTINATION include)
install(DIRECTORY data/ DESTINATION share/cudasift/data)

# ---- CPack ----
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_GENERATOR "ZIP")
include(CPack)
