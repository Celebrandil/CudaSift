cmake_minimum_required(VERSION 3.18)

# ---- Backend: CUDA (default) or Apple Metal ----
option(USE_METAL "Build the Apple Metal backend instead of CUDA (macOS only)" OFF)

# ---- CUDA architecture: defaults to Ada Lovelace (RTX 4060 Ti) ----
# Override with -DCMAKE_CUDA_ARCHITECTURES=75 (e.g. Tesla T4) or the CUDAARCHS env var.
# Must be set before project(), which otherwise initializes it to the compiler default.
if(NOT USE_METAL AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

if(USE_METAL)
  project(CudaSift VERSION 3.0.0 LANGUAGES CXX OBJCXX)
else()
  project(CudaSift VERSION 3.0.0 LANGUAGES CXX CUDA)
endif()

# ---- Build options ----
option(BUILD_TESTS "Build test and benchmark programs" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(USE_MANAGED_MEM "Use CUDA managed memory" OFF)
option(VERBOSE_OUTPUT "Enable verbose timing output" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(USE_METAL)
  set(CMAKE_OBJCXX_STANDARD 17)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
else()
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
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

if(USE_METAL)
  # ---- CudaSift library, Metal backend ----
  # The kernels are compiled at run time from cudaSift.metal, embedded in the library
  # together with the headers it needs.
  set(METAL_KERNEL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cudaSiftD.h
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/cudaSiftMetalShared.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cudaSiftRandom.h
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/cudaSift.metal
  )
  string(REPLACE ";" "|" METAL_KERNEL_SOURCES_ARG "${METAL_KERNEL_SOURCES}")
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cudaSiftMetalSource.h
    COMMAND ${CMAKE_COMMAND} "-DIN=${METAL_KERNEL_SOURCES_ARG}" -DOUT=${CMAKE_CURRENT_BINARY_DIR}/cudaSiftMetalSource.h
            -P ${CMAKE_CURRENT_SOURCE_DIR}/metal/EmbedMetalSource.cmake
    DEPENDS ${METAL_KERNEL_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/metal/EmbedMetalSource.cmake
    VERBATIM
  )
  if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
    set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -O2")
  endif()
  add_library(cudasift_lib STATIC
    metal/cudaSiftMetal.mm
    metal/cudaSiftMetalShared.h
    cudaSiftBatch.h
    metal/cudaSiftMetal.h
    metal/compat/cuda_runtime.h
    ${CMAKE_CURRENT_BINARY_DIR}/cudaSiftMetalSource.h
    cudaImage.h
    cudaSiftD.h
    cudaSift.h
    cudaSiftRandom.h
    geomFuncs.cpp
  )
  target_include_directories(cudasift_lib
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/metal ${CMAKE_CURRENT_SOURCE_DIR}/metal/compat
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
  )
  target_compile_options(cudasift_lib PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(cudasift_lib PUBLIC ${OpenCV_LIBS} "-framework Metal" "-framework Foundation")
  set(CUDASIFT_APP_LANGUAGE CXX)
else()
  # ---- CUDA flags for Ada Lovelace optimization ----
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
  if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
  endif()

  # ---- CudaSift library ----
  add_library(cudasift_lib STATIC
    cudaImage.cu
    cudaImage.h
    cudaSiftH.cu
    cudaSiftH.h
    cudaSiftD.h
    cudaSift.h
    cudaSiftBatch.h
    cudautils.h
    cudaSiftRandom.h
    matching.cu
    geomFuncs.cpp
  )

  target_include_directories(cudasift_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(cudasift_lib PUBLIC ${OpenCV_LIBS})
  set_target_properties(cudasift_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
  )
  set(CUDASIFT_APP_LANGUAGE CUDA)
endif()

# ---- Main demo executable ----
set_source_files_properties(mainSift.cpp PROPERTIES LANGUAGE ${CUDASIFT_APP_LANGUAGE})
add_executable(cudasift mainSift.cpp)
target_link_libraries(cudasift PRIVATE cudasift_lib)

# ---- Examples ----
if(BUILD_EXAMPLES)
  set_source_files_properties(
    examples/demo_extract.cpp
    examples/demo_match.cpp
    examples/demo_video.cpp
    PROPERTIES LANGUAGE ${CUDASIFT_APP_LANGUAGE}
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
    tests/test_homography_gpu.cpp
    tests/test_batch.cpp
    PROPERTIES LANGUAGE ${CUDASIFT_APP_LANGUAGE}
  )
  add_executable(benchmark tests/benchmark.cpp)
  target_link_libraries(benchmark PRIVATE cudasift_lib)

  add_executable(test_extract tests/test_extract.cpp)
  target_link_libraries(test_extract PRIVATE cudasift_lib)

  add_executable(test_match tests/test_match.cpp)
  target_link_libraries(test_match PRIVATE cudasift_lib)

  add_executable(test_homography tests/test_homography.cpp)
  target_link_libraries(test_homography PRIVATE cudasift_lib)

  add_executable(test_homography_gpu tests/test_homography_gpu.cpp)
  target_link_libraries(test_homography_gpu PRIVATE cudasift_lib)

  add_executable(test_batch tests/test_batch.cpp)
  find_package(Threads REQUIRED)   # std::thread
  target_link_libraries(test_batch PRIVATE cudasift_lib Threads::Threads)
  if(USE_METAL)
    # Access to the Metal objects (cudaSiftMetal.h), in Objective-C++
    add_executable(test_metal_interop tests/test_metal_interop.mm)
    target_compile_options(test_metal_interop PRIVATE -fobjc-arc)
    target_link_libraries(test_metal_interop PRIVATE cudasift_lib)
  endif()
endif()

# ---- Install ----
install(TARGETS cudasift cudasift_lib
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
)
if(USE_METAL)
  install(FILES cudaSift.h cudaImage.h cudaSiftBatch.h metal/cudaSiftMetal.h DESTINATION include)
else()
  install(FILES cudaSift.h cudaImage.h cudaSiftBatch.h cudautils.h DESTINATION include)
endif()
install(DIRECTORY data/ DESTINATION share/cudasift/data)

# ---- CPack ----
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_GENERATOR "ZIP")
include(CPack)

# ---- OpenCV- and VLFeat-compatible SIFT (ExtractSiftCompat, CUDA only) ----
if(NOT USE_METAL)
  target_sources(cudasift_lib PRIVATE cudaSiftCompat.cu cudaSiftCompat.h cudaSiftCompatHost.h)
  target_compile_definitions(cudasift_lib PUBLIC CUDASIFT_HAS_COMPAT)   # for users of the library
  find_package(Threads REQUIRED)   # std::mutex
  target_link_libraries(cudasift_lib PUBLIC Threads::Threads)
  install(FILES cudaSiftCompat.h DESTINATION include)
  if(BUILD_TESTS)
    # Calls from several threads at once
    set_source_files_properties(tests/test_compat_threads.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(test_compat_threads tests/test_compat_threads.cpp)
    target_link_libraries(test_compat_threads PRIVATE cudasift_lib)
    # Keypoint-by-keypoint comparison with OpenCV's and VLFeat's SIFT (cv::SIFT: OpenCV 4.4 or later)
    if(OpenCV_VERSION VERSION_LESS 4.4)
      message(STATUS "compare_sift not built: it needs cv::SIFT (OpenCV 4.4 or later)")
    else()
      set_source_files_properties(tests/compare_sift.cpp PROPERTIES LANGUAGE CUDA)
      add_executable(compare_sift tests/compare_sift.cpp)
      target_link_libraries(compare_sift PRIVATE cudasift_lib)
      # Optional: VLFeat as a reference (--vlfeat, --colmap), with -DVLFEAT_DIR=<prefix with include/vl and lib>
      find_path(VLFEAT_INCLUDE_DIR vl/sift.h HINTS ${VLFEAT_DIR}/include ${VLFEAT_DIR})
      find_library(VLFEAT_LIBRARY NAMES vl HINTS ${VLFEAT_DIR}/lib ${VLFEAT_DIR})
      if(VLFEAT_INCLUDE_DIR AND VLFEAT_LIBRARY)
        target_compile_definitions(compare_sift PRIVATE HAVE_VLFEAT)
        target_include_directories(compare_sift PRIVATE ${VLFEAT_INCLUDE_DIR})
        target_link_libraries(compare_sift PRIVATE ${VLFEAT_LIBRARY})
      endif()
    endif()
  endif()
endif()
