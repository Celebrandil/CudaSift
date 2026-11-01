cmake_minimum_required(VERSION 3.18)

project(CudaSift VERSION 3.0.0 LANGUAGES CXX CUDA)

# ---- Build options ----
option(BUILD_TESTS "Build test and benchmark programs" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(USE_MANAGED_MEM "Use CUDA managed memory" OFF)
option(VERBOSE_OUTPUT "Enable verbose timing output" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- CUDA architecture: Ada Lovelace (RTX 4060 Ti) ----
set(CMAKE_CUDA_ARCHITECTURES 89)

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
set_source_files_properties(mainSift.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cudasift mainSift.cpp)
target_link_libraries(cudasift PRIVATE cudasift_lib)

# ---- Examples ----
if(BUILD_EXAMPLES)
  set_source_files_properties(
    examples/demo_extract.cpp
    examples/demo_match.cpp
    examples/demo_video.cpp
    PROPERTIES LANGUAGE CUDA
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
    PROPERTIES LANGUAGE CUDA
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
