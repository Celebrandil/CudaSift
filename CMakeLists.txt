cmake_minimum_required(VERSION 2.6)

project(cudaSift)
set(cudaSift_VERSION_MAJOR 2)
set(cudaSift_VERSION_MINOR 0)
set(cudaSift_VERSION_PATCH 0)

set(CPACK_PACKAGE_VERSION_MAJOR "${cudaSift_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${cudaSift_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${cudaSift_VERSION_PATCH}")
set(CPACK_GENERATOR "ZIP")
include(CPack)

find_package(OpenCV REQUIRED)
find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

if (WIN32)
  set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;--compiler-options;-O2;-DVERBOSE") 
endif()
if (UNIX)
  set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2 -std=c++0x ")
  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;--compiler-bindir=/usr/bin/gcc-4.4;--compiler-options;-O2;-DVERBOSE") 
endif()

set(cuda_sources
  cudaImage.cu  
  cudaImage.h  
  cudaSiftH.cu 
  cudaSiftH.h  
  matching.cu  
  singular.cu
  cudaSiftD.h  
  cudaSift.h  
  cudautils.h
)  

set(sources
  geomFuncs.cpp  
  mainSift.cpp
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR} 
)

cuda_add_executable(cudasift ${cuda_sources} ${sources})

set_target_properties(cudasift PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)

target_link_libraries(cudasift
  ${OpenCV_LIBS}
)

install(FILES 
  ${cuda_sources} 
  ${sources}
  cudaSiftD.cu
  CMakeLists.txt
  Copyright.txt
  DESTINATION .
)
install(FILES data/left.pgm data/righ.pgm
  DESTINATION data
)