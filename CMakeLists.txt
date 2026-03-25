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

# =============================================================================
# GCC VERSION CONFIGURATION
# =============================================================================
# You must specify the GCC version to use for CUDA compilation.
# This is required because CUDA has specific GCC version compatibility requirements.
#
# Option 1: Set it here directly (uncomment and modify the line below)
# set(GCC_VERSION "10")
#
# Option 2: Pass it as a CMake parameter
# cmake .. -DGCC_VERSION=10
#
# Common compatible versions:
#   - CUDA 11.x: GCC 7, 8, 9, 10
#   - CUDA 12.x: GCC 9, 10, 11, 12
# =============================================================================

if(NOT DEFINED GCC_VERSION)
  message(FATAL_ERROR 
    "\n"
    "=======================================================================\n"
    "GCC_VERSION is not specified!\n"
    "=======================================================================\n"
    "Please specify the GCC version using one of the following methods:\n"
    "\n"
    "  1. Edit CMakeLists.txt and uncomment: set(GCC_VERSION \"10\")\n"
    "  2. Pass as parameter: cmake .. -DGCC_VERSION=10\n"
    "\n"
    "Example: cmake .. -DGCC_VERSION=10\n"
    "=======================================================================\n"
  )
endif()

find_program(GCC_COMPILER gcc-${GCC_VERSION})
find_program(GXX_COMPILER g++-${GCC_VERSION})

if(NOT GCC_COMPILER)
  message(FATAL_ERROR "gcc-${GCC_VERSION} not found! Please specify a valid GCC version.")
endif()

if(NOT GXX_COMPILER)
  message(FATAL_ERROR "g++-${GCC_VERSION} not found! Please specify a valid GCC version.")
endif()

message(STATUS "Using GCC: ${GCC_COMPILER}")
message(STATUS "Using G++: ${GXX_COMPILER}")

find_package(OpenCV REQUIRED)
find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

if (WIN32)
  set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;-O2;-DVERBOSE") 
endif()
if (UNIX)
  if (APPLE)
    set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;-O2;-DVERBOSE") 
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse2 ")
    list(APPEND CUDA_NVCC_FLAGS "-lineinfo;-ccbin;${GXX_COMPILER};--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT")
  endif()
endif()

set(cuda_sources
  cudaImage.cu  
  cudaImage.h  
  cudaSiftH.cu 
  cudaSiftH.h  
  matching.cu  
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

cuda_add_executable(cudasift ${cuda_sources} ${sources} OPTIONS -arch=sm_35)

set_target_properties(cudasift PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)

target_link_libraries(cudasift ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})

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
