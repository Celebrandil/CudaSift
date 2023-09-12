cmake_minimum_required(VERSION 3.10)
project(syclsift LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)           # SYCL code requires this
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # Enable modern C++ standards
set(CMAKE_CXX_EXTENSIONS OFF)        # Use -std, not -gnu

option(GPU_AOT                 "Build AOT for Intel GPU"      OFF)
option(USE_NVIDIA_BACKEND      "Build for NVIDIA backend"     OFF)
option(USE_AMDHIP_BACKEND      "Build for AMD HIP backend"    OFF)
option(USE_INTEL_CPU           "Build for INTEL CPU"    OFF) 
option(USE_SM                  "Specifies which streaming multiprocessor architecture to use"     )
option(OpenCV_DIR              "Path to OpenCV_DIR"     )
option(DEVICE_TIMER            "Build using Device Timer" OFF)


# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
# Add OpenCV headers location to your include paths
include_directories(${OpenCV_INCLUDE_DIRS})
endif()


set(SOURCES
    ${CMAKE_SOURCE_DIR}/../common/Utility.cpp
    cudaImage.dp.cpp
    cudaImage.h
    cudaSiftH.dp.cpp
    cudaSiftH.h
    matching.dp.cpp
    cudaSiftD.h
    cudaSift.h
    geomFuncs.cpp
    mainSift.cpp
)

include_directories(
    ${CMAKE_SOURCE_DIR}/../common/
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

if(DEVICE_TIMER)
    message(STATUS "Enabling Device Timer")
    add_compile_options(-DDEVICE_TIMER)
endif()

# Use either default or user defined CXX flags
# -DCMAKE_CXX_FLAGS=" -blah -blah " overrides the default flags

set(USE_DEFAULT_FLAGS ON)
set(INTEL_CPU_CXX_FLAGS  " -O2 -fsycl -Wall -Wextra -Wno-unused-parameter ") 
set(INTEL_GPU_CXX_FLAGS  " -O2 -fsycl -ffast-math")
set(NVIDIA_GPU_CXX_FLAGS " -O3 -fsycl -ffast-math ")
set(AMD_GPU_CXX_FLAGS    " -O3 -fsycl -ffast-math ")

if("${CMAKE_CXX_FLAGS}" STREQUAL "")
    message(STATUS "Using DEFAULT compilation flags for the application")
    string(APPEND CMAKE_CXX_FLAGS "${INTEL_GPU_CXX_FLAGS}") # Default flags for NV backend
else()
    message(STATUS "OVERRIDING compilation flags")
    set(USE_DEFAULT_FLAGS OFF)
endif()

# JIT compilation 
if(GPU_AOT) 
    if( (${GPU_AOT} STREQUAL "pvc") OR (${GPU_AOT} STREQUAL "PVC") )
        message(STATUS "Enabling Intel GPU AOT compilation for ${GPU_AOT}")
        string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=spir64_gen -Xs \"-device 0x0bd5 -revision_id 0x2f\" -Xs \"-options -ze-opt-large-register-file\" ")
    else()
        message(STATUS "Using custom AOT compilation flag ${GPU_AOT}")
        string(APPEND CMAKE_CXX_FLAGS " ${GPU_AOT} ") # User should be aware of advanced AOT compilation flags
    endif()
elseif(USE_NVIDIA_BACKEND)    
    message(STATUS "Enabling NVIDIA backend")
    if(USE_DEFAULT_FLAGS)
        set(CMAKE_CXX_FLAGS "${NVIDIA_GPU_CXX_FLAGS}") # Default flags for NV backend
    endif()
    string(APPEND CMAKE_CXX_FLAGS "-fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_${USE_SM}") # -O3 will be used, even though -O2 was set earlier 
elseif(USE_AMDHIP_BACKEND)
    message(STATUS "Enabling AMD HIP backend for ${USE_AMDHIP_BACKEND} AMD architecture")
    if(USE_DEFAULT_FLAGS)
        set(CMAKE_CXX_FLAGS "${AMD_GPU_CXX_FLAGS}")
    endif()
    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${USE_AMDHIP_BACKEND}  ")
elseif(USE_INTEL_CPU)
    message(STATUS "Enabling INTEL CPU backend for ${USE_CPU_BACKEND} INTEL architecture")
    if(USE_DEFAULT_FLAGS)   
	    set(CMAKE_CXX_FLAGS "${INTEL_CPU_CXX_FLAGS}")				    
    endif()
    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=spir64_x86_64 -Xsycl-target-backend \"--march=avx512\"  ")
    # string(APPEND CMAKE_CXX_FLAGS "  -ffast-math -mprefer-vector-width=512 -mfma -fsycl-targets=spir64_x86_64--linux  \"-device avx512\"  ")
endif()

# Output the compiler flags that were constructed for visual inspection
message(STATUS "Compilation flags set to: ${CMAKE_CXX_FLAGS}")

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} stdc++ stdc++fs)
