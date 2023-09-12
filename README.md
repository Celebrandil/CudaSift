# SyclSift
SyclSift - SIFT features with SYCL.

# Building SyclSift
**To build cuda version**

mkdir build && cd build

//For A100 Machine

cmake ../ -DUSE_SM=80

//For H100 Machine

cmake ../ -DUSE_SM=90

make

**To build SYCL version**

mkdir build

cd build

#update the path for OpenCV_DIR

CXX=icpx cmake ../ -DGPU_AOT=pvc

make -sj

**To build SYCL version on NVIDIA Backend**

source /path/to/clang/

mkdir build && cd build

//For A100 Machine

CC=clang CXX=clang++ cmake ../ -DUSE_NVIDIA_BACKEND=YES -DUSE_SM=80 

//For H100 Machine

CC=clang CXX=clang++ cmake ../ -DUSE_NVIDIA_BACKEND=YES -DUSE_SM=90

make -sj

**To build SYCL version on AMD Backend**

source /path/to/clang/

mkdir build && cd build

//For MI-100 Machine

CC=clang CXX=clang++ cmake ../ -DUSE_AMDHIP_BACKEND=gfx908

//For MI-250 Machine

CC=clang CXX=clang++ cmake ../ -DUSE_AMDHIP_BACKEND=gfx90a

make -sj

# Running SyclSift

**To run sycl version**

./syclsift

**To run SYCL on NVIDIA Backend**

./syclsift

**To run SYCL on AMD Backend**

ONEAPI_DEVICE_SELECTOR=hip:* ./syclsift
