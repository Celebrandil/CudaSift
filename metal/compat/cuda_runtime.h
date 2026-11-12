//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//              Apple Metal backend                       //
//********************************************************//

// The subset of the CUDA runtime API used by the tests, benchmark and examples,
// implemented in cudaSiftMetal.mm. Only on the include path of the Metal build.
// Since all CudaSift calls are synchronous on Metal, events are host timestamps.

#ifndef CUDASIFT_METAL_CUDA_RUNTIME_H
#define CUDASIFT_METAL_CUDA_RUNTIME_H

#include <cstddef>

typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorInvalidValue = 1 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2 };
struct cudaArray;   // CudaImage::t_data: a shared buffer of pitch-float rows on Metal

typedef struct CudaSiftMetalEvent *cudaEvent_t;
typedef struct CudaSiftMetalStream *cudaStream_t;

struct cudaDeviceProp {
  char name[256];
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int maxThreadsPerBlock;
  int major, minor;
  int multiProcessorCount;
  int memoryBusWidth;
  int l2CacheSize;
};

cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device);
// Memory: shared (unified) buffers, so all copies are plain memory copies
cudaError_t cudaMalloc(void **ptr, size_t size);
cudaError_t cudaFree(void *ptr);   // cudaErrorInvalidValue if ptr was not allocated by cudaMalloc
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch, const cudaArray *src, size_t wOffset, size_t hOffset,
                                  size_t width, size_t height, cudaMemcpyKind kind);
cudaError_t cudaGetLastError();
// A single device
cudaError_t cudaGetDeviceCount(int *count);
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaSetDevice(int device);

#endif
