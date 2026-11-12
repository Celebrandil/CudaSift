//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//              Apple Metal backend                       //
//********************************************************//

// Access to the Metal objects behind CudaSift's data (Metal backend), so that other Metal
// code can read the points and matches in place, without a copy, and encode its own work
// after CudaSift's, e.g. on the same queue or in the same batch (cudaSiftBatch.h).
//
// The objects are returned as void * (id<MTLDevice>, id<MTLCommandQueue>, id<MTLBuffer>,
// id<MTLCommandBuffer>: cast them back with (__bridge id<...>)), so this header is plain C++.
// Ownership is not transferred: they stay valid as long as the corresponding CudaSift object
// (the device and queue for the program, a buffer as long as its allocation, a command
// buffer until its batch is submitted).
//
// Outside a batch, CudaSift's calls return after their GPU work has completed. Command
// buffers on CudaSiftMetalQueue() run in order and CudaSift's buffers are hazard-tracked, so
// work encoded on that queue after a call sees its results.

#ifndef CUDASIFTMETAL_H
#define CUDASIFTMETAL_H

#include <cstddef>
#include "cudaSift.h"

#define CUDASIFT_HAS_METAL_INTEROP 1

// id<MTLDevice> and id<MTLCommandQueue> used by CudaSift
void *CudaSiftMetalDevice();
void *CudaSiftMetalQueue();

// id<MTLBuffer> containing ptr, a pointer into memory allocated by CudaSift (SiftData::d_data,
// CudaImage::d_data, the temporary memory of AllocSiftTempMemory, cudaMalloc of
// compat/cuda_runtime.h), and in *offset the byte offset of ptr in it. nullptr (and *offset
// 0) if ptr isn't in such memory.
void *CudaSiftMetalBuffer(const void *ptr, size_t *offset);

// In an open batch (BeginSiftBatch): ends the batch's current compute encoder and returns the
// batch's id<MTLCommandBuffer>, into which the caller may encode its own work (with its own
// encoders, ended before the next CudaSift call). The next CudaSift call of the batch, or
// EndSiftBatch, continues in a new encoder after that work. nullptr outside a batch.
void *CudaSiftMetalBatchCommandBuffer();

// id<MTLBuffer> holding the point count of data as an int, and in *offset its byte offset. In a
// batch, SiftData::numPts isn't valid until the batch completes, but this count is valid on the
// GPU when the work encoded after the call that computes it runs. If data hasn't been extracted
// in the open batch, or outside a batch (after waiting for a submitted batch, see
// cudaSiftBatch.h), the buffer is set to data.numPts.
void *CudaSiftMetalNumPtsBuffer(SiftData &data, size_t *offset);

#endif
