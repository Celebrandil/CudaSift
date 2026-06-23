//********************************************************//
// CUDA-to-HIP compatibility shim for CudaSift            //
// Copyright (c) 2026 Advanced Micro Devices, Inc.        //
// Author: Jeff Daily <jeff.daily@amd.com>                //
//                                                        //
// This header keeps the rest of the sources in their      //
// plain CUDA spelling. On AMD it includes the HIP runtime //
// and #defines the CUDA symbols the project uses to their //
// HIP equivalents. On NVIDIA it is a no-op that pulls in  //
// <cuda_runtime.h>.                                       //
//                                                        //
// Symbol names follow PyTorch's authoritative hipify map: //
// torch/utils/hipify/cuda_to_hip_mappings.py             //
//********************************************************//

#ifndef CUDA_TO_HIP_H
#define CUDA_TO_HIP_H

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>

// ---- Error handling ----
#define cudaError              hipError_t
#define cudaError_t            hipError_t
#define cudaSuccess            hipSuccess
#define cudaGetErrorString     hipGetErrorString
#define cudaGetLastError       hipGetLastError
#define cudaDeviceSynchronize  hipDeviceSynchronize

// ---- Device management ----
#define cudaGetDeviceCount       hipGetDeviceCount
#define cudaGetDeviceProperties  hipGetDeviceProperties
#define cudaSetDevice            hipSetDevice
#define cudaDeviceProp           hipDeviceProp_t

// ---- Events / streams ----
#define cudaEvent_t            hipEvent_t
#define cudaEventCreate        hipEventCreate
#define cudaEventDestroy       hipEventDestroy
#define cudaEventRecord        hipEventRecord
#define cudaEventSynchronize   hipEventSynchronize
#define cudaEventElapsedTime   hipEventElapsedTime
#define cudaStream_t           hipStream_t

// ---- Linear / pitched memory ----
#define cudaMalloc         hipMalloc
#define cudaMallocManaged  hipMallocManaged
#define cudaFree           hipFree
#define cudaMallocPitch    hipMallocPitch
#define cudaMemcpy         hipMemcpy
#define cudaMemcpy2D       hipMemcpy2D
#define cudaMemset         hipMemset

// ---- Constant / symbol memory ----
#define cudaMemcpyToSymbol       hipMemcpyToSymbol
#define cudaMemcpyToSymbolAsync  hipMemcpyToSymbolAsync
#define cudaGetSymbolAddress     hipGetSymbolAddress

// ---- memcpy kinds ----
#define cudaMemcpyHostToDevice    hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost    hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice  hipMemcpyDeviceToDevice

// ---- CUDA arrays + channel format (texture backing store) ----
#define cudaArray              hipArray
#define cudaArray_t            hipArray_t
#define cudaMallocArray        hipMallocArray
#define cudaFreeArray          hipFreeArray
#define cudaChannelFormatDesc  hipChannelFormatDesc
#define cudaCreateChannelDesc  hipCreateChannelDesc
// cudaMemcpyToArray is deprecated and removed from current HIP; uses are
// rewritten to hipMemcpy2DToArray at the call site (see cudaImage.cu).
#define cudaMemcpy2DToArray    hipMemcpy2DToArray

// ---- Texture objects ----
#define cudaTextureObject_t       hipTextureObject_t
#define cudaCreateTextureObject   hipCreateTextureObject
#define cudaDestroyTextureObject  hipDestroyTextureObject
#define cudaResourceDesc          hipResourceDesc
#define cudaTextureDesc           hipTextureDesc
#define cudaResourceTypePitch2D   hipResourceTypePitch2D
#define cudaResourceTypeArray     hipResourceTypeArray
#define cudaResourceTypeLinear    hipResourceTypeLinear
#define cudaAddressModeClamp      hipAddressModeClamp
#define cudaFilterModeLinear      hipFilterModeLinear
#define cudaFilterModePoint       hipFilterModePoint
#define cudaReadModeElementType   hipReadModeElementType

// ---- Device intrinsics without a 1:1 HIP spelling ----
// HIP has no round-toward-zero float multiply. The only use is the RANSAC
// homography residual test, where rounding mode is immaterial; the default
// round-to-nearest multiply is the faithful HIP equivalent.
#define __fmul_rz(a, b) ((a) * (b))

// __any_sync takes a lane mask that must be 64-bit wide on CDNA wavefronts;
// the project passes the CUDA 32-bit full mask 0xffffffff. The mask-free
// __any polls the active wavefront, which matches the original "any active
// lane" intent on the 32-thread blocks that use it. Drop the mask on HIP.
#define __any_sync(mask, pred) __any(pred)

#else // NVIDIA / CUDA

#include <cuda_runtime.h>

#endif

#endif // CUDA_TO_HIP_H
