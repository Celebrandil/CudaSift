//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

// Modifications Copyright (C) 2023 Intel Corporation

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom
// the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
// OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
// OR OTHER DEALINGS IN THE SOFTWARE.

// SPDX-License-Identifier: MIT

#ifndef CUDASIFTH_H
#define CUDASIFTH_H

#include <sycl/sycl.hpp>

#include "infra/infra.hpp"
#include "cudautils.h"
#include "cudaImage.h"
#include "cudaSift.h"

int ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh,
                    float lowestScale, float subsampling, float *memoryTmp, float *memorySub, sycl::queue &q_ct, float &totTime);
void ExtractSiftOctave(SiftData &siftData, CudaImage &img, int octave, float thresh, float lowestScale, float subsampling,
                       float *memoryTmp, sycl::queue &q_ct, float &totTime);
double ScaleDown(CudaImage &res, CudaImage &src, float variance, sycl::queue &q_ct, float &totTime);
double ScaleUp(CudaImage &res, CudaImage &src, sycl::queue &q_ct, float &totTime);
double ComputeOrientations(CudaImage &src, SiftData &siftData, int octave, sycl::queue &q_ct, float &totTime);
double ExtractSiftDescriptors(float *texObj, int pitch, SiftData &siftData, float subsampling,
                              int octave, sycl::queue &q_ct, float &totTime);
double RescalePositions(SiftData &siftData, float scale, sycl::queue &q_ct, float &totTime);
double LowPass(CudaImage &res, CudaImage &src, float scale, sycl::queue &q_ct, float &totTime);
void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel);
double LaplaceMulti(CudaImage &baseImage, CudaImage *results, int octave, sycl::queue &q_ct, float &totTime);
double FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh, float edgeLimit, float factor, float lowestScale,
                       float subsampling, int octave, sycl::queue &q_ct, float &totTime);
#endif
