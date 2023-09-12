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

#ifndef CUDASIFT_H
#define CUDASIFT_H

#include "cudaImage.h"

struct rawImg_data
{
  float *raw_d_data;
  int pitch;

  void set_pitch(int pitch)
  {
    this->pitch = pitch;
  }

  float read(float xf, float yf)
  {
    int xi = xf;
    int yi = yf;
    return *(raw_d_data + yi * pitch + xi);
  }
};

typedef struct
{
  float xpos;
  float ypos;
  float scale;
  float sharpness;
  float edgeness;
  float orientation;
  float score;
  float ambiguity;
  int match;
  float match_xpos;
  float match_ypos;
  float match_error;
  float subsampling;
  float empty[3];
  float data[128];
} SiftPoint;

typedef struct
{
  int numPts; // Number of available Sift points
  int maxPts; // Number of allocated Sift points
#ifdef MANAGEDMEM
  SiftPoint *m_data; // Managed data
#else
  SiftPoint *h_data; // Host (CPU) data
  SiftPoint *d_data; // Device (GPU) data
#endif
} SiftData;

void InitCuda(sycl::queue &q_ct, int devNum = 0);
float *AllocSiftTempMemory(int width, int height, int numOctaves, sycl::queue &q_ct, float &totTime, bool scaleUp = false);
void FreeSiftTempMemory(float *memoryTmp, sycl::queue &q_ct);
void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, sycl::queue &q_ct, float &totTime, float lowestScale = 0.0f, bool scaleUp = false, float *tempMemory = 0);
void InitSiftData(SiftData &data, sycl::queue &q_ct, float &totTime, int num = 1024, bool host = false, bool dev = true);
void FreeSiftData(SiftData &data, sycl::queue &q_ct);
void PrintSiftData(SiftData &data, sycl::queue &q_ct);
double MatchSiftData(SiftData &data1, SiftData &data2, sycl::queue &q_ct, float &time);
double FindHomography(SiftData &data, float *homography, int *numMatches, sycl::queue &q_ct, float &time, int numLoops = 1000, float minScore = 0.85f, float maxAmbiguity = 0.95f, float thresh = 5.0f);

#endif
