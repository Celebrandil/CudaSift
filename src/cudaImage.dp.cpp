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

#include <sycl/sycl.hpp>
#include <cstdio>

#include "infra/memory.hpp"
#include "cudautils.h"
#include "cudaImage.h"

int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }
int iDivDown(int a, int b) { return a / b; }
int iAlignUp(int a, int b) { return (a % b != 0) ? (a - a % b + b) : a; }
int iAlignDown(int a, int b) { return a - a % b; }

void CudaImage::Allocate(int w, int h, int p, bool host, sycl::queue &q_ct, float &time, float *devmem, float *hostmem)
{
  width = w;
  height = h;
  pitch = p;
  d_data = devmem;
  h_data = hostmem;
  if (devmem == NULL)
  {
#ifdef DEVICE_TIMER
    auto start_malloc = std::chrono::steady_clock::now();
#endif
    d_data = (float *)infra::sift_malloc(pitch, (size_t)(sizeof(float) * width), (size_t)height, q_ct);
    q_ct.wait();
#ifdef DEVICE_TIMER
    auto stop_malloc = std::chrono::steady_clock::now();
    std::cout << "Allocate Time is " << std::chrono::duration<float, std::micro>(stop_malloc - start_malloc).count() << " us" << std::endl;
    time += std::chrono::duration<float, std::micro>(stop_malloc - start_malloc).count();
#endif

    pitch /= sizeof(float);
    if (d_data == NULL)
      printf("Failed to allocate device data\n");
    d_internalAlloc = true;
  }
  if (host && hostmem == NULL)
  {
    h_data = (float *)malloc(sizeof(float) * pitch * height);
    h_internalAlloc = true;
  }
}

CudaImage::CudaImage() : width(0), height(0), pitch(0), d_data(NULL), h_data(NULL), /*t_data(NULL), */ d_internalAlloc(false), h_internalAlloc(false)
{
}

CudaImage::~CudaImage()
{
  if (d_internalAlloc && d_data != NULL)
    try
    {
      safeCall((sycl::free(d_data, infra::get_default_queue()), 0));
    }
    catch (std::exception const &e)
    {
      std::cerr << e.what() << '\n';
    }
  d_data = NULL;
  if (h_internalAlloc && h_data != NULL)
    free(h_data);
  h_data = NULL;
}

double CudaImage::Download(sycl::queue &q_ct, float &time)
{
  double downloadTime = 0.0;
  int p = sizeof(float) * pitch;
  if (d_data != NULL && h_data != NULL)
  {
#ifdef DEVICE_TIMER
    auto start_memcpy = std::chrono::steady_clock::now();
#endif
    infra::sift_memcpy(d_data, p, h_data, sizeof(float) * width, sizeof(float) * width, height, infra::host_to_device, q_ct);
    q_ct.wait();

#ifdef DEVICE_TIMER
    auto stop_memcpy = std::chrono::steady_clock::now();
    downloadTime = std::chrono::duration<float, std::micro>(stop_memcpy - start_memcpy).count();
    time += downloadTime;
    std::cout << "Download Time is " << downloadTime << " us" << std::endl;
#endif
  }
  return downloadTime;
}
