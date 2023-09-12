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
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>

#include "cudautils.h"
#include "cudaImage.h"
#include "cudaSift.h"
#include "cudaSiftD.h"
#include "cudaSiftH.h"
#include "cudaSiftD.dp.cpp"

#define PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))

template <>
struct sycl::is_device_copyable<CudaImage> : std::true_type
{
};

void InitCuda(sycl::queue &q_ct, int devNum)
{
  auto device = q_ct.get_device();
  std::cout << "Device Name:          " << device.get_info<sycl::info::device::name>() << std::endl;
  std::cout << "Max workgroup size:   " << device.get_info<sycl::info::device::max_work_group_size>() << std::endl;
  std::cout << "Max clock freq:   " << device.get_info<sycl::info::device::max_clock_frequency>() << std::endl;
}

float *AllocSiftTempMemory(int width, int height, int numOctaves, sycl::queue &q_ct, float &time, bool scaleUp)
{
  const int nd = NUM_SCALES + 3;
  int w = width * (scaleUp ? 2 : 1);
  int h = height * (scaleUp ? 2 : 1);
  int p = iAlignUp(w, 128);
  int size = h * p;         // image sizes
  int sizeTmp = nd * h * p; // laplace buffer sizes
  for (int i = 0; i < numOctaves; i++)
  {
    w /= 2;
    h /= 2;
    int p = iAlignUp(w, 128);
    size += h * p;
    sizeTmp += nd * h * p;
  }
  float *memoryTmp = NULL;
  size_t pitch;
  size += sizeTmp;

#ifdef DEVICE_TIMER
  auto start_malloc = std::chrono::steady_clock::now();
#endif
  memoryTmp = (float *)infra::sift_malloc(pitch, (size_t)4096, (size + 4095) / 4096 * sizeof(float), q_ct);
  q_ct.wait();
#ifdef DEVICE_TIMER
  auto stop_malloc = std::chrono::steady_clock::now();
  // printf("Malloc time for memoryTmp =          %.2f us\n", std::chrono::duration<float, std::micro>(stop_malloc - start_malloc).count());
  time += std::chrono::duration<float, std::micro>(stop_malloc - start_malloc).count();
#endif
  return memoryTmp;
}

void FreeSiftTempMemory(float *memoryTmp, sycl::queue &q_ct)
{
  if (memoryTmp)

    safeCall((sycl::free(memoryTmp, q_ct), 0));
}

void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, sycl::queue &q_ct,
                 float &totTime, float lowestScale, bool scaleUp, float *tempMemory)
{
  unsigned int *d_PointCounterAddr;

#ifdef DEVICE_TIMER
  auto start_memcpy = std::chrono::steady_clock::now();
#endif
  *((void **)&d_PointCounterAddr) = d_PointCounter.get_ptr();
  q_ct.memset(d_PointCounterAddr, 0, (8 * 2 + 1) * sizeof(int));
  q_ct.memcpy(d_MaxNumPoints.get_ptr(), &siftData.maxPts, sizeof(int));
  q_ct.wait();

#ifdef DEVICE_TIMER
  auto stop_memcpy = std::chrono::steady_clock::now();
  totTime += std::chrono::duration<float, std::micro>(stop_memcpy - start_memcpy).count();
#endif

  const int nd = NUM_SCALES + 3;
  int w = img.width * (scaleUp ? 2 : 1);
  int h = img.height * (scaleUp ? 2 : 1);
  int p = iAlignUp(w, 128);
  int width = w, height = h;
  int size = h * p;         // image sizes
  int sizeTmp = nd * h * p; // laplace buffer sizes
  for (int i = 0; i < numOctaves; i++)
  {
    w /= 2;
    h /= 2;
    int p = iAlignUp(w, 128);
    size += h * p;
    sizeTmp += nd * h * p;
  }
  float *memoryTmp = tempMemory;
  size += sizeTmp;
  if (!tempMemory)
  {
    size_t pitch;
#ifdef DEVICE_TIMER
    auto start_malloc2 = std::chrono::steady_clock::now();
#endif
    memoryTmp = (float *)infra::sift_malloc(pitch, (size_t)4096, (size + 4095) / 4096 * sizeof(float), q_ct);
    q_ct.wait();

#ifdef DEVICE_TIMER
    auto stop_malloc2 = std::chrono::steady_clock::now();
    // printf("Malloc time for memoryTmp =          %.2f us\n", std::chrono::duration<float, std::micro>(stop_malloc - start_malloc).count());
    totTime += std::chrono::duration<float, std::micro>(stop_malloc2 - start_malloc2).count();
#endif
  }
  float *memorySub = memoryTmp + sizeTmp;

  CudaImage lowImg;
  lowImg.Allocate(width, height, iAlignUp(width, 128), false, q_ct, totTime, memorySub);
  if (!scaleUp)
  {
    float kernel[8 * 12 * 16];
    PrepareLaplaceKernels(numOctaves, 0.0f, kernel);
#ifdef DEVICE_TIMER
    auto start_memcpy1 = std::chrono::steady_clock::now();
#endif
    q_ct.memcpy(d_LaplaceKernel.get_ptr(), kernel, 8 * 12 * 16 * sizeof(float));
    q_ct.wait();

#ifdef DEVICE_TIMER
    auto stop_memcpy1 = std::chrono::steady_clock::now();
    totTime += std::chrono::duration<float, std::micro>(stop_memcpy1 - start_memcpy1).count();
#endif

    LowPass(lowImg, img, fmax(initBlur, 0.001f), q_ct, totTime);

    ExtractSiftLoop(siftData, lowImg, numOctaves, 0.0f, thresh, lowestScale, 1.0f, memoryTmp,
                    memorySub + height * iAlignUp(width, 128), q_ct, totTime);

#ifdef DEVICE_TIMER
    auto start_memcpy2 = std::chrono::steady_clock::now();
#endif
    q_ct.memcpy(&siftData.numPts, &d_PointCounterAddr[2 * numOctaves], sizeof(int));
    q_ct.wait();
#ifdef DEVICE_TIMER
    auto stop_memcpy2 = std::chrono::steady_clock::now();
    totTime += std::chrono::duration<float, std::micro>(stop_memcpy2 - start_memcpy2).count();
#endif
    siftData.numPts = (siftData.numPts < siftData.maxPts ? siftData.numPts : siftData.maxPts);
  }
  else
  {
    CudaImage upImg;
    upImg.Allocate(width, height, iAlignUp(width, 128), false, q_ct, totTime, memoryTmp);
    ScaleUp(upImg, img, q_ct, totTime);
    LowPass(lowImg, upImg, fmax(initBlur, 0.001f), q_ct, totTime);
    float kernel[8 * 12 * 16];
    PrepareLaplaceKernels(numOctaves, 0.0f, kernel);
#ifdef DEVICE_TIMER
    auto start_memcpy3 = std::chrono::steady_clock::now();
#endif
    safeCall(
        (q_ct.memcpy(d_LaplaceKernel.get_ptr(), kernel,
                     8 * 12 * 16 * sizeof(float)),
         0));
    q_ct.wait();
#ifdef DEVICE_TIMER
    auto stop_memcpy3 = std::chrono::steady_clock::now();
    totTime += std::chrono::duration<float, std::micro>(stop_memcpy3 - start_memcpy3).count();
#endif
    ExtractSiftLoop(siftData, lowImg, numOctaves, 0.0f, thresh, lowestScale * 2.0f, 1.0f, memoryTmp,
                    memorySub + height * iAlignUp(width, 128), q_ct, totTime);
#ifdef DEVICE_TIMER
    auto start_memcpy4 = std::chrono::steady_clock::now();
#endif
    safeCall((q_ct.memcpy(&siftData.numPts, &d_PointCounterAddr[2 * numOctaves],
                          sizeof(int)),
              0));
    q_ct.wait();
#ifdef DEVICE_TIMER
    auto stop_memcpy4 = std::chrono::steady_clock::now();
    totTime += std::chrono::duration<float, std::micro>(stop_memcpy4 - start_memcpy4).count();
#endif
    siftData.numPts = (siftData.numPts < siftData.maxPts ? siftData.numPts : siftData.maxPts);
    RescalePositions(siftData, 0.5f, q_ct, totTime);
  }

  if (!tempMemory)
    safeCall((sycl::free(memoryTmp, q_ct), 0));
  if (siftData.h_data)
  {
#ifdef DEVICE_TIMER
    auto start_memcpy5 = std::chrono::steady_clock::now();
#endif
    q_ct.memcpy(siftData.h_data, siftData.d_data, sizeof(SiftPoint) * siftData.numPts);
    q_ct.wait();
#ifdef DEVICE_TIMER
    auto stop_memcpy5 = std::chrono::steady_clock::now();
    totTime += std::chrono::duration<float, std::micro>(stop_memcpy5 - start_memcpy5).count();
    printf("Total time for sift extraction =  %.2f us\n\n", totTime);
#endif
    printf("Number of Points after sift extraction =  %d\n\n", siftData.numPts);
  }
}

int ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale,
                    float subsampling, float *memoryTmp, float *memorySub, sycl::queue &q_ct, float &totTime)
{
  int w = img.width;
  int h = img.height;
  if (numOctaves > 1)
  {
    CudaImage subImg;
    int p = iAlignUp(w / 2, 128);
    subImg.Allocate(w / 2, h / 2, p, false, q_ct, totTime, memorySub);
    ScaleDown(subImg, img, 0.5f, q_ct, totTime);
    float totInitBlur = (float)sqrt(initBlur * initBlur + 0.5f * 0.5f) / 2.0f;
    ExtractSiftLoop(siftData, subImg, numOctaves - 1, totInitBlur, thresh, lowestScale,
                    subsampling * 2.0f, memoryTmp, memorySub + (h / 2) * p, q_ct, totTime);
  }
  ExtractSiftOctave(siftData, img, numOctaves, thresh, lowestScale, subsampling, memoryTmp, q_ct, totTime);
  return 0;
}

void c1toc4(float *f_ptr, sycl::float4 *f4_ptr, int width, int height,
            int f_pitch, int f4_pitch, sycl::id<2> idx)
{
  const int workItm_row = idx[0];
  const int workItm_col = idx[1];
  float *f_row_begin = f_ptr + f_pitch * workItm_row;
  sycl::float4 *f4_row_begin = f4_ptr + f4_pitch * workItm_row;

  f4_row_begin[workItm_col].x() = f_row_begin[workItm_col];
}

void ExtractSiftOctave(SiftData &siftData, CudaImage &img, int octave, float thresh, float lowestScale,
                       float subsampling, float *memoryTmp, sycl::queue &q_ct, float &totTime)
{
  const int nd = NUM_SCALES + 3;
  CudaImage diffImg[nd];
  int w = img.width;
  int h = img.height;
  int p = iAlignUp(w, 128);
  for (int i = 0; i < nd - 1; i++)
    diffImg[i].Allocate(w, h, p, false, q_ct, totTime, memoryTmp + i * p * h);
  float baseBlur = pow(2.0f, -1.0f / NUM_SCALES);
  float diffScale = pow(2.0f, 1.0f / NUM_SCALES);
  LaplaceMulti(img, diffImg, octave, q_ct, totTime);
  FindPointsMulti(diffImg, siftData, thresh, 10.0f, 1.0f / NUM_SCALES, lowestScale / subsampling, subsampling, octave, q_ct, totTime);
  ComputeOrientations(img, siftData, octave, q_ct, totTime);
  ExtractSiftDescriptors(img.d_data, img.pitch, siftData, subsampling, octave, q_ct, totTime);
}

void InitSiftData(SiftData &data, sycl::queue &q_ct, float &time, int num, bool host, bool dev)
{
  data.numPts = 0;
  data.maxPts = num;
  int sz = sizeof(SiftPoint) * num;
  data.h_data = NULL;
  if (host)
    data.h_data = (SiftPoint *)malloc(sz);
  data.d_data = NULL;
  if (dev)
  {
#ifdef DEVICE_TIMER
    auto start_malloc = std::chrono::steady_clock::now();
#endif
    data.d_data = (SiftPoint *)sycl::malloc_device(sz, q_ct);
    q_ct.wait();
#ifdef DEVICE_TIMER
    auto stop_malloc = std::chrono::steady_clock::now();
    time += std::chrono::duration<float, std::micro>(stop_malloc - start_malloc).count();
#endif
  }
}

void FreeSiftData(SiftData &data, sycl::queue &q_ct)
{
  if (data.d_data != NULL)
    sycl::free(data.d_data, q_ct.get_context());
  data.d_data = NULL;
  if (data.h_data != NULL)
    free(data.h_data);
  data.numPts = 0;
  data.maxPts = 0;
}

void PrintSiftData(SiftData &data, sycl::queue &q_ct)
{
  SiftPoint *h_data = data.h_data;
  if (data.h_data == NULL)
  {
    h_data = (SiftPoint *)malloc(sizeof(SiftPoint) * data.maxPts);
    q_ct.memcpy(h_data, data.d_data, sizeof(SiftPoint) * data.numPts)
        .wait();
    data.h_data = h_data;
  }
  for (int i = 0; i < data.numPts; i++)
  {
    printf("xpos         = %.2f\n", h_data[i].xpos);
    printf("ypos         = %.2f\n", h_data[i].ypos);
    printf("scale        = %.2f\n", h_data[i].scale);
    printf("sharpness    = %.2f\n", h_data[i].sharpness);
    printf("edgeness     = %.2f\n", h_data[i].edgeness);
    printf("orientation  = %.2f\n", h_data[i].orientation);
    printf("score        = %.2f\n", h_data[i].score);
    float *siftData = (float *)&h_data[i].data;
    for (int j = 0; j < 8; j++)
    {
      if (j == 0)
        printf("data = ");
      else
        printf("       ");
      for (int k = 0; k < 16; k++)
        if (siftData[j + 8 * k] < 0.05)
          printf(" .   ");
        else
          printf("%.2f ", siftData[j + 8 * k]);
      printf("\n");
    }
  }
  printf("Number of available points: %d\n", data.numPts);
  printf("Number of allocated points: %d\n", data.maxPts);
}

///////////////////////////////////////////////////////////////////////////////
// Host side master functions
///////////////////////////////////////////////////////////////////////////////

double ScaleDown(CudaImage &res, CudaImage &src, float variance, sycl::queue &q_ct, float &totTime)
{
  static float oldVariance = -1.0f;
  if (res.d_data == NULL || src.d_data == NULL)
  {
    printf("ScaleDown: missing data\n");
    return 0.0;
  }
  if (oldVariance != variance)
  {
    float h_Kernel[5];
    float kernelSum = 0.0f;
    for (int j = 0; j < 5; j++)
    {
      h_Kernel[j] = (float)expf(-(double)(j - 2) * (j - 2) / 2.0 / variance);
      kernelSum += h_Kernel[j];
    }
    for (int j = 0; j < 5; j++)
      h_Kernel[j] /= kernelSum;

#ifdef DEVICE_TIMER
    auto start_memcpy = std::chrono::steady_clock::now();
#endif
    q_ct.memcpy(d_ScaleDownKernel.get_ptr(), h_Kernel, 5 * sizeof(float)).wait();
#ifdef DEVICE_TIMER
    auto stop_memcpy = std::chrono::steady_clock::now();
    totTime += std::chrono::duration<float, std::micro>(stop_memcpy - start_memcpy).count();
#endif
    oldVariance = variance;
  }
#if 0
  dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
  dim3 threads(SCALEDOWN_W + 4, SCALEDOWN_H + 4);
#else
  sycl::range<3> blocks(1, iDivUp(src.height, SCALEDOWN_H),
                        iDivUp(src.width, SCALEDOWN_W));
  sycl::range<3> threads(1, 1, SCALEDOWN_W + 4);

#ifdef DEVICE_TIMER
  auto start_kernel = std::chrono::steady_clock::now();
#endif
  q_ct.submit([&](sycl::handler &cgh)
              {
                                     d_ScaleDownKernel.init();

                                     auto d_ScaleDownKernel_ptr_ct1 = d_ScaleDownKernel.get_ptr();

                                     sycl::accessor<float, 1, sycl::access_mode::read_write,
                                                    sycl::access::target::local>
                                         inrow_acc_ct1(sycl::range<1>(68 /*SCALEDOWN_W+4*/), cgh);
                                     sycl::accessor<float, 1, sycl::access_mode::read_write,
                                                    sycl::access::target::local>
                                         brow_acc_ct1(sycl::range<1>(160 /*5*(SCALEDOWN_W/2)*/), cgh);
                                     sycl::accessor<int, 1, sycl::access_mode::read_write,
                                                    sycl::access::target::local>
                                         yRead_acc_ct1(sycl::range<1>(20 /*SCALEDOWN_H+4*/), cgh);
                                     sycl::accessor<int, 1, sycl::access_mode::read_write,
                                                    sycl::access::target::local>
                                         yWrite_acc_ct1(sycl::range<1>(20 /*SCALEDOWN_H+4*/), cgh);

                                     auto res_data_ct1 = res.d_data;
                                     auto src_data_ct1 = src.d_data;
                                     auto src_width = src.width;
                                     auto src_pitch = src.pitch;
                                     auto src_height = src.height;
                                     auto res_pitch = res.pitch;

                                     cgh.parallel_for(
                                         sycl::nd_range<3>(blocks * threads, threads),
                                         [=](sycl::nd_item<3> item_ct1)[[intel::reqd_sub_group_size(32)]]
                                         {                                           
                                           ScaleDown(res_data_ct1, src_data_ct1, src_width, src_pitch, src_height,
                                                     res_pitch, item_ct1, d_ScaleDownKernel_ptr_ct1,
                                                     inrow_acc_ct1.get_pointer(), brow_acc_ct1.get_pointer(),
                                                     yRead_acc_ct1.get_pointer(), yWrite_acc_ct1.get_pointer());
                                         }); })
      .wait();
#ifdef DEVICE_TIMER
  auto stop_kernel = std::chrono::steady_clock::now();
  // printf("ScaleDown time =          %.2f us\n", std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count());
  totTime += std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count();
#endif
#endif
  checkMsg("ScaleDown() execution failed\n");
  return 0.0;
}

double ScaleUp(CudaImage &res, CudaImage &src, sycl::queue &q_ct, float &totTime)
{
  if (res.d_data == NULL || src.d_data == NULL)
  {
    printf("ScaleUp: missing data\n");
    return 0.0;
  }
  sycl::range<3> blocks(1, iDivUp(res.height, SCALEUP_H),
                        iDivUp(res.width, SCALEUP_W));
  sycl::range<3> threads(1, SCALEUP_H / 2, SCALEUP_W / 2);

#ifdef DEVICE_TIMER
  auto start_kernel = std::chrono::steady_clock::now();
#endif

  q_ct.submit([&](sycl::handler &cgh)
              {
                                     auto src_data_ct1 = src.d_data;
                                     auto res_data_ct1 = res.d_data;
                                     auto src_width = src.width;
                                     auto src_pitch = src.pitch;
                                     auto src_height = src.height;
                                     auto res_pitch = res.pitch;
                                     cgh.parallel_for(
                                         sycl::nd_range<3>(blocks * threads, threads),
                                         [=](sycl::nd_item<3> item_ct1)[[intel::reqd_sub_group_size(32)]]
                                         {                                           
                                           ScaleUp(res_data_ct1, src_data_ct1, src_width, src_pitch, src_height,
                                                   res_pitch, item_ct1);
                                         }); })
      .wait();

#ifdef DEVICE_TIMER
  auto stop_kernel = std::chrono::steady_clock::now();
  // printf("ScaleUp time =          %.2f us\n", std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count());
  totTime += std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count();
#endif
  checkMsg("ScaleUp() execution failed\n");
  return 0.0;
}

double ComputeOrientations(CudaImage &src, SiftData &siftData, int octave, sycl::queue &q_ct, float &totTime)
{
  sycl::range<3> blocks(1, 1, 512);
  sycl::range<3> threads(1, 1, 256);
#ifdef DEVICE_TIMER
  auto start_kernel = std::chrono::steady_clock::now();
#endif
  q_ct.submit([&](sycl::handler &cgh)
              {

                auto d_MaxNumPoints_ptr_ct1 = d_MaxNumPoints.get_ptr();                
                auto d_PointCounter_ptr_ct1 = d_PointCounter.get_ptr();

                sycl::accessor<float, 2, sycl::access_mode::read_write,
                                sycl::access::target::local>
                    img_acc_ct1(sycl::range<2>(19 /*WID*/, 19 /*WID*/), cgh);
                sycl::accessor<float, 2, sycl::access_mode::read_write,
                                sycl::access::target::local>
                    tmp_acc_ct1(sycl::range<2>(19 /*WID*/, 19 /*WID*/), cgh);
                sycl::accessor<float, 1, sycl::access_mode::read_write,
                                sycl::access::target::local>
                    hist_acc_ct1(sycl::range<1>(64 /*2*LEN*/), cgh);
                sycl::accessor<float, 1, sycl::access_mode::read_write,
                                sycl::access::target::local>
                    gaussx_acc_ct1(sycl::range<1>(19 /*WID*/), cgh);
                sycl::accessor<float, 1, sycl::access_mode::read_write,
                                sycl::access::target::local>
                    gaussy_acc_ct1(sycl::range<1>(19 /*WID*/), cgh);

                auto src_data_ct1 = src.d_data;
                auto src_width = src.width;
                auto src_pitch = src.pitch;
                auto src_height = src.height;
                auto siftData_data_ct1 = siftData.d_data;

                cgh.parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                      ComputeOrientationsCONSTNew(
                          src_data_ct1, src_width, src_pitch, src_height, siftData_data_ct1,
                          octave, item_ct1, *d_MaxNumPoints_ptr_ct1, d_PointCounter_ptr_ct1,
                          img_acc_ct1, tmp_acc_ct1, hist_acc_ct1.get_pointer(),
                          gaussx_acc_ct1.get_pointer(), gaussy_acc_ct1.get_pointer());
                    }); })
      .wait();
#ifdef DEVICE_TIMER
  auto stop_kernel = std::chrono::steady_clock::now();
  // printf("ComputeOrientationsCONSTNew time =          %.2f us\n", std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count());
  totTime += std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count();
#endif
  checkMsg("ComputeOrientations() execution failed\n");
  return 0.0;
}

double ExtractSiftDescriptors(float *texObj, int pitch, SiftData &siftData, float subsampling, int octave, sycl::queue &q_ct, float &totTime)
{
  sycl::range<3> blocks(1, 1, 512);
  sycl::range<3> threads(1, 8, 16);
#ifdef DEVICE_TIMER
  auto start_kernel = std::chrono::steady_clock::now();
#endif
  q_ct.submit([&](sycl::handler &cgh)
              {
                                     d_MaxNumPoints.init();
                                     d_PointCounter.init();

                                     auto d_MaxNumPoints_ptr_ct1 = d_MaxNumPoints.get_ptr();
                                     auto d_PointCounter_ptr_ct1 = d_PointCounter.get_ptr();

                                     sycl::accessor<float, 1, sycl::access_mode::read_write,
                                                    sycl::access::target::local>
                                         gauss_acc_ct1(sycl::range<1>(16), cgh);
                                     sycl::accessor<float, 1, sycl::access_mode::read_write,
                                                    sycl::access::target::local>
                                         buffer_acc_ct1(sycl::range<1>(128), cgh);
                                     sycl::accessor<float, 1, sycl::access_mode::read_write,
                                                    sycl::access::target::local>
                                         sums_acc_ct1(sycl::range<1>(4), cgh);

                                     auto siftData_data_ct1 = siftData.d_data;

                                     cgh.parallel_for(
                                         sycl::nd_range<3>(blocks * threads, threads), [=
                                     ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                                           32)]] {
                                           ExtractSiftDescriptorsCONSTNew(                                              
                                               texObj, pitch,
                                               siftData_data_ct1, subsampling, octave, item_ct1,
                                               *d_MaxNumPoints_ptr_ct1, d_PointCounter_ptr_ct1,
                                               gauss_acc_ct1.get_pointer(), buffer_acc_ct1.get_pointer(),
                                               sums_acc_ct1.get_pointer());
                                         }); })
      .wait();

#ifdef DEVICE_TIMER
  auto stop_kernel = std::chrono::steady_clock::now();
  // printf("ExtractSiftDescriptorsCONSTNew time =          %.2f us\n", std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count());
  totTime += std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count();
#endif
  checkMsg("ExtractSiftDescriptors() execution failed\n");
  return 0.0;
}

double RescalePositions(SiftData &siftData, float scale, sycl::queue &q_ct, float &totTime)
{
  sycl::range<3> blocks(1, 1, iDivUp(siftData.numPts, 64));
  sycl::range<3> threads(1, 1, 64);
#ifdef DEVICE_TIMER
  auto start_kernel = std::chrono::steady_clock::now();
#endif
  q_ct.submit([&](sycl::handler &cgh)
              {
                                     auto siftData_data_ct1 = siftData.d_data;
                                     auto sifData_numPts = siftData.numPts;
                                     cgh.parallel_for(
                                         sycl::nd_range<3>(blocks * threads, threads),
                                         [=](sycl::nd_item<3> item_ct1)[[intel::reqd_sub_group_size(32)]]
                                         {
                                           RescalePositions(siftData_data_ct1, sifData_numPts, scale, item_ct1);
                                         }); })
      .wait();
#ifdef DEVICE_TIMER
  auto stop_kernel = std::chrono::steady_clock::now();
  // printf("RescalePositions time =          %.2f us\n", std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count());
  totTime += std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count();
#endif
  checkMsg("RescapePositions() execution failed\n");
  return 0.0;
}

double LowPass(CudaImage &res, CudaImage &src, float scale, sycl::queue &q_ct, float &totTime)
{
  try
  {
    float kernel[2 * LOWPASS_R + 1];
    static float oldScale = -1.0f;
    if (scale != oldScale)
    {
      float kernelSum = 0.0f;
      float ivar2 = 1.0f / (2.0f * scale * scale);
      for (int j = -LOWPASS_R; j <= LOWPASS_R; j++)
      {
        kernel[j + LOWPASS_R] = (float)expf(-(double)j * j * ivar2);
        kernelSum += kernel[j + LOWPASS_R];
      }
      for (int j = -LOWPASS_R; j <= LOWPASS_R; j++)
        kernel[j + LOWPASS_R] /= kernelSum;

#ifdef DEVICE_TIMER
      auto start_memcpy_1 = std::chrono::steady_clock::now();
#endif
      q_ct.memcpy(d_LowPassKernel.get_ptr(), kernel,
                  (2 * LOWPASS_R + 1) * sizeof(float));
      q_ct.wait();
#ifdef DEVICE_TIMER
      auto stop_memcpy_1 = std::chrono::steady_clock::now();
      totTime += std::chrono::duration<float, std::micro>(stop_memcpy_1 - start_memcpy_1).count();
#endif
      oldScale = scale;
    }
    int width = res.width;
    int pitch = res.pitch;
    int height = res.height;
    sycl::range<3> blocks(1, iDivUp(height, LOWPASS_H), iDivUp(width, LOWPASS_W)); //(1, 34, 80)
    sycl::range<3> threads(1, 4, LOWPASS_W + 2 * LOWPASS_R);                       //(1, 4, 32)

#ifdef DEVICE_TIMER
    auto start_kernel = std::chrono::steady_clock::now();
#endif
    q_ct.submit([&](sycl::handler &cgh)
                {                                    
                                     auto d_LowPassKernel_ptr_ct1 = d_LowPassKernel.get_ptr();

                                     auto src_data_ct1 = src.d_data;
                                     auto res_data_ct1 = res.d_data;

                                     sycl::accessor<float, 2, sycl::access_mode::read_write,
                                                    sycl::access::target::local>
                                         xrows_acc_ct1(sycl::range<2>(16, 32), cgh);
                                     cgh.parallel_for(
                                         sycl::nd_range<3>(blocks * threads, threads), [=](sycl::nd_item<3> item_ct1)
                                         [[intel::reqd_sub_group_size(32)]]
                                         { LowPassBlockOld(src_data_ct1, res_data_ct1, width, pitch, height, item_ct1,
                                                        d_LowPassKernel_ptr_ct1, xrows_acc_ct1); }); })
        .wait();
#ifdef DEVICE_TIMER
    auto stop_kernel = std::chrono::steady_clock::now();
    // printf("LowPassBlock time =          %.2f us\n", std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count());
    totTime += std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count();
#endif
    checkMsg("LowPass() execution failed\n");
    return 0.0;
  }
  catch (sycl::exception const &e)
  {
    std::cout << e.what() << '\n';
  }
}

//==================== Multi-scale functions ===================//

void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel)
{
  if (numOctaves > 1)
  {
    float totInitBlur = (float)sqrt(initBlur * initBlur + 0.5f * 0.5f) / 2.0f;
    PrepareLaplaceKernels(numOctaves - 1, totInitBlur, kernel);
  }
  float scale = pow(2.0f, -1.0f / NUM_SCALES);
  float diffScale = pow(2.0f, 1.0f / NUM_SCALES);
  for (int i = 0; i < NUM_SCALES + 3; i++)
  {
    float kernelSum = 0.0f;
    float var = scale * scale - initBlur * initBlur;
    for (int j = 0; j <= LAPLACE_R; j++)
    {
      kernel[numOctaves * 12 * 16 + 16 * i + j] = (float)expf(-(double)j * j / 2.0 / var);
      kernelSum += (j == 0 ? 1 : 2) * kernel[numOctaves * 12 * 16 + 16 * i + j];
    }
    for (int j = 0; j <= LAPLACE_R; j++)
      kernel[numOctaves * 12 * 16 + 16 * i + j] /= kernelSum;
    scale *= diffScale;
  }
}

double LaplaceMulti(CudaImage &baseImage, CudaImage *results, int octave, sycl::queue &q_ct, float &totTime)
{
  int width = results[0].width;
  int pitch = results[0].pitch;
  int height = results[0].height;

#if 1
  sycl::range<3> threads(1, 1, LAPLACE_W + 2 * LAPLACE_R);    //(1, 1, 136)
  sycl::range<3> blocks(1, height, iDivUp(width, LAPLACE_W)); //(1, 1080, 15)

#ifdef DEVICE_TIMER
  auto start_kernel = std::chrono::steady_clock::now();
#endif

  q_ct.submit([&](sycl::handler &cgh)
              {
        float *d_LaplaceKernel_ptr_ct1 = d_LaplaceKernel.get_ptr();
        sycl::accessor<float, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            buff_acc_ct1(
                sycl::range<1>(1088 /*(LAPLACE_W + 2*LAPLACE_R)*LAPLACE_S*/), cgh);                       

        float *results_d_data_ct1 = results[0].d_data;
        float *baseImage_data_ct1 = baseImage.d_data;
        cgh.parallel_for(
            sycl::nd_range<3>(blocks * threads, threads),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
              LaplaceMultiMem(baseImage_data_ct1, results_d_data_ct1,
                              width, pitch, height, octave, item_ct1,
                              d_LaplaceKernel_ptr_ct1,
                              buff_acc_ct1.get_pointer());
            }); })
      .wait();

#ifdef DEVICE_TIMER
  auto stop_kernel = std::chrono::steady_clock::now();
  // printf("LaplaceMultiMem time =          %.2f us\n", std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count());
  totTime += std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count();
#endif
#endif
  checkMsg("LaplaceMulti() execution failed\n");
  return 0.0;
}

double FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh, float edgeLimit, float factor,
                       float lowestScale, float subsampling, int octave, sycl::queue &q_ct, float &totTime)
{
  if (sources->d_data == NULL)
  {
    printf("FindPointsMulti: missing data\n");
    return 0.0;
  }
  int w = sources->width;
  int p = sources->pitch;
  int h = sources->height;
#if 1
  sycl::range<3> blocks(1, iDivUp(h, MINMAX_H),
                        iDivUp(w, MINMAX_W) * NUM_SCALES);
  sycl::range<3> threads(1, 1, MINMAX_W + 2);

#ifdef DEVICE_TIMER
  auto start_kernel = std::chrono::steady_clock::now();
#endif
  auto event_FindPointsMulti = q_ct.submit([&](sycl::handler &cgh)
                                           {
                                     d_MaxNumPoints.init();
                                     d_PointCounter.init();

                                     auto d_MaxNumPoints_ptr_ct1 = d_MaxNumPoints.get_ptr();
                                     auto d_PointCounter_ptr_ct1 = d_PointCounter.get_ptr();

                                     sycl::accessor<unsigned short, 1, sycl::access_mode::read_write,
                                                    sycl::access::target::local>
                                         points_acc_ct1(sycl::range<1>(64 /*2*MEMWID*/), cgh);

                                     auto sources_d_data_ct0 = sources->d_data;
                                     auto siftData_data_ct1 = siftData.d_data;

                                     cgh.parallel_for(
                                         sycl::nd_range<3>(blocks * threads, threads), [=
                                     ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]]
                                         {                                           
                                           FindPointsMultiNew(sources_d_data_ct0, siftData_data_ct1, w, p, h,
                                                              subsampling, lowestScale, thresh, factor,
                                                              edgeLimit, octave, item_ct1,
                                                              *d_MaxNumPoints_ptr_ct1, d_PointCounter_ptr_ct1,
                                                              points_acc_ct1.get_pointer());
                                         }); });
  event_FindPointsMulti.wait();
#ifdef DEVICE_TIMER
  auto stop_kernel = std::chrono::steady_clock::now();
  // printf("FindPointsMultiNew time =          %.2f us\n", std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count())
  totTime += std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count();
#endif
#endif
  checkMsg("FindPointsMulti() execution failed\n");
  return 0.0;
}
