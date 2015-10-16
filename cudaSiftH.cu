//********************************************************//
// CUDA SIFT extractor by Mårten Björkman aka Celebrandil //
//********************************************************//  

#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <cudautils.h>

#include "cudaImage.h"
#include "cudaSift.h"
#include "cudaSiftD.h"
#include "cudaSiftH.h"

#include "cudaSiftD.cu"

void InitCuda()
{
  deviceInit(0);  
}

void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling) 
{
  TimerGPU timer(0);
  int totPts = 0;
  safeCall(cudaMemcpyToSymbol(d_PointCounter, &totPts, sizeof(int)));
  safeCall(cudaMemcpyToSymbol(d_MaxNumPoints, &siftData.maxPts, sizeof(int)));

  const int nb = NUM_SCALES + 3;
  const int nd = NUM_SCALES + 3;
  int w = img.width;
  int h = img.height;
  int p = iAlignUp(w, 128);
  int size = (nb+nd)*h*p;
  int sizeTmp = size;
  for (int i=0;i<numOctaves;i++) {
    w /= 2;
    h /= 2;
    int p = iAlignUp(w, 128);
    size += h*p;
  }
  float *memoryTmp = NULL;
  size_t pitch;
  safeCall(cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size+4095)/4096*sizeof(float)));
  float *memorySub = memoryTmp + sizeTmp;

  ExtractSiftLoop(siftData, img, numOctaves, initBlur, thresh, lowestScale, subsampling, memoryTmp, memorySub);
  safeCall(cudaMemcpyFromSymbol(&siftData.numPts, d_PointCounter, sizeof(int)));
  siftData.numPts = (siftData.numPts<siftData.maxPts ? siftData.numPts : siftData.maxPts);
  safeCall(cudaFree(memoryTmp));
#ifdef MANAGEDMEM
  safeCall(cudaDeviceSynchronize());
#else
  if (siftData.h_data)
    safeCall(cudaMemcpy(siftData.h_data, siftData.d_data, sizeof(SiftPoint)*siftData.numPts, cudaMemcpyDeviceToHost));
#endif
  double totTime = timer.read();
#ifndef VERBOSE
  printf("Total time incl memory =      %.2f ms\n", totTime);
#endif
}

void ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp, float *memorySub) 
{
  TimerGPU timer(0);
  int w = img.width;
  int h = img.height;
  if (numOctaves>1) {
    CudaImage subImg;
    int p = iAlignUp(w/2, 128);
    subImg.Allocate(w/2, h/2, p, false, memorySub); 
    ScaleDown(subImg, img, 0.5f);
    float totInitBlur = (float)sqrt(initBlur*initBlur + 0.5f*0.5f) / 2.0f;
    ExtractSiftLoop(siftData, subImg, numOctaves-1, totInitBlur, thresh, lowestScale, subsampling*2.0f, memoryTmp, memorySub + (h/2)*p);
  }
  if (lowestScale<subsampling*2.0f) 
    ExtractSiftOctave(siftData, img, initBlur, thresh, lowestScale, subsampling, memoryTmp);
  double totTime = timer.read();
#ifdef VERBOSE
  printf("ExtractSift time total =      %.2f ms\n\n", totTime);
#endif
}

void ExtractSiftOctave(SiftData &siftData, CudaImage &img, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp)
{
  const int nb = NUM_SCALES + 3;
  const int nd = NUM_SCALES + 3;
  const double baseBlur = pow(2.0, -1.0/NUM_SCALES);
  int w = img.width; 
  int h = img.height;
  CudaImage blurImg[nb];
  CudaImage diffImg[nd];

  TimerGPU timer0;
  int p = iAlignUp(w, 128);
  for (int i=0;i<nb;i++) 
    blurImg[i].Allocate(w, h, p, false, memoryTmp + i*p*h); 
  for (int i=0;i<nb-1;i++) 
    diffImg[i].Allocate(w, h, p, false, memoryTmp + (nb+i)*p*h); 
  tex.addressMode[0] = cudaAddressModeClamp;
  tex.addressMode[1] = cudaAddressModeClamp;
  tex.filterMode = cudaFilterModeLinear; 
  tex.normalized = false;
  size_t offset = 0;
  safeCall(cudaBindTexture2D(&offset, tex, img.d_data, tex.channelDesc, img.width, img.height, img.pitch*sizeof(float)));

  TimerGPU timer1;
  float diffScale = pow(2.0f, 1.0f/NUM_SCALES);
  LaplaceMulti(diffImg, img, baseBlur, diffScale, initBlur);
  int fstPts = 0;
  safeCall(cudaMemcpyFromSymbol(&fstPts, d_PointCounter, sizeof(int)));
  double sigma = baseBlur*diffScale;
  FindPointsMulti(diffImg, siftData, thresh, 10.0f, sigma, 1.0f/NUM_SCALES, lowestScale/subsampling);
  double gpuTimeDoG = timer1.read();
  TimerGPU timer4;
  int totPts = 0;
  safeCall(cudaMemcpyFromSymbol(&totPts, d_PointCounter, sizeof(int)));
  totPts = (totPts<siftData.maxPts ? totPts : siftData.maxPts);
  if (totPts>fstPts) {
    ComputeOrientations(img, siftData, fstPts, totPts); 
    safeCall(cudaMemcpyFromSymbol(&totPts, d_PointCounter, sizeof(int)));
    totPts = (totPts<siftData.maxPts ? totPts : siftData.maxPts);
    ExtractSiftDescriptors(img, siftData, fstPts, totPts, subsampling); 
  }
  safeCall(cudaUnbindTexture(tex));
  double gpuTimeSift = timer4.read();

  double totTime = timer0.read();
#ifdef VERBOSE
  printf("GPU time : %.2f ms + %.2f ms + %.2f ms = %.2f ms\n", totTime-gpuTimeDoG-gpuTimeSift, gpuTimeDoG, gpuTimeSift, totTime);
  safeCall(cudaMemcpyFromSymbol(&totPts, d_PointCounter, sizeof(int)));
  totPts = (totPts<siftData.maxPts ? totPts : siftData.maxPts);
  if (totPts>0) 
    printf("           %.2f ms / DoG,  %.4f ms / Sift,  #Sift = %d\n", gpuTimeDoG/NUM_SCALES, gpuTimeSift/(totPts-fstPts), totPts-fstPts); 
#endif
}

void InitSiftData(SiftData &data, int num, bool host, bool dev)
{
  data.numPts = 0;
  data.maxPts = num;
  int sz = sizeof(SiftPoint)*num;
#ifdef MANAGEDMEM
  safeCall(cudaMallocManaged((void **)&data.m_data, sz));
#else
  data.h_data = NULL;
  if (host)
    data.h_data = (SiftPoint *)malloc(sz);
  data.d_data = NULL;
  if (dev)
    safeCall(cudaMalloc((void **)&data.d_data, sz));
#endif
}

void FreeSiftData(SiftData &data)
{
#ifdef MANAGEDMEM
  safeCall(cudaFree(data.m_data));
#else
  if (data.d_data!=NULL)
    safeCall(cudaFree(data.d_data));
  data.d_data = NULL;
  if (data.h_data!=NULL)
    free(data.h_data);
#endif
  data.numPts = 0;
  data.maxPts = 0;
}

void PrintSiftData(SiftData &data)
{
#ifdef MANAGEDMEM
  SiftPoint *h_data = data.m_data;
#else
  SiftPoint *h_data = data.h_data;
  if (data.h_data==NULL) {
    h_data = (SiftPoint *)malloc(sizeof(SiftPoint)*data.maxPts);
    safeCall(cudaMallocHost((void **)&h_data, sizeof(SiftPoint)*data.maxPts));
    safeCall(cudaMemcpy(h_data, data.d_data, sizeof(SiftPoint)*data.numPts, cudaMemcpyDeviceToHost));
    data.h_data = h_data;
  }
#endif
  for (int i=0;i<data.numPts;i++) {
    printf("xpos         = %.2f\n", h_data[i].xpos);
    printf("ypos         = %.2f\n", h_data[i].ypos);
    printf("scale        = %.2f\n", h_data[i].scale);
    printf("sharpness    = %.2f\n", h_data[i].sharpness);
    printf("edgeness     = %.2f\n", h_data[i].edgeness);
    printf("orientation  = %.2f\n", h_data[i].orientation);
    printf("score        = %.2f\n", h_data[i].score);
    float *siftData = (float*)&h_data[i].data;
    for (int j=0;j<8;j++) {
      if (j==0) 
	printf("data = ");
      else 
	printf("       ");
      for (int k=0;k<16;k++)
	if (siftData[j+8*k]<0.05)
	  printf(" .   ");
	else
	  printf("%.2f ", siftData[j+8*k]);
      printf("\n");
    }
  }
  printf("Number of available points: %d\n", data.numPts);
  printf("Number of allocated points: %d\n", data.maxPts);
}

///////////////////////////////////////////////////////////////////////////////
// Host side master functions
///////////////////////////////////////////////////////////////////////////////

double ScaleDown(CudaImage &res, CudaImage &src, float variance)
{
  if (res.d_data==NULL || src.d_data==NULL) {
    printf("ScaleDown: missing data\n");
    return 0.0;
  }
  float h_Kernel[5];
  float kernelSum = 0.0f;
  for (int j=0;j<5;j++) {
    h_Kernel[j] = (float)expf(-(double)(j-2)*(j-2)/2.0/variance);      
    kernelSum += h_Kernel[j];
  }
  for (int j=0;j<5;j++)
    h_Kernel[j] /= kernelSum;  
  safeCall(cudaMemcpyToSymbol(d_Kernel, h_Kernel, 5*sizeof(float)));
  dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
  dim3 threads(SCALEDOWN_W + 4);
  ScaleDown<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch); 
  checkMsg("ScaleDown() execution failed\n");
  return 0.0;
}

double ComputeOrientations(CudaImage &img, SiftData &siftData, int fstPts, int totPts)
{
  dim3 blocks(totPts - fstPts);
  dim3 threads(128);
#ifdef MANAGEDMEM
  ComputeOrientations<<<blocks, threads>>>(img.d_data, siftData.m_data, fstPts);
#else
  ComputeOrientations<<<blocks, threads>>>(img.d_data, siftData.d_data, fstPts);
#endif
  checkMsg("ComputeOrientations() execution failed\n");
  return 0.0;
}

double ExtractSiftDescriptors(CudaImage &img, SiftData &siftData, int fstPts, int totPts, float subsampling)
{
  dim3 blocks(totPts - fstPts); 
  dim3 threads(16, 8);
#ifdef MANAGEDMEM
  ExtractSiftDescriptors<<<blocks, threads>>>(img.d_data, siftData.m_data, fstPts, subsampling);
#else
  ExtractSiftDescriptors<<<blocks, threads>>>(img.d_data, siftData.d_data, fstPts, subsampling);
#endif
  checkMsg("ExtractSiftDescriptors() execution failed\n");
  return 0.0; 
}

//==================== Multi-scale functions ===================//

double LaplaceMulti(CudaImage *results, CudaImage &origImg, float baseBlur, float diffScale, float initBlur)
{
  float *d_DataA = origImg.d_data;
  float *d_DataB = results[0].d_data;
  if (d_DataA==NULL || d_DataB==NULL) {
    printf("LaplaceMulti: missing data\n");
    return 0.0;
  } 
  float kernel[12*16];
  float scale = baseBlur;
  for (int i=0;i<NUM_SCALES+3;i++) {
    float kernelSum = 0.0f;
    float var = scale*scale - initBlur*initBlur;
    for (int j=-LAPLACE_R;j<=LAPLACE_R;j++) {
      kernel[16*i+j+LAPLACE_R] = (float)expf(-(double)j*j/2.0/var);
      kernelSum += kernel[16*i+j+LAPLACE_R]; 
    }
    for (int j=-LAPLACE_R;j<=LAPLACE_R;j++) 
      kernel[16*i+j+LAPLACE_R] /= kernelSum;  
    scale *= diffScale;
  }
  safeCall(cudaMemcpyToSymbol(d_Kernel, kernel, 12*16*sizeof(float)));
  int width = results[0].width;
  int pitch = results[0].pitch;
  int height = results[0].height;
  dim3 blocks(iDivUp(width+2*LAPLACE_R, LAPLACE_W), height);
  dim3 threads(LAPLACE_W+2*LAPLACE_R, LAPLACE_S); 
  LaplaceMulti<<<blocks, threads>>>(d_DataB, d_DataA, width, pitch, height);
  checkMsg("ConvRowGPU() execution failed\n");
  return 0.0; 
}

double FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh, float edgeLimit, float scale, float factor, float lowestScale)
{
  if (sources->d_data==NULL) {
    printf("FindPointsMulti: missing data\n");
    return 0.0;
  }
  int w = sources->width;
  int p = sources->pitch;
  int h = sources->height;
  float threshs[2] = { thresh, -thresh };
  float scales[NUM_SCALES];  
  int nScales = 0;
  float diffScale = pow(2.0f, factor);
  for (int i=0;i<NUM_SCALES;i++) {
    if (scale>=lowestScale) 
      scales[nScales++] = scale;
    scale *= diffScale;
  }
  safeCall(cudaMemcpyToSymbol(d_Threshold, &threshs, 2*sizeof(float)));
  safeCall(cudaMemcpyToSymbol(d_EdgeLimit, &edgeLimit, sizeof(float)));
  safeCall(cudaMemcpyToSymbol(d_Scales, scales, sizeof(float)*NUM_SCALES));
  safeCall(cudaMemcpyToSymbol(d_Factor, &factor, sizeof(float)));

  if (nScales>0) {
    dim3 blocks(iDivUp(w, MINMAX_W)*nScales, iDivUp(h, MINMAX_H));
    dim3 threads(MINMAX_W + 2); 
#ifdef MANAGEDMEM
    FindPointsMulti<<<blocks, threads>>>(sources->d_data, siftData.m_data, w, p, h, nScales); 
#else
    FindPointsMulti<<<blocks, threads>>>(sources->d_data, siftData.d_data, w, p, h, nScales); 
#endif
    checkMsg("FindPointsMulti() execution failed\n");
  }
  return 0.0;
}

