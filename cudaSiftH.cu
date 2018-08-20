//********************************************************//
// CUDA SIFT extractor by Mårten Björkman aka Celebrandil //
//********************************************************//  

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

#include "cudaSiftD.cu"

void InitCuda(int devNum)
{
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  if (!nDevices) {
    std::cerr << "No CUDA devices available" << std::endl;
    return;
  }
  devNum = std::min(nDevices-1, devNum);
  deviceInit(devNum);  
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, devNum);
  printf("Device Number: %d\n", devNum);
  printf("  Device name: %s\n", prop.name);
  printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate/1000);
  printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
  printf("  Peak Memory Bandwidth (GB/s): %.1f\n\n",
	 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
}

float *AllocSiftTempMemory(int width, int height, int numOctaves, bool scaleUp)
{
  TimerGPU timer(0);
  const int nd = NUM_SCALES + 3;
  int w = width*(scaleUp ? 2 : 1);
  int h = height*(scaleUp ? 2 : 1);
  int p = iAlignUp(w, 128);
  int size = h*p;                 // image sizes
  int sizeTmp = nd*h*p;           // laplace buffer sizes
  for (int i=0;i<numOctaves;i++) {
    w /= 2;
    h /= 2;
    int p = iAlignUp(w, 128);
    size += h*p;
    sizeTmp += nd*h*p; 
  }
  float *memoryTmp = NULL; 
  size_t pitch;
  size += sizeTmp;
  safeCall(cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size+4095)/4096*sizeof(float)));
#ifdef VERBOSE
  printf("Allocated memory size: %d bytes\n", size);
  printf("Memory allocation time =      %.2f ms\n\n", timer.read());
#endif
  return memoryTmp;
}

void FreeSiftTempMemory(float *memoryTmp)
{
  if (memoryTmp)
    safeCall(cudaFree(memoryTmp));
}

void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, bool scaleUp, float *tempMemory) 
{
  TimerGPU timer(0);
  int totPts = 0;
  safeCall(cudaMemcpyToSymbol(d_PointCounter, &totPts, sizeof(int)));
  safeCall(cudaMemcpyToSymbol(d_MaxNumPoints, &siftData.maxPts, sizeof(int)));

  const int nd = NUM_SCALES + 3;
  int w = img.width*(scaleUp ? 2 : 1);
  int h = img.height*(scaleUp ? 2 : 1);
  int p = iAlignUp(w, 128);
  int width = w, height = h;
  int size = h*p;                 // image sizes
  int sizeTmp = nd*h*p;           // laplace buffer sizes
  for (int i=0;i<numOctaves;i++) {
    w /= 2;
    h /= 2;
    int p = iAlignUp(w, 128);
    size += h*p;
    sizeTmp += nd*h*p; 
  }
  float *memoryTmp = tempMemory; 
  size += sizeTmp;
  if (!tempMemory) {
    size_t pitch;
    safeCall(cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size+4095)/4096*sizeof(float)));
#ifdef VERBOSE
    printf("Allocated memory size: %d bytes\n", size);
    printf("Memory allocation time =      %.2f ms\n\n", timer.read());
#endif
  }
  float *memorySub = memoryTmp + sizeTmp;

  CudaImage lowImg;
  lowImg.Allocate(width, height, iAlignUp(width, 128), false, memorySub);
  if (!scaleUp) {
    LowPass(lowImg, img, max(initBlur, 0.001f));
    TimerGPU timer1(0);
    ExtractSiftLoop(siftData, lowImg, numOctaves, 0.0f, thresh, lowestScale, 1.0f, memoryTmp, memorySub + height*iAlignUp(width, 128));
    safeCall(cudaMemcpyFromSymbol(&siftData.numPts, d_PointCounter, sizeof(int)));
    siftData.numPts = (siftData.numPts<siftData.maxPts ? siftData.numPts : siftData.maxPts);
    printf("SIFT extraction time =        %.2f ms\n", timer1.read());
  } else {
    CudaImage upImg;
    upImg.Allocate(width, height, iAlignUp(width, 128), false, memoryTmp);
    ScaleUp(upImg, img);
    LowPass(lowImg, upImg, max(initBlur, 0.001f));
    ExtractSiftLoop(siftData, lowImg, numOctaves, 0.0f, thresh, lowestScale*2.0f, 1.0f, memoryTmp, memorySub + height*iAlignUp(width, 128));
    safeCall(cudaMemcpyFromSymbol(&siftData.numPts, d_PointCounter, sizeof(int)));
    siftData.numPts = (siftData.numPts<siftData.maxPts ? siftData.numPts : siftData.maxPts);
    RescalePositions(siftData, 0.5f);
  } 
  
  if (!tempMemory)
    safeCall(cudaFree(memoryTmp));
#ifdef MANAGEDMEM
  safeCall(cudaDeviceSynchronize());
#else
  if (siftData.h_data)
    safeCall(cudaMemcpy(siftData.h_data, siftData.d_data, sizeof(SiftPoint)*siftData.numPts, cudaMemcpyDeviceToHost));
#endif
  double totTime = timer.read();
  printf("Incl prefiltering & memcpy =  %.2f ms\n\n", totTime);
}

void ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp, float *memorySub) 
{
#ifdef VERBOSE
  TimerGPU timer(0);
#endif
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
#ifdef VERBOSE
  double totTime = timer.read();
  printf("ExtractSift time total =      %.2f ms\n\n", totTime);
#endif
}

void ExtractSiftOctave(SiftData &siftData, CudaImage &img, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp)
{
  const int nd = NUM_SCALES + 3;
#ifdef VERBOSE
  TimerGPU timer0;
#endif
  CudaImage diffImg[nd];
  int w = img.width; 
  int h = img.height;
  int p = iAlignUp(w, 128);
  for (int i=0;i<nd-1;i++) 
    diffImg[i].Allocate(w, h, p, false, memoryTmp + i*p*h); 

  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = img.d_data;
  resDesc.res.pitch2D.width = img.width;
  resDesc.res.pitch2D.height = img.height;
  resDesc.res.pitch2D.pitchInBytes = img.pitch*sizeof(float);  
  resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeClamp;
  texDesc.addressMode[1]   = cudaAddressModeClamp;
  texDesc.filterMode       = cudaFilterModeLinear;
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;
  // Create texture object
  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

#ifdef VERBOSE
  TimerGPU timer1;
#endif
  float baseBlur = pow(2.0f, -1.0f/NUM_SCALES);
  float diffScale = pow(2.0f, 1.0f/NUM_SCALES);
  LaplaceMulti(texObj, img, diffImg, baseBlur, diffScale, initBlur);
  int fstPts = 0;
  safeCall(cudaMemcpyFromSymbol(&fstPts, d_PointCounter, sizeof(int)));
  double sigma = baseBlur*diffScale;
  FindPointsMulti(diffImg, siftData, thresh, 10.0f, sigma, 1.0f/NUM_SCALES, lowestScale/subsampling, subsampling);
#ifdef VERBOSE
  double gpuTimeDoG = timer1.read();
  TimerGPU timer4;
#endif
#if 0
  OrientAndExtract(texObj, siftData, fstPts, subsampling);
#else
  int totPts = 0;
  safeCall(cudaMemcpyFromSymbol(&totPts, d_PointCounter, sizeof(int)));
  totPts = (totPts<siftData.maxPts ? totPts : siftData.maxPts);
  if (totPts>fstPts) {
    ComputeOrientations(texObj, siftData, fstPts, totPts); 
    safeCall(cudaMemcpyFromSymbol(&totPts, d_PointCounter, sizeof(int)));
    totPts = (totPts<siftData.maxPts ? totPts : siftData.maxPts);
    ExtractSiftDescriptors(texObj, siftData, fstPts, totPts, subsampling); 
  }
#endif
  safeCall(cudaDestroyTextureObject(texObj));
#ifdef VERBOSE
  double gpuTimeSift = timer4.read();
  double totTime = timer0.read();
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
  static float oldVariance = -1.0f;
  if (res.d_data==NULL || src.d_data==NULL) {
    printf("ScaleDown: missing data\n");
    return 0.0;
  }
  if (oldVariance!=variance) {
    float h_Kernel[5];
    float kernelSum = 0.0f;
    for (int j=0;j<5;j++) {
      h_Kernel[j] = (float)expf(-(double)(j-2)*(j-2)/2.0/variance);      
      kernelSum += h_Kernel[j];
    }
    for (int j=0;j<5;j++)
      h_Kernel[j] /= kernelSum;  
    safeCall(cudaMemcpyToSymbol(d_Kernel1, h_Kernel, 5*sizeof(float)));
    oldVariance = variance;
  }
  dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
  dim3 threads(SCALEDOWN_W + 4);
  ScaleDown<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch); 
  checkMsg("ScaleDown() execution failed\n");
  return 0.0;
}

double ScaleUp(CudaImage &res, CudaImage &src)
{
  if (res.d_data==NULL || src.d_data==NULL) {
    printf("ScaleUp: missing data\n");
    return 0.0;
  }
  dim3 blocks(iDivUp(res.width, SCALEUP_W), iDivUp(res.height, SCALEUP_H));
  dim3 threads(SCALEUP_W, SCALEUP_H);
  ScaleUp<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch); 
  checkMsg("ScaleUp() execution failed\n");
  return 0.0;
}   

double ComputeOrientations(cudaTextureObject_t texObj, SiftData &siftData, int fstPts, int totPts)
{
  dim3 blocks(totPts - fstPts);
  dim3 threads(128);
#ifdef MANAGEDMEM
  ComputeOrientations<<<blocks, threads>>>(texObj, siftData.m_data, fstPts);
#else
  ComputeOrientations<<<blocks, threads>>>(texObj, siftData.d_data, fstPts);
#endif
  checkMsg("ComputeOrientations() execution failed\n");
  return 0.0;
}

double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData, int fstPts, int totPts, float subsampling)
{
  dim3 blocks(totPts - fstPts); 
  dim3 threads(16, 8);
#ifdef MANAGEDMEM
  ExtractSiftDescriptors<<<blocks, threads>>>(texObj, siftData.m_data, fstPts, subsampling);
#else
  ExtractSiftDescriptors<<<blocks, threads>>>(texObj, siftData.d_data, fstPts, subsampling);
#endif
  checkMsg("ExtractSiftDescriptors() execution failed\n");
  return 0.0; 
}

#if 0
double OrientAndExtract(cudaTextureObject_t texObj, SiftData &siftData, int fstPts, float subsampling)
{
#ifdef MANAGEDMEM
  OrientAndExtract<<<1,1>>>(texObj, siftData.m_data, fstPts, subsampling);
#else
  OrientAndExtract<<<1,1>>>(texObj, siftData.d_data, fstPts, subsampling);
#endif
  checkMsg("OrientAndExtract() execution failed\n");
  return 0.0;
}
#endif

double RescalePositions(SiftData &siftData, float scale)
{
  dim3 blocks(iDivUp(siftData.numPts, 64));
  dim3 threads(64);
  RescalePositions<<<blocks, threads>>>(siftData.d_data, siftData.numPts, scale);
  checkMsg("RescapePositions() execution failed\n");
  return 0.0; 
}

double LowPass(CudaImage &res, CudaImage &src, float scale)
{
  float kernel[16];
  float kernelSum = 0.0f;
  float ivar2 = 1.0f/(2.0f*scale*scale);
  for (int j=-LOWPASS_R;j<=LOWPASS_R;j++) {
    kernel[j+LOWPASS_R] = (float)expf(-(double)j*j*ivar2);
    kernelSum += kernel[j+LOWPASS_R]; 
  }
  for (int j=-LOWPASS_R;j<=LOWPASS_R;j++) 
    kernel[j+LOWPASS_R] /= kernelSum;  
  safeCall(cudaMemcpyToSymbol(d_Kernel2, kernel, 12*16*sizeof(float)));
  int width = res.width;
  int pitch = res.pitch;
  int height = res.height;
  dim3 blocks(iDivUp(width, LOWPASS_W), iDivUp(height, LOWPASS_H));
  dim3 threads(LOWPASS_W+2*LOWPASS_R, LOWPASS_H);
  LowPass<<<blocks, threads>>>(src.d_data, res.d_data, width, pitch, height);
  checkMsg("LowPass() execution failed\n");
  return 0.0; 
}

//==================== Multi-scale functions ===================//

double LaplaceMulti(cudaTextureObject_t texObj, CudaImage &baseImage, CudaImage *results, float baseBlur, float diffScale, float initBlur)
{
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
  safeCall(cudaMemcpyToSymbol(d_Kernel2, kernel, 12*16*sizeof(float)));
  int width = results[0].width;
  int pitch = results[0].pitch;
  int height = results[0].height;
  dim3 blocks(iDivUp(width, LAPLACE_W), height);
  dim3 threads(LAPLACE_W+2*LAPLACE_R, LAPLACE_S);
#if 1
  LaplaceMultiMem<<<blocks, threads>>>(baseImage.d_data, results[0].d_data, width, pitch, height);
#else
  LaplaceMultiTex<<<blocks, threads>>>(texObj, results[0].d_data, width, pitch, height);
#endif
  checkMsg("LaplaceMulti() execution failed\n");
  return 0.0; 
}

double FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh, float edgeLimit, float scale, float factor, float lowestScale, float subsampling)
{
  if (sources->d_data==NULL) {
    printf("FindPointsMulti: missing data\n");
    return 0.0;
  }
  int w = sources->width;
  int p = sources->pitch;
  int h = sources->height;
  
  union {
    struct {
      float threshs[2];
      float scales[8];
      float factor;
      float edgeLimit;
    } s;
    float data[12];
  } params;
  params.s.threshs[0] = thresh;
  params.s.threshs[1] = -thresh;
  params.s.factor = factor;
  params.s.edgeLimit = edgeLimit;
  float diffScale = pow(2.0f, factor);
  for (int i=0;i<NUM_SCALES;i++) {
    //scales[i] = scale;
    params.s.scales[i] = scale;
    scale *= diffScale;
  }
  safeCall(cudaMemcpyToSymbol(*d_FindParams.data, params.data, 12*sizeof(float)));

  dim3 blocks(iDivUp(w, MINMAX_W)*NUM_SCALES, iDivUp(h, MINMAX_H));
  dim3 threads(MINMAX_W + 2); 
#ifdef MANAGEDMEM
  FindPointsMulti<<<blocks, threads>>>(sources->d_data, siftData.m_data, w, p, h, NUM_SCALES, subsampling, lowestScale); 
#else
  FindPointsMulti<<<blocks, threads>>>(sources->d_data, siftData.d_data, w, p, h, NUM_SCALES, subsampling, lowestScale); 
#endif
  checkMsg("FindPointsMulti() execution failed\n");
  return 0.0;
}

