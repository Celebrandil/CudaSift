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
  int w = img.width;
  int h = img.height;
  if (numOctaves>1) {
    CudaImage subImg;
    int p = iAlignUp(w/2, 128);
    subImg.Allocate(w/2, h/2, p, false, NULL, NULL); 
    ScaleDown(subImg, img, 0.5f);
    float totInitBlur = (float)sqrt(initBlur*initBlur + 0.5f*0.5f) / 2.0f;
    ExtractSift(siftData, subImg, numOctaves-1, totInitBlur, thresh, lowestScale, subsampling*2.0f);
  }
  if (lowestScale<subsampling*2.0f) 
    ExtractSiftOctave(siftData, img, initBlur, thresh, lowestScale, subsampling);
  double totTime = timer.read();
#ifdef VERBOSE
  printf("ExtractSift time total =      %.2f ms\n\n", totTime);
#endif
}

void ExtractSiftOctave(SiftData &siftData, CudaImage &img, double initBlur, float thresh, float lowestScale, float subsampling)
{
  const int maxPts = iAlignUp(4096, 128);
  const int nb = NUM_SCALES + 3;
  const int nd = NUM_SCALES + 3;
  const double baseBlur = pow(2.0, -1.0/NUM_SCALES);
  int w = img.width; 
  int h = img.height;
  CudaImage blurImg[nb];
  CudaImage diffImg[nd];
  CudaImage tempImg;
  CudaImage sift; // { xpos, ypos, scale, strength, edge, orient1, orient2 };
  CudaImage desc;

  TimerGPU timer0;
  float *memory = NULL;
  int p = iAlignUp(w, 128);
  int allocSize = (nb+nd+1)*p*h + maxPts*7 +  128*maxPts;
  safeCall(cudaMalloc((void **)&memory, sizeof(float)*allocSize));
  for (int i=0;i<nb;i++) 
    blurImg[i].Allocate(w, h, p, false, memory + i*p*h); 
  for (int i=0;i<nb-1;i++) 
    diffImg[i].Allocate(w, h, p, false, memory + (nb+i)*p*h); 
  tempImg.Allocate(w, h, p, false, memory + (nb+nd)*p*h);
  sift.Allocate(maxPts, 7, maxPts, false, memory + (nb+nd+1)*p*h);
  desc.Allocate(128, maxPts, 128, false, memory + (nb+nd+1)*p*h + maxPts*7);
  //checkMsg("Memory allocation failed\n");
  //safeCall(cudaThreadSynchronize());

  int totPts = 0;
  safeCall(cudaMemcpyToSymbol(d_PointCounter, &totPts, sizeof(int)));
  //std::cout << "allocate: " << timer0.read() << std::endl;

  TimerGPU timer1;
  float diffScale = pow(2.0f, 1.0f/NUM_SCALES);
  LowPassMulti(blurImg, img, diffImg, baseBlur, diffScale, initBlur);
  //std::cout << "lowpass:  " << timer1.read() << std::endl;
  TimerGPU timer2;
  SubtractMulti(diffImg, blurImg);
  //std::cout << "subtract: " << timer2.read() << std::endl;
  TimerGPU timer3;
  double sigma = baseBlur*diffScale;
  FindPointsMulti(diffImg, sift, thresh, maxPts, 16.0f, sigma, 1.0f/NUM_SCALES, lowestScale/subsampling);
  //std::cout << "points:   " << timer3.read() << std::endl;
  double gpuTimeDoG = timer1.read();

  TimerGPU timer4;
  safeCall(cudaMemcpyFromSymbol(&totPts, d_PointCounter, sizeof(int)));
  totPts = (totPts>=maxPts ? maxPts-1 : totPts);
  if (totPts>0) {
    ComputeOrientations(img, sift, totPts, maxPts); 
    SecondOrientations(sift, &totPts, maxPts);
    ExtractSiftDescriptors(img, sift, desc, totPts, maxPts); 
    AddSiftData(siftData, sift.d_data, desc.d_data, totPts, maxPts, subsampling);
  }
  safeCall(cudaThreadSynchronize());
  safeCall(cudaFree(memory));
  double gpuTimeSift = timer4.read();
  //std::cout << "sift:     " << gpuTimeSift << std::endl;

  double totTime = timer0.read();
#ifdef VERBOSE
    printf("GPU time : %.2f ms + %.2f ms + %.2f ms = %.2f ms\n", totTime-gpuTimeDoG-gpuTimeSift, gpuTimeDoG, gpuTimeSift, totTime);
    if (totPts>0) 
      printf("           %.2f ms / DoG,  %.4f ms / Sift,  #Sift = %d\n", gpuTimeDoG/NUM_SCALES, gpuTimeSift/totPts, totPts); 
#endif
}

void InitSiftData(SiftData &data, int num, bool host, bool dev)
{
  data.numPts = 0;
  data.maxPts = num;
  int sz = sizeof(SiftPoint)*num;
  data.h_data = NULL;
  if (host)
    data.h_data = (SiftPoint *)malloc(sz);
  data.d_data = NULL;
  if (dev)
    safeCall(cudaMalloc((void **)&data.d_data, sz));
}

void FreeSiftData(SiftData &data)
{
  if (data.d_data!=NULL)
    safeCall(cudaFree(data.d_data));
  data.d_data = NULL;
  if (data.h_data!=NULL)
    free(data.h_data);
  data.numPts = 0;
  data.maxPts = 0;
}

double AddSiftData(SiftData &data, float *d_sift, float *d_desc, int numPts, int maxPts, float subsampling)
{
  int newNum = data.numPts + numPts;
  if (data.maxPts < newNum) {
    int newMaxNum = 2*data.maxPts;
    while (newNum>newMaxNum)
      newMaxNum *= 2;
    if (data.h_data!=NULL) {
      SiftPoint *h_data = (SiftPoint *)malloc(sizeof(SiftPoint)*newMaxNum);
      memcpy(h_data, data.h_data, sizeof(SiftPoint)*data.numPts);
      free(data.h_data);
      data.h_data = h_data;
    }
    if (data.d_data!=NULL) {
      SiftPoint *d_data = NULL;
      safeCall(cudaMalloc((void**)&d_data, sizeof(SiftPoint)*newMaxNum));
      safeCall(cudaMemcpy(d_data, data.d_data, sizeof(SiftPoint)*data.numPts, cudaMemcpyDeviceToDevice));
      safeCall(cudaFree(data.d_data));
      data.d_data = d_data;
    }
    data.maxPts = newMaxNum;
  }
  int pitch = sizeof(SiftPoint);
  float *buffer = (float *)malloc(sizeof(float)*3*numPts);
  int bwidth = sizeof(float)*numPts; 
  safeCall(cudaMemcpy2D(buffer, bwidth, d_sift, sizeof(float)*maxPts, bwidth, 3, cudaMemcpyDeviceToHost));
  for (int i=0;i<3*numPts;i++) 
    buffer[i] *= subsampling;
  safeCall(cudaMemcpy2D(d_sift, sizeof(float)*maxPts, buffer, bwidth, bwidth, 3, cudaMemcpyHostToDevice));
  safeCall(cudaThreadSynchronize());
  if (data.h_data!=NULL) {
    float *ptr = (float*)&data.h_data[data.numPts];
    for (int i=0;i<6;i++)
      safeCall(cudaMemcpy2D(&ptr[i], pitch, &d_sift[i*maxPts], 4, 4, numPts, cudaMemcpyDeviceToHost));
    safeCall(cudaMemcpy2D(&ptr[16], pitch, d_desc, sizeof(float)*128, sizeof(float)*128, numPts, cudaMemcpyDeviceToHost));
  }
  if (data.d_data!=NULL) {
    float *ptr = (float*)&data.d_data[data.numPts];
    for (int i=0;i<6;i++)
      safeCall(cudaMemcpy2D(&ptr[i], pitch, &d_sift[i*maxPts], 4, 4, numPts, cudaMemcpyDeviceToDevice));
    safeCall(cudaMemcpy2D(&ptr[16], pitch, d_desc, sizeof(float)*128, sizeof(float)*128, numPts, cudaMemcpyDeviceToDevice));
  }
  data.numPts = newNum;
  free(buffer);
  return 0.0;
}

void PrintSiftData(SiftData &data)
{
  SiftPoint *h_data = data.h_data;
  if (data.h_data==NULL) {
    h_data = (SiftPoint *)malloc(sizeof(SiftPoint)*data.maxPts);
    safeCall(cudaMallocHost((void **)&h_data, sizeof(SiftPoint)*data.maxPts));
    safeCall(cudaMemcpy(h_data, data.d_data, sizeof(SiftPoint)*data.numPts, cudaMemcpyDeviceToHost));
    data.h_data = h_data;
  }
  for (int i=0;i<data.numPts;i++) {
    printf("xpos         = %.2f\n", h_data[i].xpos);
    printf("ypos         = %.2f\n", h_data[i].ypos);
    printf("scale        = %.2f\n", h_data[i].scale);
    printf("sharpness    = %.2f\n", h_data[i].sharpness);
    printf("edgeness     = %.2f\n", h_data[i].edgeness);
    printf("orientation  = %.2f\n", h_data[i].orientation);
    printf("score/second = %.2f\n", h_data[i].score);
#if 0
    float *siftData = (float*)&h_data[i].data;
    for (int j=0;j<8;j++) {
      if (j==0) 
	printf("data = ");
      else 
	printf("       ");
      for (int k=0;k<16;k++)
	if (siftData[j+8*k]<0.05)
	  printf(" .   ", siftData[j+8*k]);
	else
	  printf("%.2f ", siftData[j+8*k]);
      printf("\n");
    }
#endif
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
  dim3 threads(SCALEDOWN_W + WARP_SIZE + 2);
  ScaleDown<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch); 
  checkMsg("ScaleDown() execution failed\n");
  safeCall(cudaThreadSynchronize());
  return 0.0;
}

double Subtract(CudaImage &res, CudaImage &srcA, CudaImage &srcB)
{    
  int w = res.width;
  int p = res.pitch;
  int h = res.height;
  if (res.d_data==NULL || srcA.d_data==NULL || srcB.d_data==NULL) {
    printf("Subtract: missing data\n");
    return 0.0;
  }
  dim3 blocks(iDivUp(w, SUBTRACT_W), iDivUp(h, SUBTRACT_H));
  dim3 threads(SUBTRACT_W, SUBTRACT_H);
  Subtract<<<blocks, threads>>>(res.d_data, srcA.d_data, srcB.d_data, w, p, h);
  checkMsg("Subtract() execution failed\n");
  safeCall(cudaThreadSynchronize());
  return 0.0;
}

double FindPoints(CudaImage &data1, CudaImage &data2, CudaImage &data3, CudaImage &sift, float thresh, int maxPts, float edgeLimit, float scale, float factor)
{
  if (data1.d_data==NULL || data2.d_data==NULL || data3.d_data==NULL) {
    printf("FindPoints: missing data\n");
    return 0.0;
  }
  int w = data1.width;
  int p = data1.pitch;
  int h = data1.height;
  float threshs[2] = { thresh, -thresh };
  safeCall(cudaMemcpyToSymbol(d_Threshold, &threshs, 2*sizeof(float)));
  safeCall(cudaMemcpyToSymbol(d_EdgeLimit, &edgeLimit, sizeof(float)));
  safeCall(cudaMemcpyToSymbol(d_Scales, &scale, sizeof(float)));
  safeCall(cudaMemcpyToSymbol(d_Factor, &factor, sizeof(float)));
  safeCall(cudaMemcpyToSymbol(d_MaxNumPoints, &maxPts, sizeof(int)));

  dim3 blocks(iDivUp(w, MINMAX_W), iDivUp(h, MINMAX_H));
  dim3 threads(MINMAX_W + 2); 
  FindPoints<<<blocks, threads>>>(data1.d_data, data2.d_data, data3.d_data, sift.d_data, w, p, h); 
  checkMsg("FindPoints() execution failed\n");
  safeCall(cudaThreadSynchronize());
  return 0.0;
}

double ComputeOrientations(CudaImage &img, CudaImage &sift, int numPts, int maxPts)
{
  int p = img.pitch;
  int h = img.height;
  dim3 blocks(numPts);
  dim3 threads(32);
  ComputeOrientations<<<blocks, threads>>>(img.d_data, sift.d_data, maxPts, p, h);
  checkMsg("ComputeOrientations() execution failed\n");
  safeCall(cudaThreadSynchronize());
  return 0.0;
}

double SecondOrientations(CudaImage &sift, int *initNumPts, int maxPts) 
{
  int numPts = *initNumPts;
  int numPts2 = 2*numPts;
  float *d_sift = sift.d_data;
  int bw = sizeof(float)*numPts2;
  float *h_sift = (float *)malloc(7*bw);
  safeCall(cudaMemcpy2D(h_sift, bw, d_sift, sizeof(float)*maxPts, sizeof(float)*numPts, 7, cudaMemcpyDeviceToHost));
  int num = numPts;  
  for (int i=0;i<numPts;i++) {
    if (h_sift[6*numPts2+i]>=0.0f && num<maxPts) {
      for (int j=0;j<5;j++) 
	h_sift[j*numPts2+num] = h_sift[j*numPts2+i];
      h_sift[5*numPts2+num] = h_sift[6*numPts2+i];
      h_sift[6*numPts2+num] = -1.0f;
      num ++;
    }
  }
  safeCall(cudaMemcpy2D(&d_sift[numPts], sizeof(float)*maxPts, &h_sift[numPts], bw, sizeof(float)*(num-numPts), 7, cudaMemcpyHostToDevice));
  free(h_sift);
  *initNumPts = num;
  return 0.0;
}

double ExtractSiftDescriptors(CudaImage &img, CudaImage &sift, CudaImage &desc, int numPts, int maxPts)
{
  float *d_sift = sift.d_data, *d_desc = desc.d_data;
  tex.addressMode[0] = cudaAddressModeClamp;
  tex.addressMode[1] = cudaAddressModeClamp;
  tex.filterMode = cudaFilterModeLinear; 
  tex.normalized = false;
  size_t offset = 0;
  safeCall(cudaBindTexture2D(&offset, tex, img.d_data, tex.channelDesc, img.width, img.height, img.pitch*sizeof(float)));
   
  dim3 blocks(numPts); 
  dim3 threads(16);
  ExtractSiftDescriptors<<<blocks, threads>>>(img.d_data, d_sift, d_desc, maxPts);
  checkMsg("ExtractSiftDescriptors() execution failed\n");
  safeCall(cudaThreadSynchronize());
  safeCall(cudaUnbindTexture(tex));
  return 0.0; 
}

//==================== Multi-scale functions ===================//

double SubtractMulti(CudaImage *results, CudaImage *sources)
{    
  int w = results[0].width;
  int p = results[0].pitch;
  int h = results[0].height;
  if (results->d_data==NULL || sources->d_data==NULL) {
    printf("SubtractMulti: missing data\n");
    return 0.0;
  }
  dim3 blocks(iDivUp(w, SUBTRACTM_W), iDivUp(h, SUBTRACTM_H));
  dim3 threads(SUBTRACTM_W, SUBTRACTM_H, NUM_SCALES + 2); 
  SubtractMulti<<<blocks, threads>>>(results[0].d_data, sources[0].d_data, w, p, h);
  checkMsg("SubtractMulti() execution failed\n");
  safeCall(cudaThreadSynchronize());
  return 0.0;
}

double FindPointsMulti(CudaImage *sources, CudaImage &sift, float thresh, int maxPts, float edgeLimit, float scale, float factor, float lowestScale)
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
  safeCall(cudaMemcpyToSymbol(d_MaxNumPoints, &maxPts, sizeof(int)));

  if (nScales>0) {
    dim3 blocks(iDivUp(w, MINMAX_W)*nScales, iDivUp(h, MINMAX_H));
    dim3 threads(MINMAX_W + 2); 
    FindPointsMulti<<<blocks, threads>>>(sources->d_data, sift.d_data, w, p, h, nScales); 
    checkMsg("FindPointsMulti() execution failed\n");
    safeCall(cudaThreadSynchronize());
  }
  return 0.0;
}

#define RADIUS 4

double LowPassMulti(CudaImage *results, CudaImage &origImg, CudaImage *tempImg, float baseBlur, float diffScale, float initBlur)
{
  float *d_DataA = origImg.d_data;
  float *d_DataB = results[0].d_data;
  float *d_Temp = tempImg[0].d_data;
  if (d_DataA==NULL || d_DataB==NULL || d_Temp==NULL) {
    printf("LowPass9: missing data\n");
    return 0.0;
  } 
  float kernel[12*16];
  float scale = baseBlur;
  for (int i=0;i<NUM_SCALES+3;i++) {
    float kernelSum = 0.0f;
    float var = scale*scale - initBlur*initBlur;
    for (int j=-RADIUS;j<=RADIUS;j++) {
      kernel[16*i+j+RADIUS] = (float)expf(-(double)j*j/2.0/var);      
      kernelSum += kernel[16*i+j+RADIUS];
    }
    for (int j=-RADIUS;j<=RADIUS;j++) 
      kernel[16*i+j+RADIUS] /= kernelSum;  
    scale *= diffScale;
  }
  safeCall(cudaMemcpyToSymbol(d_Kernel, kernel, 12*16*sizeof(float)));
    
  int width = results[0].width;
  int pitch = results[0].pitch;
  int height = results[0].height;
  dim3 blockGridRows(iDivUp(width, CONVROW_W)*(NUM_SCALES + 3), height);
  dim3 threadBlockRows(CONVROW_W + 2*RADIUS); 
  LowPassRowMulti<<<blockGridRows, threadBlockRows>>>(d_Temp, d_DataA, width, pitch, height);
  checkMsg("ConvRowGPU() execution failed\n");
  safeCall(cudaThreadSynchronize());
  dim3 blockGridColumns(iDivUp(width, CONVCOL_W)*(NUM_SCALES + 3), iDivUp(height, CONVCOL_H));
  dim3 threadBlockColumns(CONVCOL_W, CONVCOL_S);
  LowPassColMulti<<<blockGridColumns, threadBlockColumns>>>(d_DataB, d_Temp, width, pitch, height); 
  checkMsg("ConvColGPU() execution failed\n");
  safeCall(cudaThreadSynchronize());
  return 0.0; 
}
 
