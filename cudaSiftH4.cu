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

#include "cudaSiftD4.cu"

//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>

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
#if 0
  int device; 
  struct cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int align = prop.texturePitchAlignment;
  printf("ALIGN=%d\n", align);
#endif
  float *memoryTmp = NULL;
  //safeCall(cudaMalloc((void **)&memoryTmp, sizeof(float)*size));
  size_t pitch;
  safeCall(cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size+4095)/4096*sizeof(float)));
  float *memorySub = memoryTmp + sizeTmp;

  ExtractSiftLoop(siftData, img, numOctaves, initBlur, thresh, lowestScale, subsampling, memoryTmp, memorySub);
  safeCall(cudaMemcpyFromSymbol(&siftData.numPts, d_PointCounter, sizeof(int)));
  siftData.numPts = (siftData.numPts<siftData.maxPts ? siftData.numPts : siftData.maxPts);
  double totTime = timer.read();
  //PrintSiftData(siftData);
  safeCall(cudaDeviceSynchronize());
  safeCall(cudaFree(memoryTmp));
#ifdef VERBOSE
  printf("ExtractSift time total =      %.2f ms\n\n", totTime);
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
  //safeCall(cudaDeviceSynchronize());
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
  //float *memory = NULL;
  int p = iAlignUp(w, 128);
  //int allocSize = (nb+nd)*p*h;
  //safeCall(cudaMalloc((void **)&memory, sizeof(float)*allocSize));
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
#if 0
  LowPassMulti(blurImg, img, diffImg, baseBlur, diffScale, initBlur);
  SubtractMulti(diffImg, blurImg);
#else
  LaplaceMulti(diffImg, img, baseBlur, diffScale, initBlur);
#endif
  int fstPts = 0;
  safeCall(cudaMemcpyFromSymbol(&fstPts, d_PointCounter, sizeof(int)));
  double sigma = baseBlur*diffScale;
  FindPointsMulti(diffImg, siftData, thresh, 1.0f/0.07f, sigma, 1.0f/NUM_SCALES, lowestScale/subsampling);
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
  //safeCall(cudaFree(memory));
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
#if 0
  data.h_data = NULL;
  if (host)
    data.h_data = (SiftPoint *)malloc(sz);
  data.d_data = NULL;
  if (dev)
    safeCall(cudaMalloc((void **)&data.d_data, sz));
#else
  safeCall(cudaMallocManaged((void **)&data.m_data, sz));
#endif
}

void FreeSiftData(SiftData &data)
{
#if 0
  if (data.d_data!=NULL)
    safeCall(cudaFree(data.d_data));
  data.d_data = NULL;
  if (data.h_data!=NULL)
    free(data.h_data);
#else
  safeCall(cudaFree(data.m_data));
#endif
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
#if 0
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
#else
    SiftPoint *d_data = NULL;
    safeCall(cudaMallocManaged((void**)&d_data, sizeof(SiftPoint)*newMaxNum));
    safeCall(cudaMemcpy(d_data, data.m_data, sizeof(SiftPoint)*data.numPts, cudaMemcpyDeviceToDevice));
    safeCall(cudaFree(data.m_data));
    data.m_data = d_data;
    data.maxPts = newMaxNum;
#endif
  }
  int pitch = sizeof(SiftPoint);
  float *buffer = (float *)malloc(sizeof(float)*3*numPts);
  int bwidth = sizeof(float)*numPts; 
  safeCall(cudaMemcpy2D(buffer, bwidth, d_sift, sizeof(float)*maxPts, bwidth, 3, cudaMemcpyDeviceToHost));
  for (int i=0;i<3*numPts;i++) 
    buffer[i] *= subsampling;
  safeCall(cudaMemcpy2D(d_sift, sizeof(float)*maxPts, buffer, bwidth, bwidth, 3, cudaMemcpyHostToDevice));
  //safeCall(cudaThreadSynchronize());
#if 0
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
#else
  float *ptr = (float*)&data.m_data[data.numPts];
  for (int i=0;i<6;i++)
    safeCall(cudaMemcpy2D(&ptr[i], pitch, &d_sift[i*maxPts], 4, 4, numPts, cudaMemcpyDeviceToDevice));
  safeCall(cudaMemcpy2D(&ptr[16], pitch, d_desc, sizeof(float)*128, sizeof(float)*128, numPts, cudaMemcpyDeviceToDevice));
#endif
  data.numPts = newNum;
  free(buffer);
  return 0.0;
}

void PrintSiftData(SiftData &data)
{
#if 0
  SiftPoint *h_data = data.h_data;
  if (data.h_data==NULL) {
    h_data = (SiftPoint *)malloc(sizeof(SiftPoint)*data.maxPts);
    safeCall(cudaMallocHost((void **)&h_data, sizeof(SiftPoint)*data.maxPts));
    safeCall(cudaMemcpy(h_data, data.d_data, sizeof(SiftPoint)*data.numPts, cudaMemcpyDeviceToHost));
    data.h_data = h_data;
  }
#else
  SiftPoint *h_data = data.m_data;
#endif
  for (int i=0;i<data.numPts;i++) {
    printf("xpos         = %.2f\n", h_data[i].xpos);
    printf("ypos         = %.2f\n", h_data[i].ypos);
    printf("scale        = %.2f\n", h_data[i].scale);
    printf("sharpness    = %.2f\n", h_data[i].sharpness);
    printf("edgeness     = %.2f\n", h_data[i].edgeness);
    printf("orientation  = %.2f\n", h_data[i].orientation);
    printf("score        = %.2f\n", h_data[i].score);
#if 0
    float *siftData = (float*)&h_data[i].m_data;
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
  //safeCall(cudaThreadSynchronize());
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
  //safeCall(cudaThreadSynchronize());
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
  //safeCall(cudaThreadSynchronize());
  return 0.0;
}

double ComputeOrientations(CudaImage &img, SiftData &siftData, int fstPts, int totPts)
{
  int p = img.pitch;
  int h = img.height;
  dim3 blocks(totPts - fstPts);
  dim3 threads(128);
  ComputeOrientations2<<<blocks, threads>>>(img.d_data, siftData.m_data, fstPts);
  checkMsg("ComputeOrientations() execution failed\n");
  //safeCall(cudaThreadSynchronize());
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

double ExtractSiftDescriptors(CudaImage &img, SiftData &siftData, int fstPts, int totPts, float subsampling)
{
  dim3 blocks(totPts - fstPts); 
  dim3 threads(16, 16);
  ExtractSiftDescriptors2<<<blocks, threads>>>(img.d_data, siftData.m_data, fstPts, subsampling);
  checkMsg("ExtractSiftDescriptors() execution failed\n");
  //safeCall(cudaThreadSynchronize());
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
#if 0
  cv::Mat limg(h, w, CV_32F);
  safeCall(cudaMemcpy2D(limg.data, w*4, results[5].d_data, p*4, w*4, h, cudaMemcpyDeviceToHost));
  cv::imwrite("dump2.pgm", limg*8+128);
#endif
  //safeCall(cudaThreadSynchronize());
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
    FindPointsMulti2<<<blocks, threads>>>(sources->d_data, siftData.m_data, w, p, h, nScales); 
    checkMsg("FindPointsMulti() execution failed\n");
    //safeCall(cudaThreadSynchronize());
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
      //kernel[16*i+j+RADIUS] = (j==0 ? 1.0f : 0.0f);      
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
  //safeCall(cudaThreadSynchronize());
  dim3 blockGridColumns(iDivUp(width, CONVCOL_W)*(NUM_SCALES + 3), iDivUp(height, CONVCOL_H));
  dim3 threadBlockColumns(CONVCOL_W, CONVCOL_S);
  LowPassColMulti<<<blockGridColumns, threadBlockColumns>>>(d_DataB, d_Temp, width, pitch, height); 
  checkMsg("ConvColGPU() execution failed\n");
#if 0
  cv::Mat limg(height, width, CV_32F);
  safeCall(cudaMemcpy2D(limg.data, width*4, d_DataB, pitch*4, width*4, height, cudaMemcpyDeviceToHost));
  cv::imwrite("dump.pgm", limg);
#endif

  //safeCall(cudaThreadSynchronize());
  return 0.0; 
}
 
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
    for (int j=-RADIUS;j<=RADIUS;j++) {
      kernel[16*i+j+RADIUS] = (float)expf(-(double)j*j/2.0/var);
      //kernel[16*i+j+RADIUS] = (j==0 ? 1.0f : 0.0f);      
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
  dim3 blocks(iDivUp(width+2*RADIUS, LAPLACE_W), height);
  dim3 threads(LAPLACE_W+2*RADIUS, LAPLACE_S); 
  LaplaceMulti<<<blocks, threads>>>(d_DataB, d_DataA, width, pitch, height);
  checkMsg("ConvRowGPU() execution failed\n");
#if 0
  cv::Mat limg(height, width, CV_32F);
  safeCall(cudaMemcpy2D(limg.data, width*4, results[5].d_data, pitch*4, width*4, height, cudaMemcpyDeviceToHost));
  cv::imwrite("dump1.pgm", limg*8+128);
#endif
  safeCall(cudaThreadSynchronize());
  return 0.0; 
}

