//********************************************************//
// CUDA SIFT extractor by Mårten Björkman aka Celebrandil //
//********************************************************//  

#include <cstdio>
#include <cstring>
#include <iostream>
#include <cudautils.h>

#undef VERBOSE

#include "cudaImage.h"
#include "cudaSift.h"
#include "cudaSiftD.h"
#include "cudaSiftH.h"

#include "cudaSiftD.cu"

void InitCuda()
{
  deviceInit(0);
}

void ExtractSift(SiftData *siftData, CudaImage *img, int numLayers, 
  int numOctaves, double initBlur, float thresh, float subsampling) 
{
  TimerGPU timer(0);
  int w = img->width;
  int h = img->height;
  int p = img->pitch;
  if (numOctaves>1) {
    CudaImage subImg;
    AllocCudaImage(&subImg, w/2, h/2, p/2, false, true);
    ScaleDown(&subImg, img, 0.5f);
    float totInitBlur = (float)sqrt(initBlur*initBlur + 0.5f*0.5f) / 2.0f;
    ExtractSift(siftData, &subImg, numLayers, numOctaves-1, totInitBlur, 
      thresh, subsampling*2.0f);
    FreeCudaImage(&subImg);
  }
  ExtractSiftOctave(siftData, img, numLayers, initBlur, thresh, subsampling);
  double totTime = timer.read();
  printf("GPU time total : %.2f ms\n\n", totTime);
}

void ExtractSiftOctave(SiftData *siftData, CudaImage *img, int numLayers, 
  double initBlur, float thresh, float subsampling)
{
  const double baseBlur = 1.0;
  const int maxPts = 512;
  int w = img->width; 
  int h = img->height;
  int p = img->pitch;
  // Images and arrays
  CudaImage blurImg[2];
  CudaImage diffImg[3];
  CudaImage tempImg;
  CudaImage textImg;
  CudaArray sift; // { xpos, ypos, scale, strength, edge, orient1, orient2 };
  CudaArray desc;
  CudaArray minmax;

  TimerGPU timer(0);
  AllocCudaImage(&blurImg[0], w, h, p, false, true);   
  AllocCudaImage(&blurImg[1], w, h, p, false, true); 
  AllocCudaImage(&diffImg[0], w, h, p, false, true);
  AllocCudaImage(&diffImg[1], w, h, p, false, true);
  AllocCudaImage(&diffImg[2], w, h, p, false, true);
  AllocCudaImage(&tempImg, w, h, p, false, true);
  AllocCudaImage(&minmax, w, iDivUp(h,32), w, true, true);
  AllocCudaImage(&sift, maxPts, 7, maxPts, true, true);
  AllocCudaImage(&desc, 128, maxPts, 128, true, true);
  int *ptrs = 0; 
  ptrs = (int *)malloc(sizeof(int)*maxPts);
  //safeCall(cudaMallocHost((void **)&ptrs, sizeof(int)*maxPts));
  AllocCudaImage(&textImg, w, h, p, false, false);     
  InitTexture(&textImg);
  checkMsg("Memory allocation failed\n");
  safeCall(cudaThreadSynchronize());

  double var = baseBlur*baseBlur - initBlur*initBlur;
  double gpuTime = 0.0, gpuTimeDoG = 0.0, gpuTimeSift = 0.0;
  int totPts = 0;

  if (var>0.0)
    gpuTime += LowPass<2>(&blurImg[0], img, &tempImg, var);
  for (int i=0;i<numLayers+2;i++) {
    double pSigma = baseBlur*pow(2.0, (double)(i-1)/numLayers);
    double oSigma = baseBlur*pow(2.0, (double)i/numLayers);
    double nSigma = baseBlur*pow(2.0, (double)(i+1)/numLayers);
    double var = nSigma*nSigma - oSigma*oSigma;
    if (i<=1) { 
      gpuTime += LowPass<7>(&blurImg[(i+1)%2], &blurImg[i%2], &tempImg, var);
      gpuTime += Subtract(&diffImg[(i+2)%3], &blurImg[i%2], &blurImg[(i+1)%2]);
    } else {
      gpuTimeDoG += LowPass<7>(&blurImg[(i+1)%2], &blurImg[i%2], &tempImg,var);
      gpuTimeDoG += Subtract(&diffImg[(i+2)%3], &blurImg[i%2], &blurImg[(i+1)%2]);
      gpuTimeDoG += Find3DMinMax(&minmax, &diffImg[(i+2)%3], &diffImg[(i+1)%3], &diffImg[i%3], thresh, maxPts);
      int numPts = 0;
      gpuTimeSift += UnpackPointers(&minmax, maxPts, ptrs, &numPts);
      if (numPts>0) {
	gpuTimeDoG += CopyToTexture(&blurImg[i%2], &textImg, false);
	gpuTimeSift += ComputePositions(&diffImg[(i+2)%3], &diffImg[(i+1)%3], &diffImg[i%3], ptrs, &sift, numPts, maxPts, pSigma, 1.0f/numLayers);
	gpuTimeSift += RemoveEdgePoints(&sift, &numPts, maxPts, 10.0f);
	gpuTimeSift += ComputeOrientations(&blurImg[i%2], ptrs, &sift, numPts, maxPts);
	gpuTimeSift += SecondOrientations(&sift, &numPts, maxPts);
	gpuTimeSift += ExtractSiftDescriptors(&textImg, &sift, &desc, numPts, maxPts);
	gpuTimeSift += AddSiftData(siftData, sift.d_data, desc.d_data, numPts, maxPts, subsampling);
      }
      totPts += numPts; 
    }
  }

#if 0
  Readback(&diffImg[(2+2)%3]);
  float *imd = diffImg[(2+2)%3].h_data;
  for (int i=0;i<w*h;i++) imd[i] = 8*imd[i] + 128;
  CUT_SAFE_CALL(cutSavePGMf("data/limg_test2.pgm", 
    diffImg[(2+2)%3].h_data, w, h));
  Readback(&diffImg[(2+1)%3]);
  imd = diffImg[(2+1)%3].h_data;
  for (int i=0;i<w*h;i++) imd[i] = 8*imd[i] + 128;
  CUT_SAFE_CALL(cutSavePGMf("data/limg_test1.pgm", 
    diffImg[(2+1)%3].h_data, w, h));
  Readback(&diffImg[(2+0)%3]);
  imd = diffImg[(2+0)%3].h_data;
  for (int i=0;i<w*h;i++) imd[i] = 8*imd[i] + 128;
  CUT_SAFE_CALL(cutSavePGMf("data/limg_test0.pgm", 
    diffImg[(2+0)%3].h_data, w, h));
#endif

  safeCall(cudaThreadSynchronize());
  FreeCudaImage(&textImg);
  free(ptrs);
  //safeCall(cudaFreeHost(ptrs));
  FreeCudaImage(&desc);
  FreeCudaImage(&sift);
  FreeCudaImage(&minmax);
  FreeCudaImage(&tempImg);
  FreeCudaImage(&diffImg[2]);
  FreeCudaImage(&diffImg[1]);
  FreeCudaImage(&diffImg[0]);
  FreeCudaImage(&blurImg[1]);
  FreeCudaImage(&blurImg[0]);

  double totTime = timer.read();
  printf("GPU time : %.2f ms + %.2f ms + %.2f ms = %.2f ms, %.2f ms\n", 
    gpuTime, gpuTimeDoG, gpuTimeSift, gpuTime+gpuTimeDoG+gpuTimeSift, totTime);
  if (totPts>0) 
    printf("           %.2f ms / DoG,  %.4f ms / Sift,  #Sift = %d\n", 
	   gpuTimeDoG/numLayers, gpuTimeSift/totPts, totPts); 
}

void InitSiftData(SiftData *data, int num, bool host, bool dev)
{
  data->numPts = 0;
  data->maxPts = num;
  int sz = sizeof(SiftPoint)*num;
  data->h_data = NULL;
  if (host)
    data->h_data = (SiftPoint *)malloc(sz);
  //safeCall(cudaMallocHost((void **)&data->h_data, sz));
  data->d_data = NULL;
  if (dev)
    safeCall(cudaMalloc((void **)&data->d_data, sz));
}

void FreeSiftData(SiftData *data)
{
  if (data->d_data!=NULL)
    safeCall(cudaFree(data->d_data));
  data->d_data = NULL;
  if (data->h_data!=NULL)
    free(data->h_data);
    //safeCall(cudaFreeHost(data->h_data));
  data->numPts = 0;
  data->maxPts = 0;
}

double AddSiftData(SiftData *data, float *d_sift, float *d_desc, 
  int numPts, int maxPts, float subsampling)
{
  TimerGPU timer(0);
  int newNum = data->numPts + numPts;
  if (data->maxPts<(data->numPts+numPts)) {
    int newMaxNum = 2*data->maxPts;
    while (newNum>newMaxNum)
      newMaxNum *= 2;
    if (data->h_data!=NULL) {
      SiftPoint *h_data = 0;
      h_data = (SiftPoint *)malloc(sizeof(SiftPoint)*newMaxNum);
      //safeCall(cudaMallocHost((void**)&h_data, 
      //sizeof(SiftPoint)*newMaxNum));
      memcpy(h_data, data->h_data, sizeof(SiftPoint)*data->numPts);
      free(data->h_data);
      //safeCall(cudaFree(data->h_data));
      data->h_data = h_data;
    }
    if (data->d_data!=NULL) {
      SiftPoint *d_data = NULL;
      safeCall(cudaMalloc((void**)&d_data, 
	sizeof(SiftPoint)*newMaxNum));
      safeCall(cudaMemcpy(d_data, data->d_data, 
        sizeof(SiftPoint)*data->numPts, cudaMemcpyDeviceToDevice));
      safeCall(cudaFree(data->d_data));
      data->d_data = d_data;
    }
    data->maxPts = newMaxNum;
  }
  int pitch = sizeof(SiftPoint);
  float *buffer;
  buffer = (float *)malloc(sizeof(float)*3*numPts);
  //safeCall(cudaMallocHost((void**)&buffer, sizeof(float)*3*numPts));
  int bwidth = sizeof(float)*numPts;
  safeCall(cudaMemcpy2D(buffer, bwidth, d_sift, sizeof(float)*maxPts, 
    bwidth, 3, cudaMemcpyDeviceToHost));
  for (int i=0;i<3*numPts;i++) 
    buffer[i] *= subsampling;
  safeCall(cudaMemcpy2D(d_sift, sizeof(float)*maxPts, buffer, bwidth, 
    bwidth, 3, cudaMemcpyHostToDevice));
  safeCall(cudaThreadSynchronize());
  if (data->h_data!=NULL) {
    float *ptr = (float*)&data->h_data[data->numPts];
    for (int i=0;i<6;i++)
      safeCall(cudaMemcpy2D(&ptr[i], pitch, &d_sift[i*maxPts], 4, 4, 
	numPts, cudaMemcpyDeviceToHost));
    safeCall(cudaMemcpy2D(&ptr[16], pitch, d_desc, sizeof(float)*128, 
      sizeof(float)*128, numPts, cudaMemcpyDeviceToHost));
  }
  if (data->d_data!=NULL) {
    float *ptr = (float*)&data->d_data[data->numPts];
    for (int i=0;i<6;i++)
      safeCall(cudaMemcpy2D(&ptr[i], pitch, &d_sift[i*maxPts], 4, 4, 
	numPts, cudaMemcpyDeviceToDevice));
    safeCall(cudaMemcpy2D(&ptr[16], pitch, d_desc, sizeof(float)*128, 
      sizeof(float)*128, numPts, cudaMemcpyDeviceToDevice));
  }
  data->numPts = newNum;
  //safeCall(cudaFreeHost(buffer));
  free(buffer);
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("AddSiftData time =            %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}

void PrintSiftData(SiftData *data)
{
  SiftPoint *h_data = data->h_data;
  if (data->h_data==NULL) {
    h_data = (SiftPoint *)malloc(sizeof(SiftPoint)*data->maxPts);
    //safeCall(cudaMallocHost((void **)&h_data, 
    //  sizeof(SiftPoint)*data->maxPts));
    safeCall(cudaMemcpy(h_data, data->d_data, 
      sizeof(SiftPoint)*data->numPts, cudaMemcpyDeviceToHost));
    data->h_data = h_data;
  }
  for (int i=0;i<data->numPts;i++) {
    printf("xpos         = %.2f\n", h_data[i].xpos);
    printf("ypos         = %.2f\n", h_data[i].ypos);
    printf("scale        = %.2f\n", h_data[i].scale);
    printf("sharpness    = %.2f\n", h_data[i].sharpness);
    printf("edgeness     = %.2f\n", h_data[i].edgeness);
    printf("orientation  = %.2f\n", h_data[i].orientation);
    float *siftData = (float*)&h_data[i].data;
    for (int j=0;j<8;j++) {
      if (j==0) printf("data = ");
      else printf("       ");
      for (int k=0;k<16;k++)
	if (siftData[j+8*k]<0.05)
	  printf(" .   ", siftData[j+8*k]);
	else
	  printf("%.2f ", siftData[j+8*k]);
      printf("\n");
    }
  }
  printf("Number of available points: %d\n", data->numPts);
  printf("Number of allocated points: %d\n", data->maxPts);
}

double MatchSiftData(SiftData *data1, SiftData *data2)
{
  if (data1->d_data==NULL || data2->d_data==NULL)
    return 0.0f;
  TimerGPU timer(0);
  int numPts1 = data1->numPts;
  int numPts2 = data2->numPts;
  SiftPoint *sift1 = data1->d_data;
  SiftPoint *sift2 = data2->d_data;
  
  float *d_corrData; 
  int corrWidth = iDivUp(numPts2, 16)*16;
  int corrSize = sizeof(float)*numPts1*corrWidth;
  safeCall(cudaMalloc((void **)&d_corrData, corrSize));
  dim3 blocks(numPts1, iDivUp(numPts2, 16));
  dim3 threads(16, 16); // each block: 1 points x 16 points
  MatchSiftPoints<<<blocks, threads>>>(sift1, sift2, 
    d_corrData, numPts1, numPts2);
  safeCall(cudaThreadSynchronize());
  dim3 blocksMax(iDivUp(numPts1, 16));
  dim3 threadsMax(16, 16);
  FindMaxCorr<<<blocksMax, threadsMax>>>(d_corrData, sift1, sift2, 
    numPts1, corrWidth, sizeof(SiftPoint));
  safeCall(cudaThreadSynchronize());
  checkMsg("MatchSiftPoints() execution failed\n");
  safeCall(cudaFree(d_corrData));
  if (data1->h_data!=NULL) {
    float *h_ptr = &data1->h_data[0].score;
    float *d_ptr = &data1->d_data[0].score;
    safeCall(cudaMemcpy2D(h_ptr, sizeof(SiftPoint), d_ptr,
      sizeof(SiftPoint), 5*sizeof(float), data1->numPts, 
      cudaMemcpyDeviceToHost));
  }

  double gpuTime = timer.read();
  //#ifdef VERBOSE
  printf("MatchSiftData time =          %.2f msec\n", gpuTime);
  //#endif
  return gpuTime;
}		 
  
double FindHomography(SiftData *data, float *homography, int *numMatches, 
  int numLoops, float minScore, float maxAmbiguity, float thresh)
{
  *numMatches = 0;
  homography[0] = homography[4] = homography[8] = 1.0f;
  homography[1] = homography[2] = homography[3] = 0.0f;
  homography[5] = homography[6] = homography[7] = 0.0f;
  if (data->d_data==NULL)
    return 0.0f;
  SiftPoint *d_sift = data->d_data;
  TimerGPU timer(0);
  numLoops = iDivUp(numLoops,16)*16;
  int numPts = data->numPts;
  if (numPts<8)
    return 0.0f;
  int numPtsUp = iDivUp(numPts, 16)*16;
  float *d_coord, *d_homo;
  int *d_randPts, *h_randPts;
  int randSize = 4*sizeof(int)*numLoops;
  int szFl = sizeof(float);
  int szPt = sizeof(SiftPoint);
  safeCall(cudaMalloc((void **)&d_coord, 4*sizeof(float)*numPtsUp));
  safeCall(cudaMalloc((void **)&d_randPts, randSize));
  safeCall(cudaMalloc((void **)&d_homo, 8*sizeof(float)*numLoops));
  h_randPts = (int*)malloc(randSize);
  float *h_scores = (float *)malloc(sizeof(float)*numPtsUp);
  float *h_ambiguities = (float *)malloc(sizeof(float)*numPtsUp);
  safeCall(cudaMemcpy2D(h_scores, szFl, &d_sift[0].score, szPt, 
    szFl, numPts, cudaMemcpyDeviceToHost));
  safeCall(cudaMemcpy2D(h_ambiguities, szFl, &d_sift[0].ambiguity, szPt,
    szFl, numPts, cudaMemcpyDeviceToHost));
  int *validPts = (int *)malloc(sizeof(int)*numPts);
  int numValid = 0;
  for (int i=0;i<numPts;i++) {
    if (h_scores[i]>minScore && h_ambiguities[i]<maxAmbiguity)
      validPts[numValid++] = i;
  }
  free(h_scores);
  free(h_ambiguities);
  if (numValid>=8) {
    for (int i=0;i<numLoops;i++) {
      int p1 = rand() % numValid;
      int p2 = rand() % numValid;
      int p3 = rand() % numValid;
      int p4 = rand() % numValid;
      while (p2==p1) p2 = rand() % numValid;
      while (p3==p1 || p3==p2) p3 = rand() % numValid;
      while (p4==p1 || p4==p2 || p4==p3) p4 = rand() % numValid;
      h_randPts[i+0*numLoops] = validPts[p1];
      h_randPts[i+1*numLoops] = validPts[p2];
      h_randPts[i+2*numLoops] = validPts[p3];
      h_randPts[i+3*numLoops] = validPts[p4];
    }
    safeCall(cudaMemcpy(d_randPts, h_randPts, randSize, 
      cudaMemcpyHostToDevice));
    safeCall(cudaMemcpy2D(&d_coord[0*numPtsUp], szFl, &d_sift[0].xpos, 
      szPt, szFl, numPts, cudaMemcpyDeviceToDevice));
    safeCall(cudaMemcpy2D(&d_coord[1*numPtsUp], szFl, &d_sift[0].ypos, 
      szPt, szFl, numPts, cudaMemcpyDeviceToDevice));
    safeCall(cudaMemcpy2D(&d_coord[2*numPtsUp], szFl, 
      &d_sift[0].match_xpos, szPt, szFl, numPts, cudaMemcpyDeviceToDevice));
    safeCall(cudaMemcpy2D(&d_coord[3*numPtsUp], szFl, 
      &d_sift[0].match_ypos, szPt, szFl, numPts, cudaMemcpyDeviceToDevice));
    ComputeHomographies<<<numLoops/16, 16>>>(d_coord, d_randPts, 
      d_homo, numPtsUp);
    safeCall(cudaThreadSynchronize());
    checkMsg("ComputeHomographies() execution failed\n");
    dim3 blocks(1, numLoops/TESTHOMO_LOOPS);
    dim3 threads(TESTHOMO_TESTS, TESTHOMO_LOOPS);
    TestHomographies<<<blocks, threads>>>(d_coord, d_homo, 
      d_randPts, numPtsUp, thresh*thresh);
    safeCall(cudaThreadSynchronize());
    checkMsg("TestHomographies() execution failed\n");
    safeCall(cudaMemcpy(h_randPts, d_randPts, sizeof(int)*numLoops, 
			      cudaMemcpyDeviceToHost));
    int maxIndex = -1, maxCount = -1;
    for (int i=0;i<numLoops;i++) 
      if (h_randPts[i]>maxCount) {
	maxCount = h_randPts[i];
	maxIndex = i;
      }
    safeCall(cudaMemcpy2D(homography, szFl, &d_homo[maxIndex], 
      sizeof(float)*numLoops, szFl, 8, cudaMemcpyDeviceToHost));
  }
  free(validPts);
  free(h_randPts);
  safeCall(cudaFree(d_homo));
  safeCall(cudaFree(d_randPts));
  safeCall(cudaFree(d_coord));
  double gpuTime = timer.read();
  //#ifdef VERBOSE
  printf("FindHomography time =         %.2f msec\n", gpuTime);
  //#endif
  return gpuTime;
}

///////////////////////////////////////////////////////////////////////////////
// Host side master functions
///////////////////////////////////////////////////////////////////////////////

double LowPass5(CudaImage *res, CudaImage *data, float variance)
{
  int w = res->width;
  int h = res->height;
  TimerGPU timer(0);
  if (res->d_data==NULL || data->d_data==NULL) {
    printf("LowPass5: missing data\n");
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
  dim3 blocks(iDivUp(w, LOWPASS5_DX), iDivUp(h, LOWPASS5_DY));
  dim3 threads(LOWPASS5_DX + WARP_SIZE + 2);
  LowPass5<<<blocks, threads>>>(res->d_data, data->d_data, w, h); 
  checkMsg("LowPass5() execution failed\n");
  safeCall(cudaThreadSynchronize());

  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("LowPass5 time =               %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}

double ScaleDown(CudaImage *res, CudaImage *data, float variance)
{
  int w = data->width;
  int h = data->height;
  TimerGPU timer(0);
  if (res->d_data==NULL || data->d_data==NULL) {
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
  dim3 blocks(iDivUp(w, LOWPASS5_DX), iDivUp(h, LOWPASS5_DY));
  dim3 threads(LOWPASS5_DX + WARP_SIZE + 2);
  ScaleDown<<<blocks, threads>>>(res->d_data, data->d_data, w, h); 
  checkMsg("ScaleDown() execution failed\n");
  safeCall(cudaThreadSynchronize());

  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("ScaleDown time =              %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}

double Subtract(CudaImage *res, CudaImage *dataA, CudaImage *dataB)
{    
  int w = res->width;
  int h = res->height;
  TimerGPU timer(0);
  if (res->d_data==NULL || dataA->d_data==NULL || dataB->d_data==NULL) {
    printf("Subtract: missing data\n");
    return 0.0;
  }
  dim3 blocks(iDivUp(w, 16), iDivUp(h, 16));
  dim3 threads(16, 16);
  Subtract<<<blocks, threads>>>(res->d_data, dataA->d_data, dataB->d_data, 
				w, h);
  checkMsg("Subtract() execution failed\n");
  safeCall(cudaThreadSynchronize());

  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("Subtract time =               %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}

double MultiplyAdd(CudaImage *res, CudaImage *data, float constA, float constB)
{
  int w = res->width;
  int h = res->height;
  TimerGPU timer(0);
  if (res->d_data==NULL || data->d_data==NULL) {
    printf("MultiplyAdd: missing data\n");
    return 0.0;
  }
  safeCall(cudaMemcpyToSymbol(d_ConstantA, &constA, sizeof(float)));
  safeCall(cudaMemcpyToSymbol(d_ConstantB, &constB, sizeof(float)));

  dim3 blocks(iDivUp(w, 16), iDivUp(h, 16));
  dim3 threads(16, 16);
  MultiplyAdd<<<blocks, threads>>>(res->d_data, data->d_data, w, h);
  checkMsg("MultiplyAdd() execution failed\n");
  safeCall(cudaThreadSynchronize());

  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("MultiplyAdd time =            %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}

double FindMinMax(CudaImage *img, float *minval, float *maxval)
{
  int w = img->width;
  int h = img->height;
  TimerGPU timer(0);

  int dx = iDivUp(w, 128);
  int dy = iDivUp(h, 16);
  int sz = 2*dx*dy*sizeof(float);
  float *d_minmax = 0; 
  float *h_minmax = 0; 
  h_minmax = (float *)malloc(sz);
  //safeCall(cudaMallocHost((void **)&h_minmax, sz)); 
  for (int i=0;i<2*dx*dy;i+=2) {
    h_minmax[i+0] = 1e6;
    h_minmax[i+1] = -1e6;
  }
  safeCall(cudaMalloc((void **)&d_minmax, sz));
  safeCall(cudaMemcpy(d_minmax, h_minmax, sz, cudaMemcpyHostToDevice));

  dim3 blocks(dx, dy);
  dim3 threads(128, 1);
  FindMinMax<<<blocks, threads>>>(d_minmax, img->d_data, w, h);
  checkMsg("FindMinMax() execution failed\n");
  safeCall(cudaThreadSynchronize());

  safeCall(cudaMemcpy(h_minmax, d_minmax, sz, cudaMemcpyDeviceToHost));
  *minval = 1e6;
  *maxval = -1e6;
  for (int i=0;i<2*dx*dy;i+=2) {
    if (h_minmax[i+0]<*minval) 
      *minval = h_minmax[i];
    if (h_minmax[i+1]>*maxval)  
      *maxval = h_minmax[i+1];
  }

  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("FindMinMax time =             %.2f msec\n", gpuTime);
#endif
  safeCall(cudaFree(d_minmax));
  free(h_minmax);
  //safeCall(cudaFreeHost(h_minmax));
  return gpuTime; 
}
 
double Find3DMinMax(CudaArray *minmax, CudaImage *data1, CudaImage *data2, 
  CudaImage *data3, float thresh, int maxPts)
{
  int *h_res = (int *)minmax->h_data;
  int *d_res = (int *)minmax->d_data;
  if (data1->d_data==NULL || data2->d_data==NULL || data3->d_data==NULL ||
      h_res==NULL || d_res==NULL) {
    std::cout << "Find3DMinMax: missing data " << std::endl;
    printf("Find3DMinMax: missing data %08x %08x %08x %08x %08x\n", 
	   data1->d_data, data2->d_data, data3->d_data, h_res, d_res);
    return 0.0;
  }
  int w = data1->width;
  int h = data1->height;
  TimerGPU timer(0);
  float threshs[2] = { thresh, -thresh };
  safeCall(cudaMemcpyToSymbol(d_ConstantA, &threshs, 2*sizeof(float)));

  dim3 blocks(iDivUp(w, MINMAX_SIZE), iDivUp(h,32));
  dim3 threads(WARP_SIZE + MINMAX_SIZE + 1);
  Find3DMinMax<<<blocks, threads>>>(d_res, data1->d_data, data2->d_data, 
				    data3->d_data, w, h); 
  checkMsg("Find3DMinMax() execution failed\n");
  safeCall(cudaThreadSynchronize());
  safeCall(cudaMemcpy(h_res, d_res, 
    sizeof(int)*minmax->pitch*minmax->height, cudaMemcpyDeviceToHost));
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("Find3DMinMax time =           %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}

double UnpackPointers(CudaArray *minmax, int maxPts, int *ptrs, int *numPts)
{
  unsigned int *minmax_data = (unsigned int *)minmax->h_data;
  if (minmax_data==NULL || ptrs==NULL) {
    printf("UnpackPointers: missing data %08x %08x\n", minmax_data, ptrs);
    return 0.0;
  }
  int w = minmax->pitch;
  int h = 32*minmax->height;
  TimerGPU timer(0);
  int num = 0;
  for (int y=0;y<h/32;y++) {
    for (int x=0;x<w;x++) {
      unsigned int val = minmax_data[y*w+x];
      if (val) {
	//printf("%d %d %08x\n", x, y, val);
	for (int k=0;k<32;k++) {
	  if (val&0x1 && num<maxPts)
	    ptrs[num++] = (y*32+k)*w + x;
	  val >>= 1;
	}
      }
    }
  }
  *numPts = num;
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("UnpackPointers time =         %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}

double ComputePositions(CudaImage *data1, CudaImage *data2, CudaImage *data3, 
  int *h_ptrs, CudaArray *sift, int numPts, int maxPts, float scale, 
  float factor)
{
  int w = data1->width;
  int h = data1->height;
  TimerGPU timer(0);
  int *d_ptrs = 0;
  float *d_sift = sift->d_data;
  safeCall(cudaMalloc((void **)&d_ptrs, sizeof(int)*numPts));
  safeCall(cudaMemcpy(d_ptrs, h_ptrs, sizeof(int)*numPts, 
    cudaMemcpyHostToDevice));
  safeCall(cudaMemcpyToSymbol(d_ConstantA, &scale, sizeof(float)));
  safeCall(cudaMemcpyToSymbol(d_ConstantB, &factor, sizeof(float)));

  dim3 blocks(iDivUp(numPts, POSBLK_SIZE));
  dim3 threads(POSBLK_SIZE);
  ComputePositions<<<blocks, threads>>>(data1->d_data, data2->d_data, 
    data3->d_data, d_ptrs, d_sift, numPts, maxPts, w, h);
  checkMsg("ComputePositions() execution failed\n");
  safeCall(cudaThreadSynchronize());

  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("ComputePositions time =       %.2f msec\n", gpuTime);
#endif
  safeCall(cudaFree(d_ptrs));
  return gpuTime;
}

double RemoveEdgePoints(CudaArray *sift, int *initNumPts, int maxPts, 
  float edgeLimit) 
{
  TimerGPU timer(0);
  int numPts = *initNumPts;
  float *d_sift = sift->d_data;
  int bw = sizeof(float)*numPts;
  float *h_sift = 0; 
  h_sift = (float *)malloc(5*bw);
  //safeCall(cudaMallocHost((void **)&h_sift, 5*bw));
  safeCall(cudaMemcpy2D(h_sift, bw, d_sift, sizeof(float)*maxPts,  
    bw, 5, cudaMemcpyDeviceToHost));
  int num = 0;
  for (int i=0;i<numPts;i++) 
    if (h_sift[4*numPts+i]<edgeLimit) {
      for (int j=0;j<5;j++) 
	h_sift[j*numPts+num] = h_sift[j*numPts+i];
      num ++;
    }
  safeCall(cudaMemcpy2D(d_sift, sizeof(float)*maxPts, h_sift, bw,  
    bw, 5, cudaMemcpyHostToDevice));
  free(h_sift);
  //safeCall(cudaFreeHost(h_sift)); 
  *initNumPts = num;
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("RemoveEdgePoints time =       %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}

double ComputeOrientations(CudaImage *img, int *h_ptrs, CudaArray *sift, 
  int numPts, int maxPts)
{
  int w = img->pitch;
  int h = img->height;
  TimerGPU timer(0);
  int *d_ptrs = 0;
  float *d_orient = &sift->d_data[5*maxPts];
  safeCall(cudaMalloc((void **)&d_ptrs, sizeof(int)*numPts));
  safeCall(cudaMemcpy(d_ptrs, h_ptrs, sizeof(int)*numPts, 
    cudaMemcpyHostToDevice));

  dim3 blocks(numPts);
  dim3 threads(32);
  ComputeOrientations<<<blocks, threads>>>(img->d_data, 
    d_ptrs, d_orient, maxPts, w, h);
  checkMsg("ComputeOrientations() execution failed\n");
  safeCall(cudaThreadSynchronize());

  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("ComputeOrientations time =    %.2f msec\n", gpuTime);
#endif
  safeCall(cudaFree(d_ptrs));
  return gpuTime;
}

double SecondOrientations(CudaArray *sift, int *initNumPts, int maxPts) 
{
  TimerGPU timer(0);
  int numPts = *initNumPts;
  int numPts2 = 2*numPts;
  float *d_sift = sift->d_data;
  int bw = sizeof(float)*numPts2;
  float *h_sift = 0; 
  h_sift = (float *)malloc(7*bw);
  //safeCall(cudaMallocHost((void **)&h_sift, 7*bw));
  safeCall(cudaMemcpy2D(h_sift, bw, d_sift, sizeof(float)*maxPts,  
    sizeof(float)*numPts, 7, cudaMemcpyDeviceToHost));
  int num = numPts;
  for (int i=0;i<numPts;i++) 
    if (h_sift[6*numPts2+i]>=0.0f && num<maxPts) {
      for (int j=0;j<5;j++) 
	h_sift[j*numPts2+num] = h_sift[j*numPts2+i];
      h_sift[5*numPts2+num] = h_sift[6*numPts2+i];
      h_sift[6*numPts2+num] = -1.0f;
      num ++;
    }
  safeCall(cudaMemcpy2D(&d_sift[numPts], sizeof(float)*maxPts, 
    &h_sift[numPts], bw, sizeof(float)*(num-numPts), 7, 
    cudaMemcpyHostToDevice));
  free(h_sift);
  //safeCall(cudaFreeHost(h_sift)); 
  *initNumPts = num;
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("SecondOrientations time =     %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}

double ExtractSiftDescriptors(CudaImage *img, CudaArray *sift, 
  CudaArray *desc, int numPts, int maxPts)
{
  TimerGPU timer(0);
  float *d_sift = sift->d_data, *d_desc = desc->d_data;

  tex.addressMode[0] = cudaAddressModeClamp;
  tex.addressMode[1] = cudaAddressModeClamp;
  tex.filterMode = cudaFilterModeLinear; 
  tex.normalized = false;
  safeCall(cudaBindTextureToArray(tex, (cudaArray *)img->t_data));
   
  dim3 blocks(numPts); 
  dim3 threads(16);
  ExtractSiftDescriptors<<<blocks, threads>>>(img->d_data, 
    d_sift, d_desc, maxPts);
  checkMsg("ExtractSiftDescriptors() execution failed\n");
  safeCall(cudaThreadSynchronize());
  safeCall(cudaUnbindTexture(tex));

  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("ExtractSiftDescriptors time = %.2f msec\n", gpuTime);
#endif
  return gpuTime; 
}
