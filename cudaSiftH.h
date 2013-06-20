#ifndef CUDASIFTH_H
#define CUDASIFTH_H

#include "cudautils.h"
#include "cudaImage.h"

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

void ExtractSiftOctave(SiftData &siftData, CudaImage &img, double initBlur, float thresh, float lowestScale, float subsampling);
double ScaleDown(CudaImage &res, CudaImage &src, float variance);
double Subtract(CudaImage &res, CudaImage &srcA, CudaImage &srcB);
double FindPoints(CudaImage &data1, CudaImage &data2, CudaImage &data3, CudaImage &sift, float thresh, int maxPts, float edgeLimit, float scale, float factor);
double ComputeOrientations(CudaImage &img, CudaImage &sift, int numPts, int maxPts);
double SecondOrientations(CudaImage &sift, int *initNumPts, int maxPts);
double ExtractSiftDescriptors(CudaImage &img, CudaImage &sift, CudaImage &desc, int numPts, int maxPts);
double AddSiftData(SiftData &data, float *d_sift, float *d_desc, int numPts, int maxPts, float subsampling);

double LowPassMulti(CudaImage *results, CudaImage &origImg, CudaImage *tempImg, float baseBlur, float diffScale, float initBlur);
double SubtractMulti(CudaImage *results, CudaImage *sources);
double FindPointsMulti(CudaImage *sources, CudaImage &sift, float thresh, int maxPts, float edgeLimit, float scale, float factor, float lowestScale);

////////////////////////////////////////////////////////////////////
// Templated filter funtions
////////////////////////////////////////////////////////////////////
template<int RADIUS>
double SeparableFilter(CudaImage &dataA, CudaImage &dataB, CudaImage &temp, float *h_Kernel) 
{ 
  int width = dataA.width;
  int pitch = dataA.pitch;
  int height = dataA.height;
  float *d_DataA = dataA.d_data;
  float *d_DataB = dataB.d_data;
  float *d_Temp = temp.d_data;
  if (d_DataA==NULL || d_DataB==NULL || d_Temp==NULL) {
    printf("SeparableFilter: missing data\n");
    return 0.0;
  } 
  TimerGPU timer0(0);
  const unsigned int kernelSize = (2*RADIUS+1)*sizeof(float);
  safeCall(cudaMemcpyToSymbol(d_Kernel, h_Kernel, kernelSize));
        
  dim3 blockGridRows(iDivUp(width, CONVROW_W), height);
  dim3 threadBlockRows(CONVROW_W + 2*RADIUS); 
  ConvRowGPU<RADIUS><<<blockGridRows, threadBlockRows>>>(d_Temp, d_DataA, width, pitch, height);
  checkMsg("ConvRowGPU() execution failed\n");
  safeCall(cudaThreadSynchronize());
  dim3 blockGridColumns(iDivUp(width, CONVCOL_W), iDivUp(height, CONVCOL_H));
  dim3 threadBlockColumns(CONVCOL_W, CONVCOL_S);
  ConvColGPU<RADIUS><<<blockGridColumns, threadBlockColumns>>>(d_DataB, d_Temp, width, pitch, height); 
  checkMsg("ConvColGPU() execution failed\n");
  safeCall(cudaThreadSynchronize());

  double gpuTime = timer0.read();
#ifdef VERBOSE
  printf("SeparableFilter time =        %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

template<int RADIUS>
double LowPass(CudaImage &dataB, CudaImage &dataA, CudaImage &temp, double var)
{
  float kernel[2*RADIUS+1];
  float kernelSum = 0.0f;
  for (int j=-RADIUS;j<=RADIUS;j++) {
    kernel[j+RADIUS] = (float)expf(-(double)j*j/2.0/var);      
    kernelSum += kernel[j+RADIUS];
  }
  for (int j=-RADIUS;j<=RADIUS;j++) 
    kernel[j+RADIUS] /= kernelSum;  
  return SeparableFilter<RADIUS>(dataA, dataB, temp, kernel); 
}

#endif
