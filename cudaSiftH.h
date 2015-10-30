#ifndef CUDASIFTH_H
#define CUDASIFTH_H

#include "cudautils.h"
#include "cudaImage.h"

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

void ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp, float *memorySub);
void ExtractSiftOctave(SiftData &siftData, CudaImage &img, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp);
double ScaleDown(CudaImage &res, CudaImage &src, float variance);
double ComputeOrientations(cudaTextureObject_t texObj, SiftData &siftData, int fstPts, int totPts);
double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData, int fstPts, int totPts, float subsampling);
double LaplaceMulti(cudaTextureObject_t texObj, CudaImage *results, float baseBlur, float diffScale, float initBlur);
double FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh, float edgeLimit, float scale, float factor, float lowestScale, float subsampling);

#endif
