#ifndef CUDASIFTH_OLD_H
#define CUDASIFTH_OLD_H

#include "cudaImage.h"

double MultiplyAdd(CudaImage *res, CudaImage *data, float constA, float constB);
double FindMinMax(CudaImage *img, float *minval, float *maxval);
double ComputeOrientations(CudaImage *img, int *h_ptrs, CudaImage *sift, int numPts, int maxPts);
double Find3DMinMax(CudaImage *minmax, CudaImage *data1, CudaImage *data2, CudaImage *data3, float thresh, int maxPts);
double UnpackPointers(CudaImage *minmax, int maxPts, int *ptrs, int &numPts);
double ComputePositions(CudaImage *data1, CudaImage *data2, CudaImage *data3, int *h_ptrs, CudaImage *sift, int numPts, int maxPts, float scale, float factor);
double RemoveEdgePoints(CudaImage *sift, int &initNumPts, int maxPts, float edgeLimit);

#endif // CUDASIFTH_OLD_H
