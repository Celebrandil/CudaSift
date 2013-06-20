#ifndef CUDASIFT_H
#define CUDASIFT_H

#include "cudaImage.h"

typedef struct {
  float xpos;
  float ypos;   
  float scale;
  float sharpness;
  float edgeness;
  float orientation;
  float score;
  float ambiguity;
  int match;
  float match_xpos;
  float match_ypos;
  float match_error;
  float empty[4];
  float data[128];
} SiftPoint;

typedef struct {
  int numPts;         // Number of available Sift points
  int maxPts;         // Number of allocated Sift points
  SiftPoint *h_data;  // Host (CPU) data
  SiftPoint *d_data;  // Device (GPU) data
} SiftData;

void InitCuda();

void ExtractSift(SiftData *siftData, CudaImage *img, int numLayers, 
  int numOctaves, double initBlur, float thresh, float subsampling = 1.0f);
void ExtractSiftOctave(SiftData *siftData, CudaImage *img, int numLayers, 
  double initBlur, float thresh = 0.02f, float subsampling = 1.0f);

void InitSiftData(SiftData *data, int num = 1024, bool host = false, 
  bool dev = true);
void FreeSiftData(SiftData *data);
void PrintSiftData(SiftData *data);
double MatchSiftData(SiftData *data1, SiftData *data2);
double FindHomography(SiftData *data,  float *homography, int *numMatches, 
  int numLoops = 1000, float minScore = 0.85f, float maxAmbiguity = 0.95f, 
  float thresh = 5.0f);

double LowPass5(CudaImage *res, CudaImage *data, float variance);
double ScaleDown(CudaImage *res, CudaImage *data, float variance);
double Subtract(CudaImage *res, CudaImage *dataA, CudaImage *dataB);
double MultiplyAdd(CudaImage *res, CudaImage *data, float constA, 
  float constB);
double FindMinMax(CudaImage *img, float *minval, float *maxval);

#endif
