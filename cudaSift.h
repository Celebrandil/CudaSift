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
  float subsampling;
  float empty[3];
  float data[128];
} SiftPoint;

typedef struct {
  int numPts;         // Number of available Sift points
  int maxPts;         // Number of allocated Sift points
#ifdef MANAGEDMEM
  SiftPoint *m_data;  // Managed data
#else
  SiftPoint *h_data;  // Host (CPU) data
  SiftPoint *d_data;  // Device (GPU) data
#endif
} SiftData;

void InitCuda(int devNum = 0);
float *AllocSiftTempMemory(int width, int height, int numOctaves, bool scaleUp = false);
void FreeSiftTempMemory(float *memoryTmp);
void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale = 0.0f, bool scaleUp = false, float *tempMemory = 0);
void InitSiftData(SiftData &data, int num = 1024, bool host = false, bool dev = true);
void FreeSiftData(SiftData &data);
void PrintSiftData(SiftData &data);
// Sorts the points by position (x, then y), then scale, orientation and descriptor, on the
// CPU. ExtractSift writes the points in an order that varies between runs, and the
// homography functions draw their samples from the matches in the order of the points: for
// reproducible results (tests, benchmarks), sort both point sets after ExtractSift and before
// MatchSiftData, whose matches are indices into the other set. Optional; not in a batch.
void SortSiftData(SiftData &data);
#define CUDASIFT_HAS_SORT_SIFT_DATA 1
double MatchSiftData(SiftData &data1, SiftData &data2);
// RANSAC homography from the matches with score > minScore and ambiguity < maxAmbiguity;
// numMatches is the number of inliers among them
double FindHomography(SiftData &data,  float *homography, int *numMatches, int numLoops = 1000, float minScore = 0.85f, float maxAmbiguity = 0.95f, float thresh = 5.0f);

#define CUDASIFT_HAS_GPU_HOMOGRAPHY 1   // FindHomographySeeded, FindHomographyGPU, ImproveHomographyGPU

// As FindHomography, with a seeded random number generator instead of rand(), so that the
// samples are reproducible and the same on all backends
double FindHomographySeeded(SiftData &data, float *homography, int *numMatches, int numLoops = 1000, float minScore = 0.85f, float maxAmbiguity = 0.95f, float thresh = 5.0f, unsigned int seed = 42);
// As FindHomographySeeded, but entirely on the GPU: the same samples, and the same result.
// Both depend on the order of the points (see SortSiftData).
double FindHomographyGPU(SiftData &data, float *homography, int *numMatches, int numLoops = 1000, float minScore = 0.85f, float maxAmbiguity = 0.95f, float thresh = 5.0f, unsigned int seed = 42);
// As ImproveHomography (geomFuncs.cpp), on the GPU; numFit: number of inliers. With fewer
// than 4 inliers, or if the equations are degenerate, the homography is kept
double ImproveHomographyGPU(SiftData &data, float *homography, int *numFit, int numLoops = 5, float minScore = 0.0f, float maxAmbiguity = 0.80f, float thresh = 3.0f);

#endif
