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
// As ExtractSift, but each level of the scale space gets exactly its nominal blur, sigma
// 2^((i-1)/5) in octave pixels for level i = 0..7. The levels are blurred from the image
// that ExtractSift first filters with a Gaussian of sigma initBlur, and inputBlur is the blur
// already present in the input image (sigma, in pixels); the extra blur of each level is then
// computed from sqrt(initBlur^2 + inputBlur^2). ExtractSift doesn't take initBlur into
// account (and treats the variance of ScaleDown's Gaussian as a sigma), so its levels are
// blurrier than their nominal scales, most of all in the top octave: with initBlur 1.0, the
// top octave's levels have sigma 1.33 to 2.50 instead of 0.87 to 2.30, and the reported
// scales are too small by a factor that falls from 1.41 to 1.12 within the octave.
// - Keypoints change. Recommended: inputBlur 0.5 with initBlur 0.6. On the HPatches image
//   pairs, at the same number of features (about 5000 per image), about 3 points more of the
//   homographies are correct within 1 px than with ExtractSift and initBlur 1.0; within 3 and
//   5 px the difference is within noise. With the same thresh, more features are found (on
//   HPatches, 14% more at thresh 3.0): raise thresh to keep the same number.
// - sqrt(initBlur^2 + inputBlur^2) is limited to 0.80, by lowering initBlur (and inputBlur if
//   needed), with a message: the first level, of sigma 0.87, then still needs an extra blur
//   large enough for a sampled Gaussian. Beyond 0.83, the keypoints of the top octave's first
//   detection layer multiply.
// - With scaleUp, inputBlur counts double in the upsampled image, so inputBlur 0.4 or more
//   reaches the limit.
// - inputBlur < 0: the same as ExtractSift.
#define CUDASIFT_HAS_INPUT_BLUR 1
void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, bool scaleUp, float *tempMemory, float inputBlur);
void InitSiftData(SiftData &data, int num = 1024, bool host = false, bool dev = true);
void FreeSiftData(SiftData &data);
void PrintSiftData(SiftData &data);
double MatchSiftData(SiftData &data1, SiftData &data2);
double FindHomography(SiftData &data,  float *homography, int *numMatches, int numLoops = 1000, float minScore = 0.85f, float maxAmbiguity = 0.95f, float thresh = 5.0f);

#endif
