#ifndef CUDASIFTH_H
#define CUDASIFTH_H

#include <functional>
#include "cudautils.h"
#include "cudaImage.h"
#include "cudaSift.h"

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

int ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp, float *memorySub);
void ExtractSiftOctave(SiftData &siftData, CudaImage &img, int octave, float thresh, float lowestScale, float subsampling);
double ScaleDown(CudaImage &res, CudaImage &src, float variance);
double ScaleUp(CudaImage &res, CudaImage &src);
double ComputeOrientations(cudaTextureObject_t texObj, CudaImage &src, SiftData &siftData, int octave);
double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData, float subsampling, int octave);
double OrientAndExtract(cudaTextureObject_t texObj, SiftData &siftData, float subsampling, int octave);
double RescalePositions(SiftData &siftData, float scale, const int *d_numPts = NULL);
double LowPass(CudaImage &res, CudaImage &src, float scale);
void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel);
double LaplaceMulti(cudaTextureObject_t texObj, CudaImage &baseImage, CudaImage *results, int octave);
double LaplaceFindPoints(CudaImage &img, SiftData &siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave);
double FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave);

// Batches (cudaSiftBatch.h), for cudaSiftH.cu and matching.cu. In a batch, the calls launch
// their kernels without waiting for them. The point counts stay on the GPU (BatchCount), and
// the work that the calls do on the host after their GPU work (setting numPts, copying the
// results to the host, returning the homographies) is deferred until the batch has
// completed (AfterBatch).
bool InBatch();                        // between BeginSiftBatch and EndSiftBatch
unsigned int BatchId();                // changes at each BeginSiftBatch
void CompleteBatch();                  // waits for a submitted batch and completes its calls
void AfterBatch(std::function<void()> f);
int *BatchCount(SiftData &data);       // point count of data on the GPU, in a batch
SiftPoint *DevicePoints(SiftData &data);

#endif
