//********************************************************//
// CUDA SIFT extractor by Mårten Björkman aka Celebrandil //
//********************************************************//  

#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <vector>
#include "cudautils.h"

#include "cudaImage.h"
#include "cudaSift.h"
#include "cudaSiftD.h"
#include "cudaSiftH.h"
#include "cudaSiftBatch.h"

#include "cudaSiftD.cu"

// State of ExtractSift per device. The point counters and the constant memory (filter
// kernels, d_MaxNumPoints) are per device, so ExtractSift holds the lock of its device for
// the whole call. The caches of the constant uploads are per device too.
#define MAX_DEVICES 64
struct DeviceState {
  std::mutex lock;
  float scaleDownVariance = -1.0f;   // last upload to d_ScaleDownKernel
  float lowPassScale = -1.0f;        // last upload to d_LowPassKernel
  int maxPts = -1;                   // last upload to d_MaxNumPoints
  int numOctaves = -1;               // last upload to d_LaplaceKernel
};

static DeviceState &CurrentDeviceState()
{
  static DeviceState states[MAX_DEVICES];
  int device;
  safeCall(cudaGetDevice(&device));
  if (device>=MAX_DEVICES) {
    fprintf(stderr, "CudaSift: device %d not supported (at most %d devices)\n", device, MAX_DEVICES);
    exit(EXIT_FAILURE);
  }
  return states[device];
}

// Batches (cudaSiftBatch.h). A batch is the calls made between BeginSiftBatch and
// EndSiftBatch by one thread, on one device: their kernels are launched in order on the
// default stream, without waiting; the work they do on the host after their GPU work is
// deferred until the batch has completed (AfterBatch), in WaitSiftBatch or in the next call
// outside a batch. Calls from other threads are not part of it: they run as usual, after the
// work of the batch already launched (the default stream is in order), and complete a
// submitted batch first. One batch at a time.
struct BatchState {
  std::mutex lock;                      // protects counts (SiftData can be freed from any thread)
  std::mutex completion;                // protects submitted and pending: a batch is completed once
  bool open = false;                    // between BeginSiftBatch and EndSiftBatch
  std::thread::id owner;                // the thread of the open batch
  bool submitted = false;               // ended, not completed yet
  unsigned int id = 0;
  int device = -1;
  cudaEvent_t start = NULL, stop = NULL;   // GPU time of the batch, recorded on its device
  double time = 0.0;                    // GPU time of the last completed batch
  std::vector<std::function<void()>> pending;
  // SiftData extracted in the open batch (their count is on the GPU), and the point count on
  // the GPU of each SiftData, both keyed by the device points of the SiftData
  std::set<const void *> extracted;
  std::map<const void *, int *> counts;
};

static BatchState &Batch()
{
  static BatchState batch;
  return batch;
}

static void BatchFail(const char *msg)
{
  fprintf(stderr, "CudaSift: %s\n", msg);
  exit(EXIT_FAILURE);
}

bool InBatch()
{
  BatchState &b = Batch();
  if (b.open && b.owner==std::this_thread::get_id()) {
    int device;
    safeCall(cudaGetDevice(&device));
    if (device!=b.device)
      BatchFail("the calls of a batch must be made on the device of BeginSiftBatch");
  }
  return b.open && b.owner==std::this_thread::get_id();
}

unsigned int BatchId()
{
  return Batch().id;
}

void AfterBatch(std::function<void()> f)
{
  BatchState &b = Batch();
  std::lock_guard<std::mutex> lock(b.completion);
  b.pending.push_back(f);
}

void CompleteBatch()
{
  BatchState &b = Batch();
  std::lock_guard<std::mutex> lock(b.completion);
  if (!b.submitted)
    return;
  b.submitted = false;
  int device;
  safeCall(cudaGetDevice(&device));
  safeCall(cudaSetDevice(b.device));
  safeCall(cudaEventSynchronize(b.stop));
  float ms = 0.0f;
  safeCall(cudaEventElapsedTime(&ms, b.start, b.stop));
  b.time = ms;
  std::vector<std::function<void()>> pending;
  pending.swap(b.pending);
  for (auto &f : pending)
    f();
  safeCall(cudaSetDevice(device));
}

SiftPoint *DevicePoints(SiftData &data)
{
#ifdef MANAGEDMEM
  return data.m_data;
#else
  return data.d_data;
#endif
}

// The point count of a SiftData on the GPU, allocated on first use (freed by FreeSiftData)
static int *DeviceCount(SiftData &data)
{
  BatchState &b = Batch();
  std::lock_guard<std::mutex> lock(b.lock);
  int *&count = b.counts[DevicePoints(data)];
  if (count==NULL)
    safeCall(cudaMalloc((void **)&count, sizeof(int)));
  return count;
}

int *BatchCount(SiftData &data)
{
  int *count = DeviceCount(data);
  if (!Batch().extracted.count(DevicePoints(data)))
    SetCount<<<1, 1>>>(count, data.numPts);   // written when the batch reaches this call
  return count;
}

static void FreeDeviceCount(SiftData &data)
{
  BatchState &b = Batch();
  int *count = NULL;
  {
    std::lock_guard<std::mutex> lock(b.lock);
    auto it = b.counts.find(DevicePoints(data));
    if (it==b.counts.end())
      return;
    count = it->second;
    b.counts.erase(it);
  }
  safeCall(cudaFree(count));
}

void BeginSiftBatch()
{
  BatchState &b = Batch();
  if (b.open)
    BatchFail("BeginSiftBatch: a batch is already open (one batch at a time)");
  // The previous batch must complete first: batches share the counters and scratch memory
  CompleteBatch();
  int device;
  safeCall(cudaGetDevice(&device));
  if (device!=b.device) {
    if (b.start!=NULL) {
      safeCall(cudaSetDevice(b.device));
      safeCall(cudaEventDestroy(b.start));
      safeCall(cudaEventDestroy(b.stop));
      safeCall(cudaSetDevice(device));
    }
    safeCall(cudaEventCreate(&b.start));
    safeCall(cudaEventCreate(&b.stop));
    b.device = device;
  }
  safeCall(cudaEventRecord(b.start, 0));
  b.extracted.clear();
  b.id++;
  b.owner = std::this_thread::get_id();
  b.open = true;
}

void EndSiftBatch()
{
  BatchState &b = Batch();
  if (!InBatch())
    BatchFail("EndSiftBatch: no open batch (in this thread)");
  int device;
  safeCall(cudaGetDevice(&device));
  safeCall(cudaSetDevice(b.device));
  safeCall(cudaEventRecord(b.stop, 0));
  safeCall(cudaSetDevice(device));
  std::lock_guard<std::mutex> lock(b.completion);
  b.open = false;
  b.submitted = true;
}

double WaitSiftBatch()
{
  BatchState &b = Batch();
  if (InBatch())
    EndSiftBatch();
  CompleteBatch();   // (a call made after EndSiftBatch may have completed it already)
  double time = b.time;
  b.time = 0.0;
  return time;
}

void InitCuda(int devNum)
{
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  if (!nDevices) {
    std::cerr << "No CUDA devices available" << std::endl;
    return;
  }
  devNum = std::min(nDevices-1, devNum);
  deviceInit(devNum);  
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, devNum);
  printf("Device Number: %d\n", devNum);
  printf("  Device name: %s\n", prop.name);
  printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
  printf("  Total Global Memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
  printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
  printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
  printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
  printf("  Multiprocessor Count: %d\n\n", prop.multiProcessorCount);
}

static int CheckNumOctaves(int numOctaves)
{
  int num = std::max(1, std::min(numOctaves, MAX_OCTAVES));
  if (num!=numOctaves)
    printf("Warning: numOctaves = %d is not supported, using %d (1 to %d)\n", numOctaves, num, MAX_OCTAVES);
  return num;
}

// Each octave halves the image: with more octaves, the smallest ones would be empty (0
// pixels wide or high), and their kernel launches would fail
static int LimitOctaves(int width, int height, int numOctaves, bool scaleUp)
{
  int w = width*(scaleUp ? 2 : 1);
  int h = height*(scaleUp ? 2 : 1);
  int num = 1;
  while (num<numOctaves && (w>>num)>0 && (h>>num)>0)   // octave num+1 is (w>>num) x (h>>num)
    num++;
  if (num<numOctaves)
    printf("Warning: %d octaves for a %dx%d image%s, using %d\n", numOctaves, width, height, scaleUp ? " (upsampled)" : "", num);
  return num;
}

// Temporary memory of ExtractSift, in floats: sizeTmp floats at the start (the upsampled
// image with scaleUp), then the low-pass filtered image and the downsampled octaves.
// LaplaceFindPoints keeps the DoG layers in shared memory.
static int TempMemorySize(int width, int height, int numOctaves, bool scaleUp, int &sizeTmp)
{
  int w = width*(scaleUp ? 2 : 1);
  int h = height*(scaleUp ? 2 : 1);
  int p = iAlignUp(w, 128);
  int size = h*p;                 // image sizes
  sizeTmp = (scaleUp ? h*p : 0);  // upsampled image
  for (int i=0;i<numOctaves;i++) {
    w /= 2;
    h /= 2;
    int p = iAlignUp(w, 128);
    size += h*p;
  }
  return size + sizeTmp;
}

float *AllocSiftTempMemory(int width, int height, int numOctaves, bool scaleUp)
{
  TimerGPU timer(0);
  numOctaves = CheckNumOctaves(numOctaves);
  numOctaves = LimitOctaves(width, height, numOctaves, scaleUp);
  int sizeTmp;
  int size = TempMemorySize(width, height, numOctaves, scaleUp, sizeTmp);
  float *memoryTmp = NULL; 
  size_t pitch;
  safeCall(cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size+4095)/4096*sizeof(float)));
#ifdef VERBOSE
  printf("Allocated memory size: %d bytes\n", size);
  printf("Memory allocation time =      %.2f ms\n\n", timer.read());
#endif
  return memoryTmp;
}

void FreeSiftTempMemory(float *memoryTmp)
{
  if (memoryTmp)
    safeCall(cudaFree(memoryTmp));
}

// The number of points kept by ExtractSift. Outside batches, it is read into siftData.numPts,
// and NULL is returned. In a batch, it is written to the point count of siftData on the GPU,
// which is returned.
static int *CountExtractedPoints(SiftData &siftData, unsigned int *d_counter, bool batched)
{
  if (batched) {
    int *count = DeviceCount(siftData);
    Batch().extracted.insert(DevicePoints(siftData));
    CountPoints<<<1, 1>>>(d_counter, siftData.maxPts, count);
    return count;
  }
  safeCall(cudaMemcpy(&siftData.numPts, d_counter, sizeof(int), cudaMemcpyDeviceToHost)); 
  siftData.numPts = (siftData.numPts<siftData.maxPts ? siftData.numPts : siftData.maxPts);
  return NULL;
}

void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, bool scaleUp, float *tempMemory) 
{
  const bool batched = InBatch();
  if (!batched)
    CompleteBatch();
  DeviceState &state = CurrentDeviceState();
  std::lock_guard<std::mutex> lock(state.lock);
  TimerGPU timer(0);
  numOctaves = CheckNumOctaves(numOctaves);
  numOctaves = LimitOctaves(img.width, img.height, numOctaves, scaleUp);
  unsigned int *d_PointCounterAddr;
  safeCall(cudaGetSymbolAddress((void**)&d_PointCounterAddr, d_PointCounter));
  safeCall(cudaMemset(d_PointCounterAddr, 0, ((MAX_OCTAVES+1)*2+1)*sizeof(int)));
  // Constant memory uploads are synchronous; skip them when nothing changed
  if (siftData.maxPts!=state.maxPts)
    safeCall(cudaMemcpyToSymbol(d_MaxNumPoints, &siftData.maxPts, sizeof(int)));
  bool newKernels = (numOctaves!=state.numOctaves);
  state.maxPts = siftData.maxPts;
  state.numOctaves = numOctaves;

  int width = img.width*(scaleUp ? 2 : 1);
  int height = img.height*(scaleUp ? 2 : 1);
  int sizeTmp;
  int size = TempMemorySize(img.width, img.height, numOctaves, scaleUp, sizeTmp);
  float *memoryTmp = tempMemory; 
  if (!tempMemory) {
    size_t pitch;
    safeCall(cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size+4095)/4096*sizeof(float)));
#ifdef VERBOSE
    printf("Allocated memory size: %d bytes\n", size);
    printf("Memory allocation time =      %.2f ms\n\n", timer.read());
#endif
  }
  float *memorySub = memoryTmp + sizeTmp;

  CudaImage lowImg;
  lowImg.Allocate(width, height, iAlignUp(width, 128), false, memorySub);
  int *d_count = NULL;   // the point count on the GPU, in a batch
  if (!scaleUp) {
    if (newKernels) {
      float kernel[(MAX_OCTAVES+1)*12*16];
      PrepareLaplaceKernels(numOctaves, 0.0f, kernel);
      safeCall(cudaMemcpyToSymbolAsync(d_LaplaceKernel, kernel, (MAX_OCTAVES+1)*12*16*sizeof(float)));
    }
    LowPass(lowImg, img, max(initBlur, 0.001f));
    TimerGPU timer1(0);
    ExtractSiftLoop(siftData, lowImg, numOctaves, 0.0f, thresh, lowestScale, 1.0f, memoryTmp, memorySub + height*iAlignUp(width, 128));
    d_count = CountExtractedPoints(siftData, &d_PointCounterAddr[2*numOctaves], batched);
    if (!batched)
      printf("SIFT extraction time =        %.2f ms %d\n", timer1.read(), siftData.numPts);
  } else {
    CudaImage upImg;
    upImg.Allocate(width, height, iAlignUp(width, 128), false, memoryTmp);
    TimerGPU timer1(0); 
    ScaleUp(upImg, img);
    LowPass(lowImg, upImg, max(initBlur, 0.001f));
    if (newKernels) {
      float kernel[(MAX_OCTAVES+1)*12*16];
      PrepareLaplaceKernels(numOctaves, 0.0f, kernel);
      safeCall(cudaMemcpyToSymbolAsync(d_LaplaceKernel, kernel, (MAX_OCTAVES+1)*12*16*sizeof(float)));
    }
    ExtractSiftLoop(siftData, lowImg, numOctaves, 0.0f, thresh, lowestScale*2.0f, 1.0f, memoryTmp, memorySub + height*iAlignUp(width, 128));
    d_count = CountExtractedPoints(siftData, &d_PointCounterAddr[2*numOctaves], batched);
    RescalePositions(siftData, 0.5f, d_count);
    if (!batched)
      printf("SIFT extraction time =        %.2f ms\n", timer1.read());
  } 
  if (batched) {   // in a batch, the rest is done once the batch has completed
    SiftData *data = &siftData;
    float *memoryFree = (tempMemory ? NULL : memoryTmp);
    AfterBatch([=]() {
      safeCall(cudaMemcpy(&data->numPts, d_count, sizeof(int), cudaMemcpyDeviceToHost));
#ifndef MANAGEDMEM
      if (data->h_data && data->numPts>0)
        safeCall(cudaMemcpy(data->h_data, data->d_data, sizeof(SiftPoint)*data->numPts, cudaMemcpyDeviceToHost));
#endif
      if (memoryFree)
        safeCall(cudaFree(memoryFree));
    });
    return;
  }
  
  if (!tempMemory)
    safeCall(cudaFree(memoryTmp));
#ifdef MANAGEDMEM
  safeCall(cudaDeviceSynchronize());
#else
  if (siftData.h_data)
    safeCall(cudaMemcpy(siftData.h_data, siftData.d_data, sizeof(SiftPoint)*siftData.numPts, cudaMemcpyDeviceToHost));
#endif
  double totTime = timer.read();
  printf("Incl prefiltering & memcpy =  %.2f ms %d\n\n", totTime, siftData.numPts);
}

int ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp, float *memorySub) 
{
#ifdef VERBOSE
  TimerGPU timer(0);
#endif
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
  ExtractSiftOctave(siftData, img, numOctaves, thresh, lowestScale, subsampling);
#ifdef VERBOSE
  double totTime = timer.read();
  printf("ExtractSift time total =      %.2f ms %d\n\n", totTime, numOctaves);
#endif
  return 0;
}

void ExtractSiftOctave(SiftData &siftData, CudaImage &img, int octave, float thresh, float lowestScale, float subsampling)
{
#ifdef VERBOSE
  unsigned int *d_PointCounterAddr;
  safeCall(cudaGetSymbolAddress((void**)&d_PointCounterAddr, d_PointCounter));
  unsigned int fstPts, totPts;
  safeCall(cudaMemcpy(&fstPts, &d_PointCounterAddr[2*octave-1], sizeof(int), cudaMemcpyDeviceToHost)); 
  TimerGPU timer0;
#endif
  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = img.d_data;
  resDesc.res.pitch2D.width = img.width;
  resDesc.res.pitch2D.height = img.height;
  resDesc.res.pitch2D.pitchInBytes = img.pitch*sizeof(float);  
  resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeClamp;
  texDesc.addressMode[1]   = cudaAddressModeClamp;
  texDesc.filterMode       = cudaFilterModeLinear;
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;
  // Create texture object
  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

#ifdef VERBOSE
  TimerGPU timer1;
#endif
  float baseBlur = pow(2.0f, -1.0f/NUM_SCALES);
  float diffScale = pow(2.0f, 1.0f/NUM_SCALES);
  LaplaceFindPoints(img, siftData, thresh, 10.0f, 1.0f/NUM_SCALES, lowestScale/subsampling, subsampling, octave);
#ifdef VERBOSE
  double gpuTimeDoG = timer1.read();
  TimerGPU timer4;
#endif
  ComputeOrientations(texObj, img, siftData, octave); 
  ExtractSiftDescriptors(texObj, siftData, subsampling, octave); 
  //OrientAndExtract(texObj, siftData, subsampling, octave); 
  
  safeCall(cudaDestroyTextureObject(texObj));
#ifdef VERBOSE
  double gpuTimeSift = timer4.read();
  double totTime = timer0.read();
  printf("GPU time : %.2f ms + %.2f ms + %.2f ms = %.2f ms\n", totTime-gpuTimeDoG-gpuTimeSift, gpuTimeDoG, gpuTimeSift, totTime);
  safeCall(cudaMemcpy(&totPts, &d_PointCounterAddr[2*octave+1], sizeof(int), cudaMemcpyDeviceToHost));
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
#ifdef MANAGEDMEM
  safeCall(cudaMallocManaged((void **)&data.m_data, sz));
#else
  data.h_data = NULL;
  if (host)
    safeCall(cudaMallocHost((void **)&data.h_data, sz)); // pinned: much faster device-to-host copies
  data.d_data = NULL;
  if (dev)
    safeCall(cudaMalloc((void **)&data.d_data, sz));
#endif
}

void FreeSiftData(SiftData &data)
{
  FreeDeviceCount(data);
#ifdef MANAGEDMEM
  safeCall(cudaFree(data.m_data));
  data.m_data = NULL;
#else
  if (data.d_data!=NULL)
    safeCall(cudaFree(data.d_data));
  data.d_data = NULL;
  if (data.h_data!=NULL) {
    // Pinned if allocated by InitSiftData or PrintSiftData, else allocated by the caller with malloc
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, data.h_data)==cudaSuccess && attr.type==cudaMemoryTypeHost)
      safeCall(cudaFreeHost(data.h_data));
    else {
      cudaGetLastError();   // clear the error of cudaPointerGetAttributes, if any
      free(data.h_data);
    }
  }
  data.h_data = NULL;
#endif
  data.numPts = 0;
  data.maxPts = 0;
}

void PrintSiftData(SiftData &data)
{
#ifdef MANAGEDMEM
  SiftPoint *h_data = data.m_data;
#else
  SiftPoint *h_data = data.h_data;
  if (data.h_data==NULL) {
    safeCall(cudaMallocHost((void **)&h_data, sizeof(SiftPoint)*data.maxPts));   // freed by FreeSiftData
    safeCall(cudaMemcpy(h_data, data.d_data, sizeof(SiftPoint)*data.numPts, cudaMemcpyDeviceToHost));
    data.h_data = h_data;
  }
#endif
  for (int i=0;i<data.numPts;i++) {
    printf("xpos         = %.2f\n", h_data[i].xpos);
    printf("ypos         = %.2f\n", h_data[i].ypos);
    printf("scale        = %.2f\n", h_data[i].scale);
    printf("sharpness    = %.2f\n", h_data[i].sharpness);
    printf("edgeness     = %.2f\n", h_data[i].edgeness);
    printf("orientation  = %.2f\n", h_data[i].orientation);
    printf("score        = %.2f\n", h_data[i].score);
    float *siftData = (float*)&h_data[i].data;
    for (int j=0;j<8;j++) {
      if (j==0) 
	printf("data = ");
      else 
	printf("       ");
      for (int k=0;k<16;k++)
	if (siftData[j+8*k]<0.05)
	  printf(" .   ");
	else
	  printf("%.2f ", siftData[j+8*k]);
      printf("\n");
    }
  }
  printf("Number of available points: %d\n", data.numPts);
  printf("Number of allocated points: %d\n", data.maxPts);
}

// SortSiftData's order: position, then scale, orientation and descriptor
static bool SiftPointLess(const SiftPoint &a, const SiftPoint &b)
{
  if (a.xpos!=b.xpos) return a.xpos<b.xpos;
  if (a.ypos!=b.ypos) return a.ypos<b.ypos;
  if (a.scale!=b.scale) return a.scale<b.scale;
  if (a.orientation!=b.orientation) return a.orientation<b.orientation;
  return memcmp(a.data, b.data, sizeof(a.data))<0;
}

void SortSiftData(SiftData &data)
{
  if (InBatch())
    BatchFail("SortSiftData cannot be called in a batch (BeginSiftBatch)");
  CompleteBatch();
  if (data.numPts<=1)
    return;
#ifdef MANAGEDMEM
  safeCall(cudaDeviceSynchronize());
  std::sort(data.m_data, data.m_data + data.numPts, SiftPointLess);
#else
  // Sorted on the host, from the device copy, which the other functions read
  std::vector<SiftPoint> copy;
  SiftPoint *h_data = data.h_data;
  if (h_data==NULL) {
    copy.resize(data.numPts);
    h_data = copy.data();
  }
  if (data.d_data)
    safeCall(cudaMemcpy(h_data, data.d_data, sizeof(SiftPoint)*data.numPts, cudaMemcpyDeviceToHost));
  std::sort(h_data, h_data + data.numPts, SiftPointLess);
  if (data.d_data)
    safeCall(cudaMemcpy(data.d_data, h_data, sizeof(SiftPoint)*data.numPts, cudaMemcpyHostToDevice));
#endif
}

///////////////////////////////////////////////////////////////////////////////
// Host side master functions
///////////////////////////////////////////////////////////////////////////////

double ScaleDown(CudaImage &res, CudaImage &src, float variance)
{
  DeviceState &state = CurrentDeviceState();   // under the lock of ExtractSift
  if (res.d_data==NULL || src.d_data==NULL) {
    printf("ScaleDown: missing data\n");
    return 0.0;
  }
  if (variance!=state.scaleDownVariance) {
    float h_Kernel[5];
    float kernelSum = 0.0f;
    for (int j=0;j<5;j++) {
      h_Kernel[j] = (float)expf(-(double)(j-2)*(j-2)/2.0/variance);      
      kernelSum += h_Kernel[j];
    }
    for (int j=0;j<5;j++)
      h_Kernel[j] /= kernelSum;  
    safeCall(cudaMemcpyToSymbol(d_ScaleDownKernel, h_Kernel, 5*sizeof(float)));
    state.scaleDownVariance = variance;
  }
#if 0
  dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
  dim3 threads(SCALEDOWN_W + 4, SCALEDOWN_H + 4);
  ScaleDownDenseShift<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
#elif 0
  dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
  dim3 threads(SCALEDOWN_W + 4);
  ScaleDown<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
#else
  dim3 blocks(iDivUp(src.width/2, 32), iDivUp(src.height/2, 8));
  dim3 threads(32, 8);
  ScaleDownDirect<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
#endif
  checkMsg("ScaleDown() execution failed\n");
  return 0.0;
}

double ScaleUp(CudaImage &res, CudaImage &src)
{
  if (res.d_data==NULL || src.d_data==NULL) {
    printf("ScaleUp: missing data\n");
    return 0.0;
  }
  dim3 blocks(iDivUp(res.width, SCALEUP_W), iDivUp(res.height, SCALEUP_H));
  dim3 threads(SCALEUP_W/2, SCALEUP_H/2);
  ScaleUp<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch); 
  checkMsg("ScaleUp() execution failed\n");
  return 0.0;
}   

double ComputeOrientations(cudaTextureObject_t texObj, CudaImage &src, SiftData &siftData, int octave)
{
  dim3 blocks(512); 
#ifdef MANAGEDMEM
  dim3 threads(11*11);
  ComputeOrientationsCONST<<<blocks, threads>>>(texObj, siftData.m_data, octave);
#else
#if 1
  dim3 threads(11*11);
  ComputeOrientationsCONST<<<blocks, threads>>>(texObj, siftData.d_data, octave);
#else
  dim3 threads(256); 
  ComputeOrientationsCONSTNew<<<blocks, threads>>>(src.d_data, src.width, src.pitch, src.height, siftData.d_data, octave);
#endif
#endif
  checkMsg("ComputeOrientations() execution failed\n");
  return 0.0;
}

double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData, float subsampling, int octave)
{
  dim3 blocks(512); 
  dim3 threads(16, 8);
#ifdef MANAGEDMEM
  ExtractSiftDescriptorsGather<<<blocks, threads>>>(texObj, siftData.m_data, subsampling, octave);
#else
  ExtractSiftDescriptorsGather<<<blocks, threads>>>(texObj, siftData.d_data, subsampling, octave);
#endif
  checkMsg("ExtractSiftDescriptors() execution failed\n");
  return 0.0; 
}

double OrientAndExtract(cudaTextureObject_t texObj, SiftData &siftData, float subsampling, int octave)
{
  dim3 blocks(256); 
  dim3 threads(128);
#ifdef MANAGEDMEM
  OrientAndExtractCONST<<<blocks, threads>>>(texObj, siftData.m_data, subsampling, octave);
#else
  OrientAndExtractCONST<<<blocks, threads>>>(texObj, siftData.d_data, subsampling, octave);
#endif
  checkMsg("OrientAndExtract() execution failed\n");
  return 0.0;
}

double RescalePositions(SiftData &siftData, float scale, const int *d_numPts)
{
  // In a batch, the grid covers maxPts, and the kernel stops at the count on the GPU
  const int numPts = (d_numPts ? siftData.maxPts : siftData.numPts);
  if (numPts<=0)   // a launch with 0 blocks is an error
    return 0.0;
  dim3 blocks(iDivUp(numPts, 64));
  dim3 threads(64);
#ifdef MANAGEDMEM
  RescalePositions<<<blocks, threads>>>(siftData.m_data, siftData.numPts, scale, d_numPts);
#else
  RescalePositions<<<blocks, threads>>>(siftData.d_data, siftData.numPts, scale, d_numPts);
#endif
  checkMsg("RescapePositions() execution failed\n");
  return 0.0; 
}

double LowPass(CudaImage &res, CudaImage &src, float scale)
{
  float kernel[2*LOWPASS_R+1];
  DeviceState &state = CurrentDeviceState();   // under the lock of ExtractSift
  if (scale!=state.lowPassScale) {
    float kernelSum = 0.0f;
    float ivar2 = 1.0f/(2.0f*scale*scale);
    for (int j=-LOWPASS_R;j<=LOWPASS_R;j++) {
      kernel[j+LOWPASS_R] = (float)expf(-(double)j*j*ivar2);
      kernelSum += kernel[j+LOWPASS_R]; 
    }
    for (int j=-LOWPASS_R;j<=LOWPASS_R;j++) 
      kernel[j+LOWPASS_R] /= kernelSum;  
    safeCall(cudaMemcpyToSymbol(d_LowPassKernel, kernel, (2*LOWPASS_R+1)*sizeof(float)));
    state.lowPassScale = scale;
  }  
  int width = res.width;
  int height = res.height;
  dim3 blocks(iDivUp(width, LOWPASS_W), iDivUp(height, LOWPASS_H));
#if 1
  dim3 threads(LOWPASS_W+2*LOWPASS_R, 4); 
  LowPassBlock<<<blocks, threads>>>(src.d_data, res.d_data, width, src.pitch, height, res.pitch);
#else
  dim3 threads(LOWPASS_W+2*LOWPASS_R, LOWPASS_H);
  LowPass<<<blocks, threads>>>(src.d_data, res.d_data, width, res.pitch, height);
#endif
  checkMsg("LowPass() execution failed\n");
  return 0.0; 
}

//==================== Multi-scale functions ===================//

void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel)
{
  if (numOctaves>1) {
    float totInitBlur = (float)sqrt(initBlur*initBlur + 0.5f*0.5f) / 2.0f;
    PrepareLaplaceKernels(numOctaves-1, totInitBlur, kernel);
  }
  float scale = pow(2.0f, -1.0f/NUM_SCALES);
  float diffScale = pow(2.0f, 1.0f/NUM_SCALES);
  for (int i=0;i<NUM_SCALES+3;i++) {
    float kernelSum = 0.0f;
    float var = scale*scale - initBlur*initBlur;
    for (int j=0;j<=LAPLACE_R;j++) {
      kernel[numOctaves*12*16 + 16*i + j] = (float)expf(-(double)j*j/2.0/var);
      kernelSum += (j==0 ? 1 : 2)*kernel[numOctaves*12*16 + 16*i + j]; 
    }
    for (int j=0;j<=LAPLACE_R;j++)
      kernel[numOctaves*12*16 + 16*i + j] /= kernelSum;
    scale *= diffScale;
  }
}
 
double LaplaceMulti(cudaTextureObject_t texObj, CudaImage &baseImage, CudaImage *results, int octave) 
{
  int width = results[0].width;
  int pitch = results[0].pitch;
  int height = results[0].height;
#if 1
  dim3 threads(LAPLACE_W+2*LAPLACE_R);
  dim3 blocks(iDivUp(width, LAPLACE_W), height);
  LaplaceMultiMem<<<blocks, threads>>>(baseImage.d_data, results[0].d_data, width, pitch, height, octave);
#endif
#if 0
  dim3 threads(LAPLACE_W+2*LAPLACE_R, LAPLACE_S);
  dim3 blocks(iDivUp(width, LAPLACE_W), iDivUp(height, LAPLACE_H));
  LaplaceMultiMemTest<<<blocks, threads>>>(baseImage.d_data, results[0].d_data, width, pitch, height, octave);
#endif
#if 0
  dim3 threads(LAPLACE_W+2*LAPLACE_R, LAPLACE_S);
  dim3 blocks(iDivUp(width, LAPLACE_W), height);
  LaplaceMultiMemOld<<<blocks, threads>>>(baseImage.d_data, results[0].d_data, width, pitch, height, octave);
#endif
#if 0
  dim3 threads(LAPLACE_W+2*LAPLACE_R, LAPLACE_S);
  dim3 blocks(iDivUp(width, LAPLACE_W), height);
  LaplaceMultiTex<<<blocks, threads>>>(texObj, results[0].d_data, width, pitch, height, octave);
#endif
  checkMsg("LaplaceMulti() execution failed\n");
  return 0.0; 
}

double LaplaceFindPoints(CudaImage &img, SiftData &siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave)
{
  if (img.d_data==NULL) {
    printf("LaplaceFindPoints: missing data\n");
    return 0.0;
  }
  dim3 blocks(iDivUp(img.width, MINMAX_W), iDivUp(img.height, MINMAX_H*FUSED_FY));
  dim3 threads(FUSED_NT);
#ifdef MANAGEDMEM
  LaplaceFindPoints<<<blocks, threads>>>(img.d_data, siftData.m_data, img.width, img.pitch, img.height, subsampling, lowestScale, thresh, factor, edgeLimit, octave);
#else
  LaplaceFindPoints<<<blocks, threads>>>(img.d_data, siftData.d_data, img.width, img.pitch, img.height, subsampling, lowestScale, thresh, factor, edgeLimit, octave);
#endif
  checkMsg("LaplaceFindPoints() execution failed\n");
  return 0.0;
}

double FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave)
{
  if (sources->d_data==NULL) {
    printf("FindPointsMulti: missing data\n");
    return 0.0;
  }
  int w = sources->width;
  int p = sources->pitch;
  int h = sources->height;
#if 0
  dim3 blocks(iDivUp(w, MINMAX_W)*NUM_SCALES, iDivUp(h, MINMAX_H));
  dim3 threads(MINMAX_W + 2, MINMAX_H);
  FindPointsMultiTest<<<blocks, threads>>>(sources->d_data, siftData.d_data, w, p, h, subsampling, lowestScale, thresh, factor, edgeLimit, octave); 
#endif
#if 1
  dim3 blocks(iDivUp(w, MINMAX_W)*NUM_SCALES, iDivUp(h, MINMAX_H));
  dim3 threads(MINMAX_W + 2); 
#ifdef MANAGEDMEM
  FindPointsMulti<<<blocks, threads>>>(sources->d_data, siftData.m_data, w, p, h, subsampling, lowestScale, thresh, factor, edgeLimit, octave); 
#else
  FindPointsMultiNew<<<blocks, threads>>>(sources->d_data, siftData.d_data, w, p, h, subsampling, lowestScale, thresh, factor, edgeLimit, octave);
#endif
#endif
  checkMsg("FindPointsMulti() execution failed\n");
  return 0.0;
}

