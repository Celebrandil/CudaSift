//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//              Apple Metal backend                       //
//********************************************************//

// Metal implementation of the public API in cudaSift.h and cudaImage.h, following
// the CUDA host code (cudaSiftH.cu, cudaImage.cu, matching.cu) step by step, and of the
// Metal-specific API in cudaSiftBatch.h and cudaSiftMetal.h.
//
// All memory is allocated as shared (unified) Metal buffers, so the "device"
// pointers are also valid on the CPU: SiftData::h_data and SiftData::d_data point to
// the same memory when both are requested, and no copies are needed. Device
// pointers are mapped back to their MTLBuffer when kernels are encoded. All
// functions are synchronous, except in batches (cudaSiftBatch.h): they return after
// the GPU work has completed. ExtractSift, MatchSiftData and the GPU homography calls
// can be made from several threads, and run one at a time; the allocation and release of
// images and SiftData are thread-safe too. Batches must be built from one thread. Each call drains its own autorelease
// pool, so that command buffers and temporary buffers are released without a run loop.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>

#include "cudaImage.h"
#include "cudaSift.h"
#include "cudaSiftBatch.h"
#include "cudaSiftMetal.h"
#include "cudaSiftD.h"
#include "cudaSiftMetalShared.h"
#include "cudaSiftRandom.h"
#include "cuda_runtime.h"
#include "cudaSiftMetalSource.h"

///////////////////////////////////////////////////////////////////////////////
// Metal context, buffers and command encoding
///////////////////////////////////////////////////////////////////////////////

namespace {

struct Context {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  id<MTLLibrary> preciseLibrary = nil;          // same source, without fast math (double-float kernels)
  std::map<std::string, id<MTLComputePipelineState>> pipelines;
  std::mutex mutex;                            // buffers and numPts (allocations may come from any thread)
  std::mutex callMutex;                        // held by ExtractSift, MatchSiftData and the homography calls
  std::map<uintptr_t, id<MTLBuffer>> buffers;  // keyed by contents address
  id<MTLBuffer> pointCounter = nil;            // d_PointCounter[NUM_COUNTERS]
  std::map<int, id<MTLBuffer>> laplaceKernels; // d_LaplaceKernel[(MAX_OCTAVES+1)*12*16], per number of octaves
  std::map<const void *, id<MTLBuffer>> numPts; // point counts on the GPU, per SiftData
  std::set<const void *> extracted;            // SiftData extracted in the open batch
  std::map<const float *, id<MTLBuffer>> homographies;  // homographies computed in the open batch
  class Commands *batch = nullptr;             // open batch (BeginSiftBatch)
  id<MTLCommandBuffer> submitted = nil;        // submitted batch (EndSiftBatch)
  double batchTime = 0.0;                      // GPU time of the last completed batch, for WaitSiftBatch
  std::vector<std::function<void()>> pending;  // to do after the batch has completed
  id<MTLBuffer> best = nil;                    // FindMaxCorr12 per-tile results
  id<MTLBuffer> cand = nil;                    // ExtractSift keypoint candidates, maxPts per octave
};

void Fail(const char *msg, NSError *error = nil)
{
  fprintf(stderr, "CudaSift Metal error: %s%s%s\n", msg, error ? ": " : "",
          error ? [[error localizedDescription] UTF8String] : "");
  exit(-1);
}

Context *CreateContext()
{
  Context *c = new Context();   // never destroyed: may be used until the end of the program
  Context &ctx = *c;
  @autoreleasepool {
    ctx.device = MTLCreateSystemDefaultDevice();
    if (ctx.device==nil)
      Fail("no Metal device available");
    if (![ctx.device supportsFamily:MTLGPUFamilyApple7])   // SIMD group matrices (M1 and later)
      Fail("an Apple GPU of family Apple7 or later (Apple M1 or later) is required");
    if (![ctx.device supports32BitFloatFiltering])
      Fail("device does not support linear filtering of 32-bit float textures");
    ctx.queue = [ctx.device newCommandQueue];
    NSError *error = nil;
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    if (@available(macOS 15.0, *))
      options.mathMode = MTLMathModeFast;
    else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      options.fastMathEnabled = YES;   // earlier macOS versions (e.g. on older Macs)
#pragma clang diagnostic pop
    }
    ctx.library = [ctx.device newLibraryWithSource:[NSString stringWithUTF8String:cudaSiftMetalSource]
                                           options:options error:&error];
    if (ctx.library==nil)
      Fail("failed to compile Metal kernels", error);
    ctx.pointCounter = [ctx.device newBufferWithLength:NUM_COUNTERS*sizeof(unsigned int) options:MTLResourceStorageModeShared];
    if (ctx.pointCounter==nil)
      Fail("out of GPU memory");
  }
  return c;
}

Context &GetContext()
{
  static Context *ctx = CreateContext();   // thread-safe initialization
  return *ctx;
}

id<MTLBuffer> NewBuffer(size_t size, MTLResourceOptions options)
{
  id<MTLBuffer> buffer = [GetContext().device newBufferWithLength:std::max(size, (size_t)16) options:options];
  if (buffer==nil) {
    char msg[64];
    snprintf(msg, sizeof(msg), "out of GPU memory (%zu bytes)", size);
    Fail(msg);
  }
  return buffer;
}

void *MetalAlloc(size_t size)
{
  Context &ctx = GetContext();
  id<MTLBuffer> buffer = NewBuffer(size, MTLResourceStorageModeShared);
  void *ptr = [buffer contents];
  std::lock_guard<std::mutex> lock(ctx.mutex);
  ctx.buffers[(uintptr_t)ptr] = buffer;
  return ptr;
}

void MetalFree(void *ptr)
{
  if (ptr!=NULL) {
    Context &ctx = GetContext();
    std::lock_guard<std::mutex> lock(ctx.mutex);
    ctx.buffers.erase((uintptr_t)ptr);
  }
}

// Buffer containing ptr (possibly in the middle of an allocation), and offset within it;
// nil if ptr wasn't allocated by CudaSift
id<MTLBuffer> LookupBuffer(const void *ptr, NSUInteger &offset)
{
  Context &ctx = GetContext();
  std::lock_guard<std::mutex> lock(ctx.mutex);
  uintptr_t addr = (uintptr_t)ptr;
  auto it = ctx.buffers.upper_bound(addr);
  if (it!=ctx.buffers.begin()) {
    --it;
    if (addr<it->first + [it->second length]) {
      offset = addr - it->first;
      return it->second;
    }
  }
  return nil;
}

// The same, stopping the program if ptr wasn't allocated by CudaSift
id<MTLBuffer> FindBuffer(const void *ptr, NSUInteger &offset)
{
  id<MTLBuffer> buffer = LookupBuffer(ptr, offset);   // the lock is released before Fail(): exit() doesn't unwind
  if (buffer==nil)
    Fail("pointer not allocated by the CudaSift Metal backend");
  return buffer;
}

double Milliseconds(std::chrono::steady_clock::time_point start)
{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

// The kernels compiled without fast math, for the double-float arithmetic of
// ImproveHomographyGPU (fast math would break its error-free transformations)
id<MTLLibrary> PreciseLibrary()
{
  Context &ctx = GetContext();
  if (ctx.preciseLibrary==nil) {
    @autoreleasepool {
      NSError *error = nil;
      MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
      if (@available(macOS 15.0, *))
        options.mathMode = MTLMathModeSafe;
      else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
#pragma clang diagnostic pop
      }
      ctx.preciseLibrary = [ctx.device newLibraryWithSource:[NSString stringWithUTF8String:cudaSiftMetalSource]
                                                    options:options error:&error];
      if (ctx.preciseLibrary==nil)
        Fail("failed to compile Metal kernels", error);
    }
  }
  return ctx.preciseLibrary;
}

// One command buffer with a compute encoder. By default, dispatches run in order. With
// concurrent=true, they may overlap until the next Barrier(). A batch's encoder can be ended
// so that other code encodes into the command buffer (CudaSiftMetalBatchCommandBuffer):
// Resume() then opens a new one, which runs after that work.
class Commands {
public:
  id<MTLCommandBuffer> cb;
  id<MTLComputeCommandEncoder> enc;
  bool concurrent;
  Commands(bool concurrent_ = false) : concurrent(concurrent_) {
    cb = [GetContext().queue commandBuffer];
    Resume();
  }
  void Suspend() {
    if (enc!=nil) {
      [enc endEncoding];
      enc = nil;
    }
  }
  void Resume() {
    if (enc==nil)
      enc = (concurrent ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] : [cb computeCommandEncoder]);
  }
  void Barrier() {
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers | MTLBarrierScopeTextures];
  }
  void Barrier(const void *ptr) {   // only for the buffer containing ptr
    NSUInteger offset = 0;
    id<MTLResource> r = FindBuffer(ptr, offset);
    [enc memoryBarrierWithResources:&r count:1];
  }
  void Pipeline(const char *name, bool precise = false) {
    Pipeline(name, precise ? PreciseLibrary() : GetContext().library);
  }
  void Pipeline(const char *name, id<MTLLibrary> library) {   // cached by name: a kernel is used with one library
    Context &ctx = GetContext();
    id<MTLComputePipelineState> pso = ctx.pipelines[name];
    if (pso==nil) {
      NSError *error = nil;
      id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
      if (function==nil)
        Fail(name);
      pso = [ctx.device newComputePipelineStateWithFunction:function error:&error];
      if (pso==nil)
        Fail(name, error);
      if ([pso threadExecutionWidth]!=32)   // the kernels assume 32-wide SIMD groups
        Fail("the GPU's SIMD groups are not 32 threads wide");
      ctx.pipelines[name] = pso;
    }
    [enc setComputePipelineState:pso];
  }
  void Buffer(const void *ptr, int index) {
    NSUInteger offset = 0;
    id<MTLBuffer> buffer = FindBuffer(ptr, offset);
    [enc setBuffer:buffer offset:offset atIndex:index];
  }
  void Buffer(id<MTLBuffer> buffer, int index, NSUInteger offset = 0) {
    [enc setBuffer:buffer offset:offset atIndex:index];
  }
  template <class T> void Bytes(const T &value, int index) {
    [enc setBytes:&value length:sizeof(T) atIndex:index];
  }
  void Bytes(const void *data, size_t size, int index) {
    [enc setBytes:data length:size atIndex:index];
  }
  void Texture(id<MTLTexture> texture, int index) {
    [enc setTexture:texture atIndex:index];
  }
  void Dispatch(MTLSize groups, MTLSize threads) {
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    if (!concurrent)
      Barrier();
  }
  void DispatchIndirect(id<MTLBuffer> args, NSUInteger offset, MTLSize threads) {
    [enc dispatchThreadgroupsWithIndirectBuffer:args indirectBufferOffset:offset threadsPerThreadgroup:threads];
    if (!concurrent)
      Barrier();
  }
  void DispatchThreads(MTLSize grid, MTLSize threads) {
    [enc dispatchThreads:grid threadsPerThreadgroup:threads];
    if (!concurrent)
      Barrier();
  }
  // Submits, waits and returns the GPU execution time in ms
  double Finish() {
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if ([cb status]==MTLCommandBufferStatusError)
      Fail("command buffer failed", [cb error]);
    return 1000.0*([cb GPUEndTime] - [cb GPUStartTime]);
  }
};

// Texture view of a pitched image, for linear sampling (tex2D in the CUDA code)
id<MTLTexture> MakeTexture(CudaImage &img)
{
  Context &ctx = GetContext();
  NSUInteger offset = 0;
  id<MTLBuffer> buffer = FindBuffer(img.d_data, offset);
  NSUInteger align = [ctx.device minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatR32Float];
  NSUInteger rowBytes = sizeof(float)*img.pitch;
  if (offset%align || rowBytes%align)
    Fail("image not aligned for a Metal texture");
  MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                  width:img.width height:img.height mipmapped:NO];
  desc.storageMode = MTLStorageModeShared;
  desc.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> texture = [buffer newTextureWithDescriptor:desc offset:offset bytesPerRow:rowBytes];
  if (texture==nil)
    Fail("failed to create texture");
  return texture;
}

// Point count of a SiftData on the GPU (an int), kept equal to SiftData::numPts
// outside batches
id<MTLBuffer> NumPtsBuffer(SiftData &data)
{
  Context &ctx = GetContext();
  {
    std::lock_guard<std::mutex> lock(ctx.mutex);
    id<MTLBuffer> b = ctx.numPts[data.d_data];
    if (b!=nil)
      return b;
  }
  id<MTLBuffer> b = NewBuffer(sizeof(int), MTLResourceStorageModeShared);   // may Fail(): without the lock
  std::lock_guard<std::mutex> lock(ctx.mutex);
  ctx.numPts[data.d_data] = b;
  return b;
}

///////////////////////////////////////////////////////////////////////////////
// Kernel launches, as the host side master functions in cudaSiftH.cu
///////////////////////////////////////////////////////////////////////////////

void ScaleDown(Commands &cmd, CudaImage &res, CudaImage &src, float variance)
{
  float h_Kernel[5];
  float kernelSum = 0.0f;
  for (int j=0;j<5;j++) {
    h_Kernel[j] = (float)expf(-(double)(j-2)*(j-2)/2.0/variance);
    kernelSum += h_Kernel[j];
  }
  for (int j=0;j<5;j++)
    h_Kernel[j] /= kernelSum;
  ImageParams p = {src.width, src.pitch, src.height, res.pitch};
  cmd.Pipeline("ScaleDownDirect");
  cmd.Buffer(res.d_data, 0);
  cmd.Buffer(src.d_data, 1);
  cmd.Bytes(p, 2);
  cmd.Bytes(h_Kernel, sizeof(h_Kernel), 3);
  cmd.DispatchThreads(MTLSizeMake(src.width/2, src.height/2, 1), MTLSizeMake(32, 8, 1));
}

void ScaleUp(Commands &cmd, CudaImage &res, CudaImage &src)
{
  ImageParams p = {src.width, src.pitch, src.height, res.pitch};
  cmd.Pipeline("ScaleUp");
  cmd.Buffer(res.d_data, 0);
  cmd.Buffer(src.d_data, 1);
  cmd.Bytes(p, 2);
  cmd.Dispatch(MTLSizeMake(iDivUp(res.width, SCALEUP_W), iDivUp(res.height, SCALEUP_H), 1),
               MTLSizeMake(SCALEUP_W/2, SCALEUP_H/2, 1));
}

void LowPass(Commands &cmd, CudaImage &res, CudaImage &src, float scale)
{
  float kernel[2*LOWPASS_R+1];
  float kernelSum = 0.0f;
  float ivar2 = 1.0f/(2.0f*scale*scale);
  for (int j=-LOWPASS_R;j<=LOWPASS_R;j++) {
    kernel[j+LOWPASS_R] = (float)expf(-(double)j*j*ivar2);
    kernelSum += kernel[j+LOWPASS_R];
  }
  for (int j=-LOWPASS_R;j<=LOWPASS_R;j++)
    kernel[j+LOWPASS_R] /= kernelSum;
  ImageParams p = {res.width, src.pitch, res.height, res.pitch};
  cmd.Pipeline("LowPassBlock");
  cmd.Buffer(src.d_data, 0);
  cmd.Buffer(res.d_data, 1);
  cmd.Bytes(p, 2);
  cmd.Bytes(kernel, sizeof(kernel), 3);
  cmd.Dispatch(MTLSizeMake(iDivUp(res.width, LOWPASS_W), iDivUp(res.height, LOWPASS_H), 1),
               MTLSizeMake(LOWPASS_W+2*LOWPASS_R, 4, 1));
}

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

// Candidates of an octave: maxPts CandPoint (8 floats) per octave in ctx.cand
NSUInteger CandOffset(SiftData &siftData, int octave)
{
  return (NSUInteger)(octave - 1)*siftData.maxPts*8*sizeof(float);
}

void LaplaceFindPoints(Commands &cmd, CudaImage &img, SiftData &siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave, id<MTLBuffer> laplaceKernel)
{
  FindPointsParams p = {img.width, img.pitch, img.height, octave, siftData.maxPts, subsampling, lowestScale, thresh, factor, edgeLimit};
  Context &ctx = GetContext();
  cmd.Pipeline("LaplaceFindPoints");
  cmd.Buffer(img.d_data, 0);
  cmd.Buffer(ctx.cand, 1, CandOffset(siftData, octave));
  cmd.Bytes(p, 2);
  cmd.Buffer(laplaceKernel, 3);
  cmd.Buffer(ctx.pointCounter, 4);
  cmd.Dispatch(MTLSizeMake(iDivUp(img.width, MINMAX_W), iDivUp(img.height, MINMAX_H*FUSED_FY), 1), MTLSizeMake(FUSED_NT, 1, 1));
}

void ComputeOrientations(Commands &cmd, id<MTLTexture> texture, SiftData &siftData, int octave)
{
  OctaveParams p = {octave, siftData.maxPts, 1.0f};
  Context &ctx = GetContext();
  cmd.Pipeline("ComputeOrientationsSimd");
  cmd.Texture(texture, 0);
  cmd.Buffer(ctx.cand, 0, CandOffset(siftData, octave));
  cmd.Bytes(p, 1);
  cmd.Buffer(ctx.pointCounter, 2);
  cmd.Dispatch(MTLSizeMake(ORIENT_GROUPS, 1, 1), MTLSizeMake(32*ORIENT_SG, 1, 1));
}

void ExtractSiftDescriptors(Commands &cmd, id<MTLTexture> texture, SiftData &siftData, float subsampling, int octave)
{
  OctaveParams p = {octave, siftData.maxPts, subsampling};
  Context &ctx = GetContext();
  cmd.Pipeline("ExtractSiftDescriptorsGather");
  cmd.Texture(texture, 0);
  cmd.Buffer(siftData.d_data, 0);
  cmd.Bytes(p, 1);
  cmd.Buffer(ctx.pointCounter, 2);
  cmd.Buffer(ctx.cand, 3, CandOffset(siftData, octave));
  cmd.Dispatch(MTLSizeMake(DESC_GROUPS, 1, 1), MTLSizeMake(16*8, 1, 1));
}

void RescalePositions(Commands &cmd, SiftData &siftData, float scale)
{
  cmd.Pipeline("RescalePositions");
  cmd.Buffer(siftData.d_data, 0);
  cmd.Bytes(scale, 1);
  cmd.Buffer(NumPtsBuffer(siftData), 2);   // count computed on the GPU
  cmd.DispatchThreads(MTLSizeMake(siftData.maxPts, 1, 1), MTLSizeMake(64, 1, 1));
}

// Extraction from the low-pass filtered image, octave by octave as ExtractSiftLoop and
// ExtractSiftOctave in cudaSiftH.cu, but in phases, so that the octaves are processed
// concurrently: downsampling, then DoG and extrema detection, then orientations, then
// descriptors. Each octave has its own point counters (2*octave: candidates,
// 2*octave+1: with the extra orientations) and candidate region, and the descriptors
// write the points of octave o after those of octaves 1..o-1, i.e. in the same order
// as the CUDA code (smallest octave first, extra orientations after each octave).
void ExtractSiftOctaves(Commands &cmd, SiftData &siftData, CudaImage &lowImg, int numOctaves, float thresh, float lowestScale, float *memorySub, id<MTLBuffer> laplaceKernel)
{
  std::vector<CudaImage> imgs(numOctaves + 1);   // imgs[numOctaves]: full size ... imgs[1]: smallest
  std::vector<float> subsampling(numOctaves + 1);
  imgs[numOctaves].Allocate(lowImg.width, lowImg.height, lowImg.pitch, false, lowImg.d_data);
  subsampling[numOctaves] = 1.0f;
  std::vector<id<MTLTexture>> textures(numOctaves + 1);
  // The full size octave only needs the low-pass filtered image: start it first, and
  // downsample in parallel, with barriers only on the image memory
  textures[numOctaves] = MakeTexture(imgs[numOctaves]);
  LaplaceFindPoints(cmd, imgs[numOctaves], siftData, thresh, 10.0f, 1.0f/NUM_SCALES, lowestScale, 1.0f, numOctaves, laplaceKernel);
  for (int o=numOctaves-1;o>=1;o--) {
    int w = imgs[o+1].width, h = imgs[o+1].height;
    int p = iAlignUp(w/2, 128);
    imgs[o].Allocate(w/2, h/2, p, false, memorySub);
    memorySub += (h/2)*p;
    subsampling[o] = 2.0f*subsampling[o+1];
    ScaleDown(cmd, imgs[o], imgs[o+1], 0.5f);
    cmd.Barrier(imgs[o].d_data);
  }
  for (int o=numOctaves-1;o>=1;o--) {
    textures[o] = MakeTexture(imgs[o]);
    LaplaceFindPoints(cmd, imgs[o], siftData, thresh, 10.0f, 1.0f/NUM_SCALES, lowestScale/subsampling[o], subsampling[o], o, laplaceKernel);
  }
  cmd.Barrier();
  for (int o=numOctaves;o>=1;o--)
    ComputeOrientations(cmd, textures[o], siftData, o);
  cmd.Barrier();
  for (int o=numOctaves;o>=1;o--)
    ExtractSiftDescriptors(cmd, textures[o], siftData, subsampling[o], o);
}

} // namespace

///////////////////////////////////////////////////////////////////////////////
// Public API (cudaSift.h)
///////////////////////////////////////////////////////////////////////////////

void InitCuda(int)   // a single Metal device
{
  @autoreleasepool {
    Context &ctx = GetContext();
    printf("Device Number: %d\n", 0);
    printf("  Device name: %s (Metal)\n", [[ctx.device name] UTF8String]);
    printf("  Recommended Max Working Set: %llu MB\n", (unsigned long long)[ctx.device recommendedMaxWorkingSetSize]/(1024*1024));
    printf("  Max Threads per Threadgroup: %lu\n", (unsigned long)[ctx.device maxThreadsPerThreadgroup].width);
    printf("  Max Threadgroup Memory: %lu KB\n\n", (unsigned long)[ctx.device maxThreadgroupMemoryLength]/1024);
  }
}

// Temporary memory of ExtractSift, in floats: the upsampled image (scaleUp only, upSize),
// then the low-pass filtered image and the downsampled octaves. As with CUDA,
// LaplaceFindPoints keeps the DoG layers in threadgroup memory.
static size_t TempMemorySize(int width, int height, int numOctaves, bool scaleUp, size_t &upSize)
{
  int w = width*(scaleUp ? 2 : 1);
  int h = height*(scaleUp ? 2 : 1);
  upSize = (scaleUp ? (size_t)h*iAlignUp(w, 128) : 0);
  size_t size = upSize + (size_t)h*iAlignUp(w, 128);
  for (int i=0;i<numOctaves;i++) {
    w /= 2;
    h /= 2;
    size += (size_t)h*iAlignUp(w, 128);
  }
  return size;
}

// At most MAX_OCTAVES octaves, the size of the per-octave counters and Laplace kernels.
// Each octave halves the image: with more octaves, the smallest ones would be empty (0
// pixels wide or high), and so would their textures. Same limits as in the CUDA code.
static int LimitOctaves(int width, int height, int numOctaves, bool scaleUp)
{
  if (numOctaves<1 || numOctaves>MAX_OCTAVES) {
    int n = std::max(1, std::min(numOctaves, MAX_OCTAVES));
    printf("Warning: numOctaves = %d is not supported, using %d (1 to %d)\n", numOctaves, n, MAX_OCTAVES);
    numOctaves = n;
  }
  int w = width*(scaleUp ? 2 : 1);
  int h = height*(scaleUp ? 2 : 1);
  int num = 1;
  while (num<numOctaves && (w>>num)>0 && (h>>num)>0)   // octave num+1 is (w>>num) x (h>>num)
    num++;
  if (num<numOctaves)
    printf("Warning: %d octaves for a %dx%d image%s, using %d\n", numOctaves, width, height, scaleUp ? " (upsampled)" : "", num);
  return num;
}

float *AllocSiftTempMemory(int width, int height, int numOctaves, bool scaleUp)
{
  numOctaves = LimitOctaves(width, height, numOctaves, scaleUp);
  size_t upSize;
  size_t size = TempMemorySize(width, height, numOctaves, scaleUp, upSize);
  return (float *)MetalAlloc((size+4095)/4096*4096*sizeof(float));
}

void FreeSiftTempMemory(float *memoryTmp)
{
  MetalFree(memoryTmp);
}

// Waits for the submitted batch (EndSiftBatch), if any, and completes its calls. Calls
// outside a batch do this first: they share the point counters and scratch buffers, and
// may use the results of the batch.
static void CompleteBatch()
{
  Context &ctx = GetContext();
  if (ctx.submitted==nil)
    return;
  id<MTLCommandBuffer> cb = ctx.submitted;
  ctx.submitted = nil;
  [cb waitUntilCompleted];
  if ([cb status]==MTLCommandBufferStatusError)
    Fail("command buffer failed", [cb error]);
  std::vector<std::function<void()>> pending;
  pending.swap(ctx.pending);
  for (auto &f : pending)
    f();
  ctx.batchTime = 1000.0*([cb GPUEndTime] - [cb GPUStartTime]);
}

// Commands of a call: those of the open batch if any (after a barrier, so that calls
// run in order), or new ones (in local) that FinishCall() submits and waits for.
static Commands *CallCommands(std::unique_ptr<Commands> &local, bool concurrent)
{
  Context &ctx = GetContext();
  if (ctx.batch) {
    ctx.batch->Resume();   // after the caller's work, if any (CudaSiftMetalBatchCommandBuffer)
    ctx.batch->Barrier();
    return ctx.batch;
  }
  local.reset(new Commands(concurrent));
  return local.get();
}

// Runs f after the call has completed: in FinishCall() outside a batch, else in WaitSiftBatch()
static void AfterCall(std::function<void()> f)
{
  GetContext().pending.push_back(f);
}

// Outside a batch: submits, waits, and runs the AfterCall() functions
static double FinishCall(Commands &cmd)
{
  Context &ctx = GetContext();
  double gpuTime = cmd.Finish();
  std::vector<std::function<void()>> pending;
  pending.swap(ctx.pending);
  for (auto &f : pending)
    f();
  return gpuTime;
}

void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, bool scaleUp, float *tempMemory)
{
  @autoreleasepool {
    std::lock_guard<std::mutex> callLock(GetContext().callMutex);
    auto start = std::chrono::steady_clock::now();
    Context &ctx = GetContext();
    const bool batched = (ctx.batch!=nullptr);
    if (!batched)
      CompleteBatch();
    numOctaves = LimitOctaves(img.width, img.height, numOctaves, scaleUp);
    id<MTLBuffer> laplaceKernel = ctx.laplaceKernels[numOctaves];
    if (laplaceKernel==nil) {
      laplaceKernel = NewBuffer((MAX_OCTAVES+1)*12*16*sizeof(float), MTLResourceStorageModeShared);
      PrepareLaplaceKernels(numOctaves, 0.0f, (float *)[laplaceKernel contents]);
      ctx.laplaceKernels[numOctaves] = laplaceKernel;
    }

    const int width = img.width*(scaleUp ? 2 : 1);
    const int height = img.height*(scaleUp ? 2 : 1);
    size_t upSize;
    TempMemorySize(img.width, img.height, numOctaves, scaleUp, upSize);
    float *memoryTmp = (tempMemory ? tempMemory : AllocSiftTempMemory(img.width, img.height, numOctaves, scaleUp));
    float *memorySub = memoryTmp + upSize;
    size_t candSize = (size_t)numOctaves*siftData.maxPts*8*sizeof(float);
    if (ctx.cand==nil || [ctx.cand length]<candSize)
      ctx.cand = NewBuffer(candSize, MTLResourceStorageModePrivate);   // previous one retained by its command buffers

    CudaImage lowImg;
    lowImg.Allocate(width, height, iAlignUp(width, 128), false, memorySub);
    std::unique_ptr<Commands> local;
    Commands &cmd = *CallCommands(local, true);
    if (batched) {   // the counters may still be in use by previous calls of the batch
      cmd.Pipeline("ClearCounters");
      cmd.Buffer(ctx.pointCounter, 0);
      cmd.DispatchThreads(MTLSizeMake(NUM_COUNTERS, 1, 1), MTLSizeMake(32, 1, 1));
      cmd.Barrier();
    } else
      memset([ctx.pointCounter contents], 0, NUM_COUNTERS*sizeof(unsigned int));
    if (!scaleUp) {
      LowPass(cmd, lowImg, img, std::max(initBlur, 0.001));
      cmd.Barrier();
      ExtractSiftOctaves(cmd, siftData, lowImg, numOctaves, thresh, lowestScale, memorySub + height*iAlignUp(width, 128), laplaceKernel);
    } else {
      CudaImage upImg;
      upImg.Allocate(width, height, iAlignUp(width, 128), false, memoryTmp);
      ScaleUp(cmd, upImg, img);
      cmd.Barrier();
      LowPass(cmd, lowImg, upImg, std::max(initBlur, 0.001));
      cmd.Barrier();
      ExtractSiftOctaves(cmd, siftData, lowImg, numOctaves, thresh, lowestScale*2.0f, memorySub + height*iAlignUp(width, 128), laplaceKernel);
    }
    // Number of points, on the GPU (for the next calls of a batch)
    cmd.Barrier();
    int octMax[2] = {numOctaves, siftData.maxPts};
    id<MTLBuffer> numPtsBuffer = NumPtsBuffer(siftData);
    cmd.Pipeline("CountPoints");
    cmd.Buffer(ctx.pointCounter, 0);
    cmd.Bytes(octMax, sizeof(octMax), 1);
    cmd.Buffer(numPtsBuffer, 2);
    cmd.DispatchThreads(MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
    if (scaleUp) {
      cmd.Barrier();
      RescalePositions(cmd, siftData, 0.5f);
    }
    if (batched)
      ctx.extracted.insert(siftData.d_data);
    SiftData *data = &siftData;
    AfterCall([=]() {
      data->numPts = *(int *)[numPtsBuffer contents];
      if (!tempMemory)
        FreeSiftTempMemory(memoryTmp);
      if (data->h_data!=NULL && data->h_data!=data->d_data)
        memcpy(data->h_data, data->d_data, sizeof(SiftPoint)*data->numPts);
    });
    if (!batched) {
      double gpuTime = FinishCall(cmd);
      printf("SIFT extraction time =        %.2f ms %d\n", gpuTime, siftData.numPts);
      printf("Incl prefiltering & memcpy =  %.2f ms %d\n\n", Milliseconds(start), siftData.numPts);
    }
  }
}

void InitSiftData(SiftData &data, int num, bool host, bool dev)
{
  data.numPts = 0;
  data.maxPts = num;
  size_t sz = sizeof(SiftPoint)*num;
  data.d_data = (dev ? (SiftPoint *)MetalAlloc(sz) : NULL);
  data.h_data = NULL;
  if (host)
    data.h_data = (dev ? data.d_data : (SiftPoint *)malloc(sz));  // unified memory: no separate host copy
}

void FreeSiftData(SiftData &data)
{
  if (data.h_data!=NULL && data.h_data!=data.d_data)
    free(data.h_data);
  {
    Context &ctx = GetContext();
    std::lock_guard<std::mutex> lock(ctx.mutex);
    ctx.numPts.erase(data.d_data);
  }
  MetalFree(data.d_data);
  data.d_data = NULL;
  data.h_data = NULL;
  data.numPts = 0;
  data.maxPts = 0;
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
  std::lock_guard<std::mutex> callLock(GetContext().callMutex);
  if (GetContext().batch)
    Fail("SortSiftData cannot be called in a batch (BeginSiftBatch)");
  CompleteBatch();
  SiftPoint *points = (data.d_data!=NULL ? data.d_data : data.h_data);   // shared memory: sorted in place
  if (points!=NULL && data.numPts>1)
    std::sort(points, points + data.numPts, SiftPointLess);
}

void PrintSiftData(SiftData &data)
{
  SiftPoint *h_data = (data.h_data!=NULL ? data.h_data : data.d_data);
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

// In a batch, the point count of data on the GPU: as computed by a previous call of the
// batch, or as set on the host (no batch is in flight)
static id<MTLBuffer> BatchNumPts(SiftData &data)
{
  id<MTLBuffer> n = NumPtsBuffer(data);
  if (!GetContext().extracted.count(data.d_data))
    *(int *)[n contents] = data.numPts;
  return n;
}

// Binds the point count of data: on the GPU in a batch, else the host value
static void NumPtsArg(Commands &cmd, SiftData &data, int index)
{
  if (GetContext().batch)
    cmd.Buffer(BatchNumPts(data), index);
  else
    cmd.Bytes(data.numPts, index);
}

double MatchSiftData(SiftData &data1, SiftData &data2)
{
  @autoreleasepool {
    std::lock_guard<std::mutex> callLock(GetContext().callMutex);
    auto start = std::chrono::steady_clock::now();
    Context &ctx = GetContext();
    const bool batched = (ctx.batch!=nullptr);
    if (!batched)
      CompleteBatch();
    int numPts1 = data1.numPts;
    int numPts2 = data2.numPts;
    if (!batched && (!numPts1 || !numPts2))
      return 0.0;
    if (data1.d_data==NULL || data2.d_data==NULL)
      return 0.0;
    // In a batch, the point counts may be computed by previous calls of the batch, so
    // they are read on the GPU, and the grid size is computed on the GPU too
    int maxPts1 = (batched ? data1.maxPts : numPts1);
    size_t size = 4*sizeof(float)*M12_CHUNKS*maxPts1;
    if (ctx.best==nil || [ctx.best length]<size)
      ctx.best = NewBuffer(size, MTLResourceStorageModePrivate);
    std::unique_ptr<Commands> local;
    Commands &cmd = *CallCommands(local, false);
    int numChunks = (batched ? M12_CHUNKS : std::min(iDivUp(numPts2, M12_T), M12_CHUNKS));
    MatchParams p = {numChunks};
    id<MTLBuffer> grid = nil;
    if (batched) {
      grid = NewBuffer(6*sizeof(uint32_t), MTLResourceStorageModePrivate);
      cmd.Pipeline("MatchGrid");
      cmd.Buffer(BatchNumPts(data1), 0);
      cmd.Buffer(grid, 1);
      cmd.Bytes(numChunks, 2);
      cmd.Buffer(BatchNumPts(data2), 3);
      cmd.DispatchThreads(MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
      cmd.Barrier();
    }
    cmd.Pipeline("FindMaxCorr12");
    cmd.Buffer(data1.d_data, 0);
    cmd.Buffer(data2.d_data, 1);
    cmd.Bytes(p, 2);
    cmd.Buffer(ctx.best, 3);
    NumPtsArg(cmd, data1, 4);
    NumPtsArg(cmd, data2, 5);
    if (batched)
      cmd.DispatchIndirect(grid, 0, MTLSizeMake(M12_NT, 1, 1));
    else
      cmd.Dispatch(MTLSizeMake(iDivUp(numPts1, M12_T), numChunks, 1), MTLSizeMake(M12_NT, 1, 1));
    cmd.Barrier();
    cmd.Pipeline("MergeMaxCorr12");
    cmd.Buffer(data1.d_data, 0);
    cmd.Buffer(data2.d_data, 1);
    cmd.Bytes(p, 2);
    cmd.Buffer(ctx.best, 3);
    NumPtsArg(cmd, data1, 4);
    NumPtsArg(cmd, data2, 5);
    if (batched)
      cmd.DispatchIndirect(grid, 3*sizeof(uint32_t), MTLSizeMake(MERGE_NT, 1, 1));
    else
      cmd.DispatchThreads(MTLSizeMake(numPts1, 1, 1), MTLSizeMake(MERGE_NT, 1, 1));

    SiftData *d1 = &data1;
    AfterCall([=]() {
      if (d1->h_data!=NULL && d1->h_data!=d1->d_data)
        for (int i=0;i<d1->numPts;i++)
          memcpy(&d1->h_data[i].score, &d1->d_data[i].score, 5*sizeof(float));
    });
    if (batched)
      return 0.0;
    FinishCall(cmd);
    double time = Milliseconds(start);
#ifdef VERBOSE
    printf("MatchSiftData time =          %.2f ms\n", time);
#endif
    return time;
  }
}

// FindHomography and FindHomographySeeded: the RANSAC samples are drawn on the CPU, with
// rand(), or with the seeded generator of cudaSiftRandom.h if seed is not NULL
static double FindHomographyCPU(SiftData &data, float *homography, int *numMatches, int numLoops, float minScore, float maxAmbiguity, float thresh, const unsigned int *seed)
{
  @autoreleasepool {
    std::lock_guard<std::mutex> callLock(GetContext().callMutex);
    if (GetContext().batch)
      Fail("FindHomography and FindHomographySeeded cannot be called in a batch (BeginSiftBatch)");
    CompleteBatch();
    *numMatches = 0;
    homography[0] = homography[4] = homography[8] = 1.0f;
    homography[1] = homography[2] = homography[3] = 0.0f;
    homography[5] = homography[6] = homography[7] = 0.0f;
    if (data.d_data==NULL)
      return 0.0;
    SiftPoint *d_sift = data.d_data;   // unified memory, readable on the CPU
    auto start = std::chrono::steady_clock::now();
    numLoops = iDivUp(numLoops,16)*16;
    int numPts = data.numPts;
    if (numPts<8)
      return 0.0;
    int numPtsUp = iDivUp(numPts, 16)*16;
    float *d_coord = (float *)MetalAlloc(4*sizeof(float)*numPtsUp);
    int *d_randPts = (int *)MetalAlloc(4*sizeof(int)*numLoops);
    float *d_homo = (float *)MetalAlloc(8*sizeof(float)*numLoops);
    int *validPts = (int *)malloc(sizeof(int)*numPts);
    int numValid = 0;
    for (int i=0;i<numPts;i++) {
      if (d_sift[i].score>minScore && d_sift[i].ambiguity<maxAmbiguity)
        validPts[numValid++] = i;
    }
    if (numValid>=8) {
      // Coordinates of the valid points only, so that the inliers are counted among the
      // matches the hypotheses are sampled from
      const int numValidUp = iDivUp(numValid, 16)*16;
      for (int i=0;i<numValid;i++) {
        const SiftPoint &pt = d_sift[validPts[i]];
        d_coord[i+0*numValidUp] = pt.xpos;
        d_coord[i+1*numValidUp] = pt.ypos;
        d_coord[i+2*numValidUp] = pt.match_xpos;
        d_coord[i+3*numValidUp] = pt.match_ypos;
      }
      std::vector<bool> sameOrientation(numLoops);
      for (int i=0;i<numLoops;i++) {
        int p[4];
        if (seed)
          SiftRandomSample4(*seed, i, numValid, p);
        else {
          p[0] = rand() % numValid;
          p[1] = rand() % numValid;
          p[2] = rand() % numValid;
          p[3] = rand() % numValid;
          while (p[1]==p[0]) p[1] = rand() % numValid;
          while (p[2]==p[0] || p[2]==p[1]) p[2] = rand() % numValid;
          while (p[3]==p[0] || p[3]==p[1] || p[3]==p[2]) p[3] = rand() % numValid;
        }
        float x1[4], y1[4], x2[4], y2[4];
        for (int k=0;k<4;k++) {
          d_randPts[i+k*numLoops] = p[k];
          x1[k] = d_coord[p[k]+0*numValidUp];
          y1[k] = d_coord[p[k]+1*numValidUp];
          x2[k] = d_coord[p[k]+2*numValidUp];
          y2[k] = d_coord[p[k]+3*numValidUp];
        }
        sameOrientation[i] = SiftSameOrientation(x1, y1, x2, y2);
      }
      HomographyParams p = {numValidUp, numLoops, thresh*thresh, numValid};
      Commands cmd;
      cmd.Pipeline("ComputeHomographies");
      cmd.Buffer(d_coord, 0);
      cmd.Buffer(d_randPts, 1);
      cmd.Buffer(d_homo, 2);
      cmd.Bytes(p, 3);
      cmd.Dispatch(MTLSizeMake(numLoops/16, 1, 1), MTLSizeMake(16, 1, 1));
      cmd.Pipeline("TestHomographies");
      cmd.Buffer(d_coord, 0);
      cmd.Buffer(d_homo, 1);
      cmd.Buffer(d_randPts, 2);
      cmd.Bytes(p, 3);
      cmd.Dispatch(MTLSizeMake(1, numLoops/TESTHOMO_LOOPS, 1), MTLSizeMake(TESTHOMO_TESTS, TESTHOMO_LOOPS, 1));
      cmd.Finish();
      int maxIndex = -1, maxCount = -1;
      for (int i=0;i<numLoops;i++)
        if (sameOrientation[i] && d_randPts[i]>maxCount) {
          maxCount = d_randPts[i];
          maxIndex = i;
        }
      if (maxIndex>=0) {
        *numMatches = maxCount;
        for (int j=0;j<8;j++)
          homography[j] = d_homo[maxIndex + j*numLoops];
      }
    }
    free(validPts);
    MetalFree(d_homo);
    MetalFree(d_randPts);
    MetalFree(d_coord);
    double time = Milliseconds(start);
#ifdef VERBOSE
    printf("FindHomography time =         %.2f ms\n", time);
#endif
    return time;
  }
}

double FindHomography(SiftData &data, float *homography, int *numMatches, int numLoops, float minScore, float maxAmbiguity, float thresh)
{
  return FindHomographyCPU(data, homography, numMatches, numLoops, minScore, maxAmbiguity, thresh, NULL);
}

double FindHomographySeeded(SiftData &data, float *homography, int *numMatches, int numLoops, float minScore, float maxAmbiguity, float thresh, unsigned int seed)
{
  return FindHomographyCPU(data, homography, numMatches, numLoops, minScore, maxAmbiguity, thresh, &seed);
}

static void SetIdentity(float *homography)
{
  for (int i=0;i<9;i++)
    homography[i] = (i%4==0 ? 1.0f : 0.0f);
}

double FindHomographyGPU(SiftData &data, float *homography, int *numMatches, int numLoops, float minScore, float maxAmbiguity, float thresh, unsigned int seed)
{
  @autoreleasepool {
    std::lock_guard<std::mutex> callLock(GetContext().callMutex);
    auto start = std::chrono::steady_clock::now();
    Context &ctx = GetContext();
    const bool batched = (ctx.batch!=nullptr);
    if (!batched)
      CompleteBatch();
    *numMatches = 0;
    SetIdentity(homography);
    if (data.d_data==NULL || (!batched && data.numPts<8))
      return 0.0;
    numLoops = iDivUp(numLoops, 16)*16;
    const int stride = iDivUp(data.maxPts, 16)*16;
    const MTLResourceOptions priv = MTLResourceStorageModePrivate;
    id<MTLBuffer> coord = NewBuffer(4*sizeof(float)*stride, priv);
    id<MTLBuffer> valid = NewBuffer(sizeof(int)*data.maxPts, priv);
    id<MTLBuffer> validCoord = NewBuffer(4*sizeof(float)*stride, priv);   // coordinates of the valid points
    id<MTLBuffer> numValid = NewBuffer(sizeof(int), priv);
    id<MTLBuffer> randPts = NewBuffer(4*sizeof(int)*numLoops, priv);   // also the counts
    id<MTLBuffer> sameOrientation = NewBuffer(sizeof(int)*numLoops, priv);
    id<MTLBuffer> homo = NewBuffer(8*sizeof(float)*numLoops, priv);
    id<MTLBuffer> result = NewBuffer(sizeof(HomographyResult), MTLResourceStorageModeShared);
    HomographyGPUParams p = {numLoops, stride, minScore, maxAmbiguity, thresh*thresh, seed};
    HomographyParams hp = {stride, numLoops, thresh*thresh, 0};   // ComputeHomographies only

    std::unique_ptr<Commands> local;
    Commands &cmd = *CallCommands(local, true);
    cmd.Pipeline("HomographyPrepare");
    cmd.Buffer(data.d_data, 0);
    NumPtsArg(cmd, data, 1);
    cmd.Buffer(coord, 2);
    cmd.Buffer(valid, 3);
    cmd.Buffer(numValid, 4);
    cmd.Bytes(p, 5);
    cmd.Dispatch(MTLSizeMake(1, 1, 1), MTLSizeMake(HOMOGRAPHY_NT, 1, 1));
    cmd.Barrier();
    cmd.Pipeline("HomographyGather");
    cmd.Buffer(coord, 0);
    cmd.Buffer(valid, 1);
    cmd.Buffer(numValid, 2);
    cmd.Buffer(validCoord, 3);
    cmd.Bytes(p, 4);
    cmd.DispatchThreads(MTLSizeMake(batched ? data.maxPts : data.numPts, 1, 1), MTLSizeMake(256, 1, 1));
    cmd.Barrier();
    cmd.Pipeline("HomographySample");
    cmd.Buffer(validCoord, 0);
    cmd.Buffer(numValid, 1);
    cmd.Buffer(randPts, 2);
    cmd.Bytes(p, 3);
    cmd.Buffer(sameOrientation, 4);
    cmd.DispatchThreads(MTLSizeMake(numLoops, 1, 1), MTLSizeMake(64, 1, 1));
    cmd.Barrier();
    cmd.Pipeline("ComputeHomographies");
    cmd.Buffer(validCoord, 0);
    cmd.Buffer(randPts, 1);
    cmd.Buffer(homo, 2);
    cmd.Bytes(hp, 3);
    cmd.Dispatch(MTLSizeMake(numLoops/16, 1, 1), MTLSizeMake(16, 1, 1));
    cmd.Barrier();
    cmd.Pipeline("TestHomographiesGPU");
    cmd.Buffer(validCoord, 0);
    cmd.Buffer(homo, 1);
    cmd.Buffer(randPts, 2);
    cmd.Bytes(p, 3);
    cmd.Buffer(numValid, 4);
    cmd.Dispatch(MTLSizeMake(1, iDivUp(numLoops, 16*TESTHOMO_PER), 1), MTLSizeMake(16, 16, 1));
    cmd.Barrier();
    cmd.Pipeline("HomographyBest");
    cmd.Buffer(randPts, 0);
    cmd.Buffer(homo, 1);
    cmd.Buffer(numValid, 2);
    cmd.Buffer(result, 3);
    cmd.Bytes(p, 4);
    cmd.Buffer(sameOrientation, 5);
    cmd.Dispatch(MTLSizeMake(1, 1, 1), MTLSizeMake(HOMOGRAPHY_NT, 1, 1));
    if (batched)
      ctx.homographies[homography] = result;   // for ImproveHomographyGPU in the same batch
    AfterCall([=]() {
      const HomographyResult *r = (const HomographyResult *)[result contents];
      memcpy(homography, r->h, sizeof(r->h));
      *numMatches = r->count;
    });
    if (batched)
      return 0.0;
    FinishCall(cmd);
    double time = Milliseconds(start);
#ifdef VERBOSE
    printf("FindHomographyGPU time =      %.2f ms\n", time);
#endif
    return time;
  }
}

double ImproveHomographyGPU(SiftData &data, float *homography, int *numFit, int numLoops, float minScore, float maxAmbiguity, float thresh)
{
  @autoreleasepool {
    std::lock_guard<std::mutex> callLock(GetContext().callMutex);
    auto start = std::chrono::steady_clock::now();
    Context &ctx = GetContext();
    const bool batched = (ctx.batch!=nullptr);
    if (!batched)
      CompleteBatch();
    *numFit = 0;
    if (data.d_data==NULL)
      return 0.0;
    // The homography: from a previous call of the batch, else from the host
    id<MTLBuffer> input = nil;
    if (batched && ctx.homographies.count(homography))
      input = ctx.homographies[homography];
    else {
      input = NewBuffer(sizeof(HomographyResult), MTLResourceStorageModeShared);
      HomographyResult *r = (HomographyResult *)[input contents];
      memcpy(r->h, homography, sizeof(r->h));
      r->count = 0;
    }
    id<MTLBuffer> result = NewBuffer(sizeof(HomographyResult), MTLResourceStorageModeShared);
    HomographyGPUParams p = {0, 0, minScore, maxAmbiguity, thresh*thresh, 0};
    id<MTLBuffer> state = NewBuffer(IMPROVE_STATE_SIZE, MTLResourceStorageModePrivate);
    std::unique_ptr<Commands> local;
    Commands &cmd = *CallCommands(local, true);
    cmd.Pipeline("ImproveInit", true);
    cmd.Buffer(input, 0);
    cmd.Buffer(state, 1);
    cmd.DispatchThreads(MTLSizeMake(8, 1, 1), MTLSizeMake(8, 1, 1));
    for (int loop=0;loop<numLoops;loop++) {
      cmd.Barrier();
      cmd.Pipeline("ImproveAccumulate", true);
      cmd.Buffer(data.d_data, 0);
      NumPtsArg(cmd, data, 1);
      cmd.Buffer(state, 2);
      cmd.Bytes(p, 3);
      cmd.Dispatch(MTLSizeMake(IMPROVE_GROUPS, 1, 1), MTLSizeMake(IMPROVE_NT, 1, 1));
      cmd.Barrier();
      cmd.Pipeline("ImproveSolve", true);
      cmd.Buffer(state, 0);
      cmd.Dispatch(MTLSizeMake(1, 1, 1), MTLSizeMake(32, 1, 1));
    }
    cmd.Barrier();
    cmd.Pipeline("ImproveFinal", true);
    cmd.Buffer(data.d_data, 0);
    NumPtsArg(cmd, data, 1);
    cmd.Buffer(state, 2);
    cmd.Bytes(p, 3);
    cmd.Dispatch(MTLSizeMake(IMPROVE_GROUPS, 1, 1), MTLSizeMake(IMPROVE_NT, 1, 1));
    cmd.Barrier();
    cmd.Pipeline("ImproveCount", true);
    cmd.Buffer(state, 0);
    cmd.Buffer(result, 1);
    cmd.DispatchThreads(MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
    if (batched)
      ctx.homographies[homography] = result;
    SiftData *d = &data;
    AfterCall([=]() {
      const HomographyResult *r = (const HomographyResult *)[result contents];
      memcpy(homography, r->h, sizeof(r->h));
      *numFit = r->count;
      if (d->h_data!=NULL && d->h_data!=d->d_data)
        for (int i=0;i<d->numPts;i++)
          d->h_data[i].match_error = d->d_data[i].match_error;
    });
    if (batched)
      return 0.0;
    FinishCall(cmd);
    return Milliseconds(start);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Batches (cudaSiftBatch.h)
///////////////////////////////////////////////////////////////////////////////

void BeginSiftBatch()
{
  @autoreleasepool {
    Context &ctx = GetContext();
    if (ctx.batch)
      Fail("BeginSiftBatch: a batch is already open");
    CompleteBatch();   // the previous batch must complete first (shared counters and scratch)
    ctx.batch = new Commands(true);
  }
}

void EndSiftBatch()
{
  @autoreleasepool {
    Context &ctx = GetContext();
    if (!ctx.batch)
      Fail("EndSiftBatch: no open batch");
    ctx.batch->Suspend();
    [ctx.batch->cb commit];
    ctx.submitted = ctx.batch->cb;
    delete ctx.batch;
    ctx.batch = nullptr;
    ctx.extracted.clear();
    ctx.homographies.clear();
  }
}

double WaitSiftBatch()
{
  @autoreleasepool {
    Context &ctx = GetContext();
    if (ctx.batch)
      EndSiftBatch();
    CompleteBatch();   // unless a call made after EndSiftBatch already did
    double time = ctx.batchTime;
    ctx.batchTime = 0.0;
    return time;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Interop with other Metal code (cudaSiftMetal.h)
///////////////////////////////////////////////////////////////////////////////

void *CudaSiftMetalDevice()
{
  return (__bridge void *)GetContext().device;
}

void *CudaSiftMetalQueue()
{
  return (__bridge void *)GetContext().queue;
}

void *CudaSiftMetalBuffer(const void *ptr, size_t *offset)
{
  NSUInteger off = 0;
  id<MTLBuffer> buffer = (ptr!=NULL ? LookupBuffer(ptr, off) : nil);
  if (offset)
    *offset = (buffer!=nil ? off : 0);
  return (__bridge void *)buffer;
}

void *CudaSiftMetalBatchCommandBuffer()
{
  Context &ctx = GetContext();
  if (!ctx.batch)
    return nullptr;
  ctx.batch->Suspend();
  return (__bridge void *)ctx.batch->cb;
}

void *CudaSiftMetalNumPtsBuffer(SiftData &data, size_t *offset)
{
  @autoreleasepool {
    Context &ctx = GetContext();
    if (offset)
      *offset = 0;
    if (data.d_data==NULL)
      return nullptr;
    if (ctx.batch)
      return (__bridge void *)BatchNumPts(data);
    std::lock_guard<std::mutex> callLock(ctx.callMutex);
    CompleteBatch();   // a submitted batch may still write the count, and sets data.numPts
    id<MTLBuffer> n = NumPtsBuffer(data);
    *(int *)[n contents] = data.numPts;
    return (__bridge void *)n;
  }
}

///////////////////////////////////////////////////////////////////////////////
// CudaImage (cudaImage.h)
///////////////////////////////////////////////////////////////////////////////

int iDivUp(int a, int b) { return (a%b != 0) ? (a/b + 1) : (a/b); }
int iDivDown(int a, int b) { return a/b; }
int iAlignUp(int a, int b) { return (a%b != 0) ?  (a - a%b + b) : a; }
int iAlignDown(int a, int b) { return a - a%b; }

// Pitch (in floats) of the texture data of each CudaImage, for cudaMemcpy2DFromArray
static std::map<const void *, int> &TexturePitches()
{
  static std::map<const void *, int> pitches;
  return pitches;
}

static void FreeTexture(float *t_data)
{
  {
    std::lock_guard<std::mutex> lock(GetContext().mutex);
    TexturePitches().erase(t_data);
  }
  MetalFree(t_data);
}

void CudaImage::Allocate(int w, int h, int p, bool host, float *devmem, float *hostmem)
{
  if (d_internalAlloc && d_data!=NULL)   // free the data of a previous Allocate
    MetalFree(d_data);
  if (h_internalAlloc && h_data!=NULL)
    free(h_data);
  if (t_data!=NULL)
    FreeTexture(t_data);
  d_internalAlloc = false;
  h_internalAlloc = false;
  width = w;
  height = h;
  pitch = p;
  d_data = devmem;
  h_data = hostmem;
  t_data = NULL;
  if (devmem==NULL) {
    pitch = iAlignUp(w, 128);   // as cudaMallocPitch, aligned rows (also valid for Metal textures)
    d_data = (float *)MetalAlloc(sizeof(float)*pitch*height);
    if (d_data==NULL)
      printf("Failed to allocate device data\n");
    d_internalAlloc = true;
  }
  if (host && hostmem==NULL) {
    h_data = (float *)malloc(sizeof(float)*pitch*height);
    h_internalAlloc = true;
  }
}

CudaImage::CudaImage() :
  width(0), height(0), d_data(NULL), h_data(NULL), t_data(NULL), d_internalAlloc(false), h_internalAlloc(false)
{

}

CudaImage::~CudaImage()
{
  if (d_internalAlloc && d_data!=NULL)
    MetalFree(d_data);
  d_data = NULL;
  if (h_internalAlloc && h_data!=NULL)
    free(h_data);
  h_data = NULL;
  if (t_data!=NULL)
    FreeTexture(t_data);
  t_data = NULL;
}

double CudaImage::Download()
{
  auto start = std::chrono::steady_clock::now();
  if (d_data!=NULL && h_data!=NULL)
    for (int y=0;y<height;y++)
      memcpy(d_data + y*pitch, h_data + y*width, sizeof(float)*width);
  double time = Milliseconds(start);
#ifdef VERBOSE
  printf("Download time =               %.2f ms\n", time);
#endif
  return time;
}

double CudaImage::Readback()
{
  auto start = std::chrono::steady_clock::now();
  for (int y=0;y<height;y++)
    memcpy(h_data + y*width, d_data + y*pitch, sizeof(float)*width);
  double time = Milliseconds(start);
#ifdef VERBOSE
  printf("Readback time =               %.2f ms\n", time);
#endif
  return time;
}

double CudaImage::InitTexture()
{
  t_data = (float *)MetalAlloc(sizeof(float)*pitch*height);
  {
    std::lock_guard<std::mutex> lock(GetContext().mutex);
    TexturePitches()[t_data] = pitch;
  }
  if (t_data==NULL)
    printf("Failed to allocated texture data\n");
  return 0.0;
}

double CudaImage::CopyToTexture(CudaImage &dst, bool host)
{
  if (dst.t_data==NULL) {
    printf("Error CopyToTexture: No texture data\n");
    return 0.0;
  }
  if ((!host || h_data==NULL) && (host || d_data==NULL)) {
    printf("Error CopyToTexture: No source data\n");
    return 0.0;
  }
  auto start = std::chrono::steady_clock::now();
  if (host)   // h_data rows are width floats
    for (int y=0;y<dst.height;y++)
      memcpy(dst.t_data + y*pitch, h_data + y*width, sizeof(float)*width);
  else
    memcpy(dst.t_data, d_data, sizeof(float)*pitch*dst.height);
  return Milliseconds(start);
}

///////////////////////////////////////////////////////////////////////////////
// Subset of the CUDA runtime used by the tests and examples (compat/cuda_runtime.h)
///////////////////////////////////////////////////////////////////////////////

struct CudaSiftMetalEvent {
  std::chrono::steady_clock::time_point time;
};

cudaError_t cudaEventCreate(cudaEvent_t *event)
{
  *event = new CudaSiftMetalEvent();
  return cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
  delete event;
  return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t)
{
  event->time = std::chrono::steady_clock::now();   // all CudaSift calls are synchronous
  return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t)
{
  return cudaSuccess;
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
  *ms = std::chrono::duration<float, std::milli>(end->time - start->time).count();
  return cudaSuccess;
}

cudaError_t cudaMalloc(void **ptr, size_t size)
{
  *ptr = MetalAlloc(size);
  return cudaSuccess;
}

cudaError_t cudaFree(void *ptr)
{
  if (ptr==NULL)
    return cudaSuccess;
  Context &ctx = GetContext();
  {
    std::lock_guard<std::mutex> lock(ctx.mutex);
    if (!ctx.buffers.count((uintptr_t)ptr))
      return cudaErrorInvalidValue;
  }
  MetalFree(ptr);
  return cudaSuccess;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind)
{
  memmove(dst, src, count);
  return cudaSuccess;
}

cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch, const cudaArray *src, size_t wOffset, size_t hOffset,
                                  size_t width, size_t height, cudaMemcpyKind)
{
  int pitch;
  {
    std::lock_guard<std::mutex> lock(GetContext().mutex);
    auto it = TexturePitches().find(src);
    if (it==TexturePitches().end())
      return cudaErrorInvalidValue;
    pitch = it->second;
  }
  const char *s = (const char *)src + (hOffset*pitch*sizeof(float) + wOffset);
  for (size_t y=0;y<height;y++)
    memcpy((char *)dst + y*dpitch, s + y*pitch*sizeof(float), width);
  return cudaSuccess;
}

cudaError_t cudaGetLastError()
{
  return cudaSuccess;
}

cudaError_t cudaGetDeviceCount(int *count)
{
  *count = 1;
  return cudaSuccess;
}

cudaError_t cudaGetDevice(int *device)
{
  *device = 0;
  return cudaSuccess;
}

cudaError_t cudaSetDevice(int device)
{
  return device==0 ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t cudaDeviceSynchronize()   // calls complete before they return; this completes a submitted batch
{
  std::lock_guard<std::mutex> callLock(GetContext().callMutex);
  CompleteBatch();
  return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int)
{
  @autoreleasepool {
    Context &ctx = GetContext();
    memset(prop, 0, sizeof(cudaDeviceProp));
    snprintf(prop->name, sizeof(prop->name), "%s", [[ctx.device name] UTF8String]);
    prop->totalGlobalMem = [ctx.device recommendedMaxWorkingSetSize];
    prop->sharedMemPerBlock = [ctx.device maxThreadgroupMemoryLength];
    prop->maxThreadsPerBlock = (int)[ctx.device maxThreadsPerThreadgroup].width;
    return cudaSuccess;
  }
}

// ExtractSiftCompat (cudaSiftCompat.h): its host code, which uses the context and the command
// encoding above
#include "cudaSiftCompatMetal.mm"
