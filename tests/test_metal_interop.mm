//********************************************************//
// Test: access to CudaSift's Metal objects               //
//        (cudaSiftMetal.h, Metal only)                   //
//********************************************************//

#import <Metal/Metal.h>

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cudaImage.h"
#include "cudaSift.h"
#include "cudaSiftBatch.h"
#include "cudaSiftMetal.h"

int testsPassed = 0;
int testsFailed = 0;

void CHECK(bool condition, const char *msg)
{
  if (condition) {
    std::cout << "  [PASS] " << msg << std::endl;
    testsPassed++;
  } else {
    std::cout << "  [FAIL] " << msg << std::endl;
    testsFailed++;
  }
}

static bool Load(CudaImage &img, cv::Mat &fimg, const char *file)
{
  cv::Mat src = cv::imread(file, 0);
  if (src.empty())
    return false;
  src.convertTo(fimg, CV_32FC1);
  img.Allocate(fimg.cols, fimg.rows, iAlignUp(fimg.cols, 128), false, NULL, (float *)fimg.data);
  img.Download();
  return true;
}

// The buffer behind pointers allocated by CudaSift, and the device and queue
void TestBuffers()
{
  std::cout << "\n--- Test: buffers, device and queue ---" << std::endl;
  SiftData data;
  InitSiftData(data, 1000, true, true);
  CudaImage img;
  img.Allocate(640, 480, 640, false, NULL, NULL);

  size_t off = 1;
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)CudaSiftMetalBuffer(data.d_data, &off);
  CHECK(buffer!=nil && off==0 && (char *)[buffer contents] + off==(char *)data.d_data, "Buffer of SiftData::d_data");
  off = 0;
  id<MTLBuffer> inside = (__bridge id<MTLBuffer>)CudaSiftMetalBuffer(data.d_data + 10, &off);
  CHECK(inside==buffer && off==10*sizeof(SiftPoint), "Buffer and offset of a pointer inside the allocation");
  off = 1;
  id<MTLBuffer> imgBuffer = (__bridge id<MTLBuffer>)CudaSiftMetalBuffer(img.d_data, &off);
  CHECK(imgBuffer!=nil && (char *)[imgBuffer contents] + off==(char *)img.d_data, "Buffer of CudaImage::d_data");
  void *host = malloc(1024);
  off = 1;
  CHECK(CudaSiftMetalBuffer(host, &off)==nullptr && off==0, "nullptr for memory not allocated by CudaSift");
  free(host);
  CHECK(CudaSiftMetalBuffer(NULL, NULL)==nullptr, "nullptr for NULL");

  id<MTLDevice> device = (__bridge id<MTLDevice>)CudaSiftMetalDevice();
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)CudaSiftMetalQueue();
  CHECK(device!=nil && [buffer device]==device, "The buffers are on CudaSiftMetalDevice()");
  CHECK(queue!=nil && [queue device]==device, "CudaSiftMetalQueue() is on CudaSiftMetalDevice()");
  CHECK(CudaSiftMetalBatchCommandBuffer()==nullptr, "No batch command buffer outside a batch");
  FreeSiftData(data);
}

// The caller's kernel: copies the point count to out
static const char *kCopyCount =
  "kernel void CopyCount(device const int *count [[buffer(0)]], device int *out [[buffer(1)]])\n"
  "{ out[0] = count[0]; }\n";

// Work encoded into a batch between CudaSift's calls, reading a point count computed in the batch
void TestBatchCommandBuffer()
{
  std::cout << "\n--- Test: encoding into a batch ---" << std::endl;
  CudaImage img1, img2;
  cv::Mat f1, f2;
  if (!Load(img1, f1, "data/img1.png") || !Load(img2, f2, "data/img2.png")) {
    CHECK(false, "data/img1.png and data/img2.png");
    return;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)CudaSiftMetalDevice();
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kCopyCount] options:nil error:&error];
  id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"CopyCount"] error:&error];
  id<MTLBuffer> out = [device newBufferWithLength:2*sizeof(int) options:MTLResourceStorageModeShared];
  if (pso==nil || out==nil) {
    CHECK(false, "Caller's kernel compiled");
    return;
  }
  ((int *)[out contents])[0] = ((int *)[out contents])[1] = -1;
  float *mem = AllocSiftTempMemory(img1.width, img1.height, 5, false);

  // Reference point counts: the same extractions outside a batch
  SiftData r1, r2;
  InitSiftData(r1, 32768, true, true);
  InitSiftData(r2, 32768, true, true);
  ExtractSift(r1, img1, 5, 1.0f, 3.0f, 0.0f, false, mem);
  ExtractSift(r2, img2, 5, 1.0f, 3.0f, 0.0f, false, mem);

  SiftData s1, s2;
  InitSiftData(s1, 32768, true, true);
  InitSiftData(s2, 32768, true, true);
  float H[9];
  int n = 0;
  BeginSiftBatch();
  ExtractSift(s1, img1, 5, 1.0f, 3.0f, 0.0f, false, mem);
  ExtractSift(s2, img2, 5, 1.0f, 3.0f, 0.0f, false, mem);
  MatchSiftData(s1, s2);
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)CudaSiftMetalBatchCommandBuffer();
  CHECK(cb!=nil, "Batch command buffer in a batch");
  size_t off1 = 1, off2 = 1;
  id<MTLBuffer> count1 = (__bridge id<MTLBuffer>)CudaSiftMetalNumPtsBuffer(s1, &off1);
  id<MTLBuffer> count2 = (__bridge id<MTLBuffer>)CudaSiftMetalNumPtsBuffer(s2, &off2);
  if (cb!=nil && count1!=nil && count2!=nil) {
    for (int k=0;k<2;k++) {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:(k==0 ? count1 : count2) offset:(k==0 ? off1 : off2) atIndex:0];
      [enc setBuffer:out offset:k*sizeof(int) atIndex:1];
      [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [enc endEncoding];
    }
  }
  FindHomographyGPU(s1, H, &n, 10000, 0.00f, 0.80f, 5.0f, 3);   // after the caller's work
  EndSiftBatch();
  WaitSiftBatch();

  const int *counts = (const int *)[out contents];
  std::cout << "    " << s1.numPts << " and " << s2.numPts << " points, counts read on the GPU: " << counts[0] << " and " << counts[1] << std::endl;
  CHECK(s1.numPts>0 && counts[0]==s1.numPts && counts[1]==s2.numPts, "Point counts read in the batch equal numPts");
  CHECK(s1.numPts==r1.numPts && s2.numPts==r2.numPts, "Same point counts as outside the batch");
  // The same call outside a batch, on the same points (FindHomographyGPU depends on their order)
  float Hr[9];
  int nr = 0;
  FindHomographyGPU(s1, Hr, &nr, 10000, 0.00f, 0.80f, 5.0f, 3);
  CHECK(n==nr && n>0 && memcmp(H, Hr, sizeof(H))==0, "Same homography as outside the batch");

  // Outside a batch, the count buffer holds numPts
  size_t off = 1;
  id<MTLBuffer> count = (__bridge id<MTLBuffer>)CudaSiftMetalNumPtsBuffer(r1, &off);
  CHECK(count!=nil && *(const int *)((const char *)[count contents] + off)==r1.numPts, "Count buffer outside a batch holds numPts");

  FreeSiftTempMemory(mem);
  FreeSiftData(r1);
  FreeSiftData(r2);
  FreeSiftData(s1);
  FreeSiftData(s2);
}

int main(int argc, char **argv)
{
  std::cout << "=============================================" << std::endl;
  std::cout << "   CudaSift Metal Interop Tests" << std::endl;
  std::cout << "=============================================" << std::endl;
  InitCuda(argc>1 ? std::atoi(argv[1]) : 0);
  @autoreleasepool {
    TestBuffers();
    TestBatchCommandBuffer();
  }
  std::cout << "\n=============================================" << std::endl;
  std::cout << "  Results: " << testsPassed << " passed, " << testsFailed << " failed" << std::endl;
  std::cout << "=============================================" << std::endl;
  return testsFailed > 0 ? 1 : 0;
}
