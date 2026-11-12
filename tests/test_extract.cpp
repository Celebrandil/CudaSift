//********************************************************//
// Test: SIFT Feature Extraction Correctness              //
//********************************************************//

#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include <thread>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cudaImage.h"
#include "cudaSift.h"

#include <cuda_runtime.h>

int testsPassed = 0;
int testsFailed = 0;
int testsSkipped = 0;

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

void SKIP(const char *msg)
{
  std::cout << "  [SKIP] " << msg << std::endl;
  testsSkipped++;
}

void TestBasicExtraction()
{
  std::cout << "\n--- Test: Basic Extraction ---" << std::endl;

  cv::Mat img = cv::imread("data/img1.png", 0);
  CHECK(!img.empty(), "Image loaded successfully");
  if (img.empty()) return;

  cv::Mat fimg;
  img.convertTo(fimg, CV_32FC1);

  CudaImage cudaImg;
  cudaImg.Allocate(fimg.cols, fimg.rows, iAlignUp(fimg.cols, 128), false, NULL, (float *)fimg.data);
  cudaImg.Download();

  SiftData siftData;
  InitSiftData(siftData, 32768, true, true);

  float *mem = AllocSiftTempMemory(fimg.cols, fimg.rows, 5, false);
  ExtractSift(siftData, cudaImg, 5, 1.0f, 3.0f, 0.0f, false, mem);
  FreeSiftTempMemory(mem);

  CHECK(siftData.numPts > 0, "Features detected (numPts > 0)");
  CHECK(siftData.numPts < 32768, "Features within capacity");

  // Check feature validity
  bool allValid = true;
  for (int i = 0; i < std::min(siftData.numPts, 100); i++) {
    SiftPoint &pt = siftData.h_data[i];
    if (pt.xpos < 0 || pt.xpos >= fimg.cols ||
        pt.ypos < 0 || pt.ypos >= fimg.rows ||
        pt.scale <= 0 || std::isnan(pt.scale) ||
        pt.orientation < 0 || pt.orientation >= 360) {
      allValid = false;
      break;
    }
  }
  CHECK(allValid, "All feature positions/scales/orientations are valid");

  // Check descriptor normalization (should be roughly unit length)
  bool descriptorsOk = true;
  for (int i = 0; i < std::min(siftData.numPts, 50); i++) {
    float sum = 0.0f;
    for (int j = 0; j < 128; j++)
      sum += siftData.h_data[i].data[j] * siftData.h_data[i].data[j];
    float norm = sqrtf(sum);
    if (norm < 0.8f || norm > 1.2f) {
      descriptorsOk = false;
      std::cout << "    Descriptor " << i << " norm = " << norm << std::endl;
      break;
    }
  }
  CHECK(descriptorsOk, "Descriptor vectors are approximately unit length");

  FreeSiftData(siftData);
}

void TestDifferentThresholds()
{
  std::cout << "\n--- Test: Threshold Sensitivity ---" << std::endl;

  cv::Mat img = cv::imread("data/img1.png", 0);
  if (img.empty()) return;
  cv::Mat fimg;
  img.convertTo(fimg, CV_32FC1);

  CudaImage cudaImg;
  cudaImg.Allocate(fimg.cols, fimg.rows, iAlignUp(fimg.cols, 128), false, NULL, (float *)fimg.data);
  cudaImg.Download();

  int prevCount = 99999;
  bool monotonic = true;
  float thresholds[] = {1.0f, 3.0f, 5.0f, 10.0f};
  for (float t : thresholds) {
    SiftData siftData;
    InitSiftData(siftData, 32768, true, true);
    float *mem = AllocSiftTempMemory(fimg.cols, fimg.rows, 5, false);
    ExtractSift(siftData, cudaImg, 5, 1.0f, t, 0.0f, false, mem);
    FreeSiftTempMemory(mem);

    std::cout << "    thresh=" << t << " -> " << siftData.numPts << " features" << std::endl;
    if (siftData.numPts > prevCount)
      monotonic = false;
    prevCount = siftData.numPts;
    FreeSiftData(siftData);
  }
  CHECK(monotonic, "Higher threshold = fewer features (monotonic decrease)");
}

void TestDifferentOctaves()
{
  std::cout << "\n--- Test: Octave Count ---" << std::endl;

  cv::Mat img = cv::imread("data/img1.png", 0);
  if (img.empty()) return;
  cv::Mat fimg;
  img.convertTo(fimg, CV_32FC1);

  CudaImage cudaImg;
  cudaImg.Allocate(fimg.cols, fimg.rows, iAlignUp(fimg.cols, 128), false, NULL, (float *)fimg.data);
  cudaImg.Download();

  int counts[5];
  for (int oct = 3; oct <= 6; oct++) {
    SiftData siftData;
    InitSiftData(siftData, 32768, true, true);
    float *mem = AllocSiftTempMemory(fimg.cols, fimg.rows, oct, false);
    ExtractSift(siftData, cudaImg, oct, 1.0f, 3.0f, 0.0f, false, mem);
    FreeSiftTempMemory(mem);
    counts[oct - 3] = siftData.numPts;
    std::cout << "    octaves=" << oct << " -> " << siftData.numPts << " features" << std::endl;
    FreeSiftData(siftData);
  }
  CHECK(counts[1] >= counts[0], "More octaves = more or equal features");
}

void TestReproducibility()
{
  std::cout << "\n--- Test: Reproducibility ---" << std::endl;

  cv::Mat img = cv::imread("data/img1.png", 0);
  if (img.empty()) return;
  cv::Mat fimg;
  img.convertTo(fimg, CV_32FC1);

  CudaImage cudaImg;
  cudaImg.Allocate(fimg.cols, fimg.rows, iAlignUp(fimg.cols, 128), false, NULL, (float *)fimg.data);
  cudaImg.Download();

  // Run twice
  SiftData siftData1, siftData2;
  InitSiftData(siftData1, 32768, true, true);
  InitSiftData(siftData2, 32768, true, true);

  float *mem = AllocSiftTempMemory(fimg.cols, fimg.rows, 5, false);
  ExtractSift(siftData1, cudaImg, 5, 1.0f, 3.0f, 0.0f, false, mem);
  ExtractSift(siftData2, cudaImg, 5, 1.0f, 3.0f, 0.0f, false, mem);
  FreeSiftTempMemory(mem);

  CHECK(siftData1.numPts == siftData2.numPts, "Same number of features on repeated runs");

  if (siftData1.numPts == siftData2.numPts && siftData1.numPts > 0) {
    // Features may be stored in different order due to GPU atomics.
    // Sort by (ypos, xpos) then compare.
    auto cmp = [](const SiftPoint &a, const SiftPoint &b) {
      if (a.ypos != b.ypos) return a.ypos < b.ypos;
      return a.xpos < b.xpos;
    };
    std::vector<SiftPoint> pts1(siftData1.h_data, siftData1.h_data + siftData1.numPts);
    std::vector<SiftPoint> pts2(siftData2.h_data, siftData2.h_data + siftData2.numPts);
    std::sort(pts1.begin(), pts1.end(), cmp);
    std::sort(pts2.begin(), pts2.end(), cmp);
    int matchCount = 0;
    for (int i = 0; i < (int)pts1.size(); i++) {
      if (fabs(pts1[i].xpos - pts2[i].xpos) < 0.1f &&
          fabs(pts1[i].ypos - pts2[i].ypos) < 0.1f) {
        matchCount++;
      }
    }
    float matchRatio = (float)matchCount / siftData1.numPts;
    std::cout << "    Matching positions (sorted): " << matchCount << "/" << siftData1.numPts
              << " (" << (matchRatio * 100.0f) << "%)" << std::endl;
    CHECK(matchRatio > 0.95f, "Feature positions are consistent across runs (>95%)");
  }

  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
}

// Regression test: sign errors in the sub-pixel refinement of the DoG
// extrema moved tilted, elongated blobs away from their centres and refined the scale the wrong way
void TestSubpixelRefinement()
{
  std::cout << "\n--- Test: Sub-pixel refinement on synthetic blobs ---" << std::endl;

  // Row 0-1: round blobs with sigma from 2 to 4; row 2-3: 4x2 blobs rotated by 0 to 168.75 degrees
  const int W = 1024, H = 512, N = 32;
  const float fr[12][2] = {{0.1f,0.3f},{0.4f,0.8f},{0.7f,0.2f},{0.25f,0.55f},{0.9f,0.45f},{0.5f,0.0f},
                           {0.0f,0.5f},{0.35f,0.15f},{0.65f,0.95f},{0.8f,0.7f},{0.15f,0.85f},{0.55f,0.35f}};
  float bx[N], by[N], bs[N];
  cv::Mat fimg(H, W, CV_32FC1, cv::Scalar(0.0f));
  for (int k = 0; k < N; k++) {
    bx[k] = 64 + 128*(k%8) + fr[k%12][0];
    by[k] = 64 + 128*(k/8) + fr[k%12][1];
    float s1 = 4.0f, s2 = 2.0f, ang = 11.25f*(k%16)*3.14159265f/180;
    if (k < 16)
      s1 = s2 = 2.0f*powf(2.0f, k/16.0f);
    bs[k] = s1;
    float ca = cosf(ang), sa = sinf(ang);
    for (int y = (int)by[k] - 40; y <= (int)by[k] + 40; y++)
      for (int x = (int)bx[k] - 40; x <= (int)bx[k] + 40; x++) {
        float u = ca*(x - bx[k]) + sa*(y - by[k]), v = -sa*(x - bx[k]) + ca*(y - by[k]);
        fimg.at<float>(y, x) += 100.0f*expf(-0.5f*(u*u/(s1*s1) + v*v/(s2*s2)));
      }
  }
  CudaImage cudaImg;
  cudaImg.Allocate(W, H, iAlignUp(W, 128), false, NULL, (float *)fimg.data);
  cudaImg.Download();
  SiftData data;
  InitSiftData(data, 4096, true, true);
  ExtractSift(data, cudaImg, 5, 1.0f, 3.0f, 0.0f, false);

  // For each blob, the keypoint of largest scale within 1 pixel of its centre
  float err[N], scale[N];
  for (int k = 0; k < N; k++) {
    err[k] = 1.0f;
    scale[k] = 0.0f;
    for (int i = 0; i < data.numPts; i++) {
      SiftPoint &p = data.h_data[i];
      float e = sqrtf((p.xpos - bx[k])*(p.xpos - bx[k]) + (p.ypos - by[k])*(p.ypos - by[k]));
      if (e < 1.0f && p.scale > scale[k]) {
        scale[k] = p.scale;
        err[k] = e;
      }
    }
  }
  int notIncreasing = 0;
  for (int k = 1; k < 16; k++)
    notIncreasing += (scale[k] <= scale[k-1]);
  float meanErr = 0.0f;
  for (int k = 16; k < N; k++)
    meanErr += err[k]/16;
  std::cout << "    round blobs, sigma " << bs[0] << " to " << bs[15] << ": scales " << scale[0] << " to "
            << scale[15] << ", " << notIncreasing << " of 15 steps not increasing" << std::endl;
  std::cout << "    tilted 4x2 blobs: mean position error " << meanErr << " px" << std::endl;
  CHECK(notIncreasing == 0, "Keypoint scale increases with the blob size");
  CHECK(meanErr < 0.15f, "Tilted elongated blobs located within 0.15 px on average");

  FreeSiftData(data);
}

// Regression test: gradients at an angle of exactly +pi relative to
// the keypoint orientation got angle bin 8, i.e. bin 0 of the next spatial cell, while those
// at -pi got bin 0 of their own cell. On a horizontal ramp with blobs centred on integer rows,
// keypoints have an orientation of exactly 180 degrees, many gradients are at +pi, and the
// image is symmetric about each keypoint row, so each descriptor must be symmetric: cell row
// v and bin b equal cell row 3-v and bin -b.
void TestDescriptorAnglePi()
{
  std::cout << "\n--- Test: Descriptor symmetry for gradients at +-pi ---" << std::endl;

  const int W = 1024, H = 600;
  cv::Mat fimg(H, W, CV_32FC1);
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++)
      fimg.at<float>(y, x) = 4.0f*x;
  const float sigma[4] = {2.5f, 3.0f, 4.0f, 5.0f};
  for (int r = 0; r < 4; r++)
    for (int k = 0; k < 6; k++) {
      float bx = 120 + 150*k + 0.13f*k, by = 90 + 140*r, s = sigma[r];
      for (int y = (int)by - 60; y <= (int)by + 60; y++)
        for (int x = (int)bx - 60; x <= (int)bx + 60; x++)
          fimg.at<float>(y, x) += 100.0f*expf(-0.5f*((x - bx)*(x - bx) + (y - by)*(y - by))/(s*s));
    }
  CudaImage cudaImg;
  cudaImg.Allocate(W, H, iAlignUp(W, 128), false, NULL, (float *)fimg.data);
  cudaImg.Download();
  SiftData data;
  InitSiftData(data, 4096, true, true);
  ExtractSift(data, cudaImg, 5, 1.0f, 3.0f, 0.0f, false);

  int num = 0;
  float maxAsym = 0.0f;
  for (int i = 0; i < data.numPts; i++) {
    const float *d = data.h_data[i].data;
    if (fabsf(data.h_data[i].orientation - 180.0f) > 0.01f)
      continue;
    num++;
    for (int v = 0; v < 4; v++)
      for (int h = 0; h < 4; h++)
        for (int b = 0; b < 8; b++)
          maxAsym = fmaxf(maxAsym, fabsf(d[(4*v + h)*8 + b] - d[(4*(3 - v) + h)*8 + (8 - b)%8]));
  }
  std::cout << "    " << num << " of " << data.numPts << " keypoints at 180 degrees, max asymmetry " << maxAsym << std::endl;
  CHECK(num >= 20 && maxAsym < 0.005f, "Symmetric descriptors for a symmetric image");

  FreeSiftData(data);
}

// Regression test: a descriptor sample with a zero gradient
// made the whole descriptor constant (all 128 values 1/sqrt(128))
void TestZeroGradientDescriptors()
{
  std::cout << "\n--- Test: Descriptors next to flat regions ---" << std::endl;

  cv::Mat img = cv::imread("data/img1.png", 0);
  if (img.empty()) return;
  cv::Mat bordered, fimg;
  cv::copyMakeBorder(img, bordered, 100, 100, 100, 100, cv::BORDER_CONSTANT, 0);
  bordered.convertTo(fimg, CV_32FC1);

  CudaImage cudaImg;
  cudaImg.Allocate(fimg.cols, fimg.rows, iAlignUp(fimg.cols, 128), false, NULL, (float *)fimg.data);
  cudaImg.Download();
  SiftData siftData;
  InitSiftData(siftData, 32768, true, true);
  ExtractSift(siftData, cudaImg, 5, 1.0f, 3.0f, 0.0f, false);

  int constant = 0;
  for (int i = 0; i < siftData.numPts; i++) {
    const float *d = siftData.h_data[i].data;
    bool same = true;
    for (int j = 1; j < 128 && same; j++)
      same = (d[j] == d[0]);
    constant += same;
  }
  std::cout << "    " << siftData.numPts << " features, " << constant << " constant descriptors" << std::endl;
  CHECK(siftData.numPts > 0 && constant == 0, "No constant descriptors next to a black border");
  FreeSiftData(siftData);
}

void TestScaleUp()
{
  std::cout << "\n--- Test: Scale-Up Mode ---" << std::endl;

  cv::Mat img = cv::imread("data/img1.png", 0);
  if (img.empty()) return;
  cv::Mat fimg;
  img.convertTo(fimg, CV_32FC1);

  CudaImage cudaImg;
  cudaImg.Allocate(fimg.cols, fimg.rows, iAlignUp(fimg.cols, 128), false, NULL, (float *)fimg.data);
  cudaImg.Download();

  SiftData normalData, scaleUpData;
  InitSiftData(normalData, 32768, true, true);
  InitSiftData(scaleUpData, 32768, true, true);

  float *mem = AllocSiftTempMemory(fimg.cols, fimg.rows, 5, false);
  ExtractSift(normalData, cudaImg, 5, 1.0f, 3.0f, 0.0f, false, mem);
  FreeSiftTempMemory(mem);

  float *memUp = AllocSiftTempMemory(fimg.cols, fimg.rows, 5, true);
  ExtractSift(scaleUpData, cudaImg, 5, 1.0f, 3.0f, 0.0f, true, memUp);
  FreeSiftTempMemory(memUp);

  std::cout << "    Normal: " << normalData.numPts << " features" << std::endl;
  std::cout << "    ScaleUp: " << scaleUpData.numPts << " features" << std::endl;
  CHECK(scaleUpData.numPts > normalData.numPts, "Scale-up mode detects more features");

  FreeSiftData(normalData);
  FreeSiftData(scaleUpData);
}

// Regression test: with scaleUp and no points found, RescalePositions
// launched 0 blocks, and the launch error made the process exit
void TestScaleUpNoPoints()
{
  std::cout << "\n--- Test: Scale-up extraction with no features ---" << std::endl;

  cv::Mat fimg(240, 320, CV_32FC1, cv::Scalar(128.0f));   // uniform image: no features
  CudaImage cudaImg;
  cudaImg.Allocate(fimg.cols, fimg.rows, iAlignUp(fimg.cols, 128), false, NULL, (float *)fimg.data);
  cudaImg.Download();
  SiftData siftData;
  InitSiftData(siftData, 4096, true, true);
  ExtractSift(siftData, cudaImg, 5, 1.0f, 3.0f, 0.0f, true);
  CHECK(siftData.numPts == 0, "Scale-up extraction of a uniform image returns 0 features");
  FreeSiftData(siftData);
}

// Regression test: numOctaves >= 8 wrote past the per-octave arrays
// (Laplace kernels on the host stack and in constant memory, point counters), which corrupted
// the results of this call and of later calls. More than 7 octaves now gives the 7-octave result.
static int ExtractOctaves(SiftData &data, const cv::Mat &fimg, int numOctaves)
{
  CudaImage cudaImg;
  cudaImg.Allocate(fimg.cols, fimg.rows, iAlignUp(fimg.cols, 128), false, NULL, (float *)fimg.data);
  cudaImg.Download();
  ExtractSift(data, cudaImg, numOctaves, 1.0f, 3.0f, 0.0f, false);
  return data.numPts;
}

// Same positions and scales, in any order (the order of the points is not deterministic)
static bool SamePoints(const SiftData &a, const SiftData &b)
{
  if (a.numPts != b.numPts) return false;
  std::vector<std::vector<float> > pa, pb;
  for (int i = 0; i < a.numPts; i++) {
    pa.push_back({a.h_data[i].xpos, a.h_data[i].ypos, a.h_data[i].scale});
    pb.push_back({b.h_data[i].xpos, b.h_data[i].ypos, b.h_data[i].scale});
  }
  std::sort(pa.begin(), pa.end());
  std::sort(pb.begin(), pb.end());
  return pa == pb;
}

void TestTooManyOctaves()
{
  std::cout << "\n--- Test: More than 7 octaves ---" << std::endl;

  cv::Mat img = cv::imread("data/img1.png", 0);
  if (img.empty()) return;
  cv::Mat big, fbig, fimg;
  cv::resize(img, big, cv::Size(4000, 3000));
  big.convertTo(fbig, CV_32FC1);
  img.convertTo(fimg, CV_32FC1);

  SiftData data7, data8;
  InitSiftData(data7, 32768, true, true);
  InitSiftData(data8, 32768, true, true);
  ExtractOctaves(data7, fbig, 7);
  ExtractOctaves(data8, fbig, 8);
  std::cout << "    4000x3000: " << data7.numPts << " features with 7 octaves, " << data8.numPts << " with 8" << std::endl;
  CHECK(data7.numPts > 0 && SamePoints(data7, data8), "8 octaves give the 7-octave features (4000x3000)");
  ExtractOctaves(data8, fimg, 8);
  ExtractOctaves(data7, fimg, 7);
  std::cout << "    " << fimg.cols << "x" << fimg.rows << ": " << data7.numPts << " features with 7 octaves, " << data8.numPts << " with 8" << std::endl;
  CHECK(data7.numPts > 0 && SamePoints(data7, data8), "8 octaves give the 7-octave features after a larger image");
  FreeSiftData(data7);
  FreeSiftData(data8);
}

// Regression test: when the height of an octave is not a multiple of 16,
// ScaleDown wrote up to 8 rows past the next octave. At 1280x272 with 6 octaves, the writes
// of the last octave went 40 floats past the end of the temporary memory. The results are
// unchanged: run this under compute-sanitizer --tool memcheck to see the out-of-bounds write.
void TestScaleDownLastRows()
{
  std::cout << "\n--- Test: Octave heights not multiple of 16 (1280x272, 6 octaves) ---" << std::endl;

  cv::Mat img = cv::imread("data/img1.png", 0);
  if (img.empty()) return;
  cv::Mat small, fimg;
  cv::resize(img, small, cv::Size(1280, 272));
  small.convertTo(fimg, CV_32FC1);

  CudaImage cudaImg;
  cudaImg.Allocate(fimg.cols, fimg.rows, iAlignUp(fimg.cols, 128), false, NULL, (float *)fimg.data);
  cudaImg.Download();
  SiftData siftData;
  InitSiftData(siftData, 32768, true, true);
  float *mem = AllocSiftTempMemory(fimg.cols, fimg.rows, 6, false);
  ExtractSift(siftData, cudaImg, 6, 1.0f, 3.0f, 0.0f, false, mem);
  FreeSiftTempMemory(mem);
  CHECK(siftData.numPts > 0, "Features detected at 1280x272 with 6 octaves");
  FreeSiftData(siftData);
}

// Regression test: with min(width, height) < 2^(numOctaves-1), the
// smallest octaves were empty, their kernel launch had a zero grid, and the process exited.
// The number of octaves is now reduced. Run under compute-sanitizer --tool memcheck: the
// 1920x3 and 2x2 cases wrote past the temporary memory without the ScaleDown last-row fix.
void TestSmallImages()
{
  std::cout << "\n--- Test: Images too small for 5 octaves ---" << std::endl;

  cv::Mat img = cv::imread("data/img1.png", 0);
  if (img.empty()) return;
  const int sizes[][2] = { {30, 12}, {1920, 10}, {1920, 3}, {2, 2} };
  SiftData siftData;
  InitSiftData(siftData, 4096, true, true);
  int done = 0;
  for (auto &sz : sizes) {
    cv::Mat small, fimg;
    cv::resize(img, small, cv::Size(sz[0], sz[1]), 0, 0, cv::INTER_AREA);
    small.convertTo(fimg, CV_32FC1);
    CudaImage cudaImg;
    cudaImg.Allocate(fimg.cols, fimg.rows, iAlignUp(fimg.cols, 128), false, NULL, (float *)fimg.data);
    cudaImg.Download();
    float *mem = AllocSiftTempMemory(fimg.cols, fimg.rows, 5, false);
    ExtractSift(siftData, cudaImg, 5, 1.0f, 3.0f, 0.0f, false, mem);
    FreeSiftTempMemory(mem);
    std::cout << "    " << sz[0] << "x" << sz[1] << ": " << siftData.numPts << " features" << std::endl;
    done += (siftData.numPts >= 0 && siftData.numPts <= siftData.maxPts);
  }
  CHECK(done == 4, "Extraction with 5 octaves returns on 30x12, 1920x10, 1920x3 and 2x2 images");
  FreeSiftData(siftData);
}

// Regression test: LowPass read the source image with the pitch of the
// destination, so an input image whose pitch differs from the internal one gave garbage features
void TestUnpaddedInput()
{
  std::cout << "\n--- Test: Input image with pitch == width ---" << std::endl;

  cv::Mat img = cv::imread("data/img1.png", 0);
  if (img.empty()) return;
  cv::Mat small, fimg;
  cv::resize(img, small, cv::Size(1000, 700), 0, 0, cv::INTER_AREA);   // width not a multiple of 128
  small.convertTo(fimg, CV_32FC1);
  const int w = fimg.cols, h = fimg.rows;

  SiftData data1, data2;
  InitSiftData(data1, 32768, true, true);
  InitSiftData(data2, 32768, true, true);
  float *d_unpadded = NULL;
  cudaMalloc((void **)&d_unpadded, sizeof(float)*w*h);
  {
    CudaImage padded, unpadded;
    padded.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float *)fimg.data);
    padded.Download();
    unpadded.Allocate(w, h, w, false, d_unpadded, (float *)fimg.data);
    unpadded.Download();
    ExtractSift(data1, padded, 5, 1.0f, 3.0f, 0.0f, false);
    ExtractSift(data2, unpadded, 5, 1.0f, 3.0f, 0.0f, false);
  }
  cudaFree(d_unpadded);

  int same = 0;
  for (int i = 0; i < data1.numPts; i++) {
    SiftPoint &p1 = data1.h_data[i];
    for (int j = 0; j < data2.numPts; j++) {
      SiftPoint &p2 = data2.h_data[j];
      if (fabsf(p1.xpos - p2.xpos) < 0.01f && fabsf(p1.ypos - p2.ypos) < 0.01f && fabsf(p1.scale - p2.scale) < 0.01f) {
        same++;
        break;
      }
    }
  }
  std::cout << "    pitch " << iAlignUp(w, 128) << ": " << data1.numPts << " features, pitch " << w << ": "
            << data2.numPts << " features, " << same << " in common" << std::endl;
  CHECK(data1.numPts > 0 && data2.numPts == data1.numPts && same == data1.numPts, "Same features whatever the input pitch");

  FreeSiftData(data1);
  FreeSiftData(data2);
}

// Regression test: CopyToTexture(dst, true) copied pitch*height floats
// from the host image, whose rows are width floats, which sheared the rows of the texture
void TestCopyToTextureFromHost()
{
  std::cout << "\n--- Test: CopyToTexture from a host image ---" << std::endl;

  const int w = 1000, h = 700, p = iAlignUp(w, 128);   // width not a multiple of 128
  std::vector<float> host(p*h, -1.0f);                 // rows of width floats, then unused space
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
      host[y*w + x] = (float)((x + 3*y)%251);

  CudaImage img;
  img.Allocate(w, h, p, false, NULL, host.data());
  img.InitTexture();
  img.CopyToTexture(img, true);
  std::vector<float> back(p*h, 0.0f);
  cudaMemcpy2DFromArray(back.data(), sizeof(float)*w, (cudaArray *)img.t_data, 0, 0, sizeof(float)*w, h, cudaMemcpyDeviceToHost);
  int wrong = 0;
  for (int i = 0; i < w*h; i++)
    wrong += (back[i] != host[i]);
  std::cout << "    " << wrong << " of " << w*h << " texture values differ from the image" << std::endl;
  CHECK(wrong == 0, "Texture holds the host image");

  // The device copy (rows of pitch floats) must give the same texture: clear the
  // texture with a copy from a zero image, then copy the image from the device
  std::vector<float> zeros(p*h, 0.0f);
  CudaImage zero;
  zero.Allocate(w, h, p, false, NULL, zeros.data());
  zero.CopyToTexture(img, true);
  img.Download();
  img.CopyToTexture(img, false);
  cudaMemcpy2DFromArray(back.data(), sizeof(float)*w, (cudaArray *)img.t_data, 0, 0, sizeof(float)*w, h, cudaMemcpyDeviceToHost);
  wrong = 0;
  for (int i = 0; i < w*h; i++)
    wrong += (back[i] != host[i]);
  std::cout << "    " << wrong << " of " << w*h << " texture values differ after a device copy" << std::endl;
  CHECK(wrong == 0, "Texture holds the device image");
}

// Regression test: a second Allocate over a user buffer kept the ownership flag
// of the first, internal allocation, so the destructor freed the user buffer; FreeSiftData left
// h_data dangling, so a second FreeSiftData freed it twice
void TestImageOwnership()
{
  std::cout << "\n--- Test: CudaImage and SiftData ownership ---" << std::endl;

  const int w = 1000, h = 700;
  float *d_user = NULL;
  cudaMalloc((void **)&d_user, sizeof(float)*w*h);
  {
    CudaImage img;
    img.Allocate(w, h, iAlignUp(w, 128), false);
    img.Allocate(w, h, w, false, d_user);
  }
  cudaError_t err = cudaFree(d_user);   // fails if the destructor already freed it
  cudaGetLastError();
  CHECK(err == cudaSuccess, "CudaImage does not free a user buffer given to a second Allocate");

  SiftData data;
  InitSiftData(data, 1024, true, true);
  FreeSiftData(data);
  bool reset = (data.h_data == NULL && data.d_data == NULL);
  if (reset)
    FreeSiftData(data);   // must be a no-op
  CHECK(reset, "FreeSiftData resets its pointers, so a second call is safe");
}

// Regression test: after a switch to another GPU, LowPass and
// ScaleDown skipped the upload of their coefficients to the constant memory of the new device,
// and ExtractSift found no features there. Needs 2 GPUs: skipped otherwise.
void TestSecondDevice()
{
  std::cout << "\n--- Test: Extraction on a second GPU ---" << std::endl;

  int numDevices = 0, device0 = 0;
  if (cudaGetDeviceCount(&numDevices) != cudaSuccess || numDevices < 2) {
    std::cout << "    " << numDevices << " GPU found" << std::endl;
    SKIP("Same number of features on a second GPU (needs 2 GPUs)");
    return;
  }
  cv::Mat img = cv::imread("data/img1.png", 0);
  if (img.empty()) return;
  cv::Mat fimg;
  img.convertTo(fimg, CV_32FC1);

  cudaGetDevice(&device0);
  int numPts[2];
  for (int k = 0; k < 2; k++) {
    cudaSetDevice(k == 0 ? device0 : (device0 + 1) % numDevices);
    CudaImage cudaImg;
    cudaImg.Allocate(fimg.cols, fimg.rows, iAlignUp(fimg.cols, 128), false, NULL, (float *)fimg.data);
    cudaImg.Download();
    SiftData siftData;
    InitSiftData(siftData, 32768, true, true);
    ExtractSift(siftData, cudaImg, 5, 1.0f, 3.0f, 0.0f, false);
    numPts[k] = siftData.numPts;
    FreeSiftData(siftData);
  }
  cudaSetDevice(device0);
  std::cout << "    " << numPts[0] << " features on the first GPU, " << numPts[1] << " on the second" << std::endl;
  // The bug gives 0 features; different GPU models may round slightly differently
  CHECK(numPts[0] > 0 && std::abs(numPts[1] - numPts[0]) <= numPts[0]/50, "Same number of features on a second GPU (within 2%)");
}

// Regression test: concurrent ExtractSift calls on the same device shared the point counters,
// d_MaxNumPoints and the filter kernels in constant memory, so they returned wrong points
void TestConcurrentExtraction()
{
  std::cout << "\n--- Test: Concurrent extraction from several threads ---" << std::endl;

  cv::Mat img = cv::imread("data/img1.png", 0);
  if (img.empty()) return;
  cv::Mat fimg;
  img.convertTo(fimg, CV_32FC1);
  CudaImage cudaImg;
  cudaImg.Allocate(fimg.cols, fimg.rows, iAlignUp(fimg.cols, 128), false, NULL, (float *)fimg.data);
  cudaImg.Download();

  // Two settings, so that the threads also need different filter kernels
  const int numOctaves[2] = {5, 4};
  const float initBlur[2] = {1.0f, 0.5f};
  auto positions = [](SiftData &data) {
    std::vector<std::pair<float, float>> pos;
    for (int i = 0; i < data.numPts; i++)
      pos.push_back({data.h_data[i].ypos, data.h_data[i].xpos});
    std::sort(pos.begin(), pos.end());
    return pos;
  };
  std::vector<std::pair<float, float>> ref[2];
  for (int k = 0; k < 2; k++) {
    SiftData data;
    InitSiftData(data, 32768, true, true);
    ExtractSift(data, cudaImg, numOctaves[k], initBlur[k], 3.0f);
    ref[k] = positions(data);
    FreeSiftData(data);
  }

  const int numThreads = 4, numRuns = 10;
  int device;
  cudaGetDevice(&device);
  int wrong[numThreads] = {0};
  std::vector<std::thread> threads;
  for (int t = 0; t < numThreads; t++)
    threads.emplace_back([&, t]() {
      cudaSetDevice(device);
      SiftData data;
      InitSiftData(data, 32768, true, true);
      for (int r = 0; r < numRuns; r++) {
        ExtractSift(data, cudaImg, numOctaves[t%2], initBlur[t%2], 3.0f);
        wrong[t] += (positions(data) != ref[t%2]);
      }
      FreeSiftData(data);
    });
  for (auto &th : threads)
    th.join();
  int numWrong = 0;
  for (int t = 0; t < numThreads; t++)
    numWrong += wrong[t];
  std::cout << "    " << numThreads << " threads x " << numRuns << " extractions: " << numWrong
            << " with other points than a single-threaded extraction" << std::endl;
  CHECK(numWrong == 0, "Concurrent extractions give the same points as single-threaded ones");
}

int main(int argc, char **argv)
{
  int devNum = 0;
  if (argc > 1) devNum = std::atoi(argv[1]);

  std::cout << "=============================================" << std::endl;
  std::cout << "   CudaSift Extraction Tests" << std::endl;
  std::cout << "=============================================" << std::endl;

  InitCuda(devNum);

  TestBasicExtraction();
  TestDifferentThresholds();
  TestDifferentOctaves();
  TestReproducibility();
  TestSubpixelRefinement();
  TestDescriptorAnglePi();
  TestZeroGradientDescriptors();
  TestScaleUp();
  TestScaleDownLastRows();
  TestUnpaddedInput();
  TestCopyToTextureFromHost();
  TestImageOwnership();
  TestSecondDevice();
  TestConcurrentExtraction();
  // Last: without their fixes, these end the process (exit or stack overflow)
  TestScaleUpNoPoints();
  TestTooManyOctaves();
  TestSmallImages();

  std::cout << "\n=============================================" << std::endl;
  std::cout << "  Results: " << testsPassed << " passed, " << testsFailed << " failed";
  if (testsSkipped > 0)
    std::cout << ", " << testsSkipped << " skipped";
  std::cout << std::endl;
  std::cout << "=============================================" << std::endl;

  return testsFailed > 0 ? 1 : 0;
}
