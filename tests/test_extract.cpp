//********************************************************//
// Test: SIFT Feature Extraction Correctness              //
//********************************************************//

#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cudaImage.h"
#include "cudaSift.h"

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

  std::cout << "\n=============================================" << std::endl;
  std::cout << "  Results: " << testsPassed << " passed, " << testsFailed << " failed" << std::endl;
  std::cout << "=============================================" << std::endl;

  return testsFailed > 0 ? 1 : 0;
}
