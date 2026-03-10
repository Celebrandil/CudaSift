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
  TestScaleUp();

  std::cout << "\n=============================================" << std::endl;
  std::cout << "  Results: " << testsPassed << " passed, " << testsFailed << " failed" << std::endl;
  std::cout << "=============================================" << std::endl;

  return testsFailed > 0 ? 1 : 0;
}
