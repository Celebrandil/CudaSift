//********************************************************//
// Test: SIFT Feature Matching Correctness                //
//********************************************************//

#include <iostream>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cuda_runtime.h>
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

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);

void TestSelfMatch()
{
  std::cout << "\n--- Test: Self-Matching ---" << std::endl;

  cv::Mat img = cv::imread("data/img1.png", 0);
  if (img.empty()) return;
  cv::Mat fimg;
  img.convertTo(fimg, CV_32FC1);

  CudaImage cudaImg;
  cudaImg.Allocate(fimg.cols, fimg.rows, iAlignUp(fimg.cols, 128), false, NULL, (float *)fimg.data);
  cudaImg.Download();

  SiftData siftData1, siftData2;
  InitSiftData(siftData1, 32768, true, true);
  InitSiftData(siftData2, 32768, true, true);

  float *mem = AllocSiftTempMemory(fimg.cols, fimg.rows, 5, false);
  ExtractSift(siftData1, cudaImg, 5, 1.0f, 3.0f, 0.0f, false, mem);
  ExtractSift(siftData2, cudaImg, 5, 1.0f, 3.0f, 0.0f, false, mem);
  FreeSiftTempMemory(mem);

  MatchSiftData(siftData1, siftData2);

  // Self-match should produce perfect matches
  int perfectMatches = 0;
  int highScoreMatches = 0;
  for (int i = 0; i < siftData1.numPts; i++) {
    if (siftData1.h_data[i].match == i)
      perfectMatches++;
    if (siftData1.h_data[i].score > 0.95f)
      highScoreMatches++;
  }

  std::cout << "    Perfect matches (index match): " << perfectMatches << "/" << siftData1.numPts << std::endl;
  std::cout << "    High score matches (>0.95): " << highScoreMatches << "/" << siftData1.numPts << std::endl;

  float perfectRatio = (float)perfectMatches / siftData1.numPts;
  float highScoreRatio = (float)highScoreMatches / siftData1.numPts;
  CHECK(highScoreRatio > 0.95f, "Self-match: >95% high score matches (>0.95)");
  CHECK(highScoreRatio > 0.80f, "Self-match: >80% high score matches (sanity check)");

  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
}

void TestCrossMatch()
{
  std::cout << "\n--- Test: Cross-Image Matching ---" << std::endl;

  cv::Mat limg = cv::imread("data/img1.png", 0);
  cv::Mat rimg = cv::imread("data/img2.png", 0);
  if (limg.empty() || rimg.empty()) return;

  cv::Mat fimg1, fimg2;
  limg.convertTo(fimg1, CV_32FC1);
  rimg.convertTo(fimg2, CV_32FC1);

  CudaImage cudaImg1, cudaImg2;
  cudaImg1.Allocate(fimg1.cols, fimg1.rows, iAlignUp(fimg1.cols, 128), false, NULL, (float *)fimg1.data);
  cudaImg2.Allocate(fimg2.cols, fimg2.rows, iAlignUp(fimg2.cols, 128), false, NULL, (float *)fimg2.data);
  cudaImg1.Download();
  cudaImg2.Download();

  SiftData siftData1, siftData2;
  InitSiftData(siftData1, 32768, true, true);
  InitSiftData(siftData2, 32768, true, true);

  float *mem = AllocSiftTempMemory(fimg1.cols, fimg1.rows, 5, false);
  ExtractSift(siftData1, cudaImg1, 5, 1.0f, 3.0f, 0.0f, false, mem);
  ExtractSift(siftData2, cudaImg2, 5, 1.0f, 3.0f, 0.0f, false, mem);
  FreeSiftTempMemory(mem);

  std::cout << "    Features: " << siftData1.numPts << " vs " << siftData2.numPts << std::endl;

  MatchSiftData(siftData1, siftData2);

  // Check match quality
  int goodMatches = 0;
  int validMatches = 0;
  for (int i = 0; i < siftData1.numPts; i++) {
    if (siftData1.h_data[i].score > 0.7f)
      goodMatches++;
    if (siftData1.h_data[i].match >= 0 && siftData1.h_data[i].match < siftData2.numPts)
      validMatches++;
  }

  std::cout << "    Valid matches: " << validMatches << std::endl;
  std::cout << "    Good matches (score > 0.7): " << goodMatches << std::endl;

  CHECK(validMatches == siftData1.numPts, "All matches have valid indices");
  CHECK(goodMatches > 50, "At least 50 good matches found");

  // Test ambiguity values
  bool ambiguityOk = true;
  for (int i = 0; i < siftData1.numPts; i++) {
    float amb = siftData1.h_data[i].ambiguity;
    if (amb < 0.0f || amb > 1.0f) {
      ambiguityOk = false;
      break;
    }
  }
  CHECK(ambiguityOk, "All ambiguity values in [0, 1]");

  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
}

void TestHomography()
{
  std::cout << "\n--- Test: Homography Estimation ---" << std::endl;

  cv::Mat limg = cv::imread("data/img1.png", 0);
  cv::Mat rimg = cv::imread("data/img2.png", 0);
  if (limg.empty() || rimg.empty()) return;

  cv::Mat fimg1, fimg2;
  limg.convertTo(fimg1, CV_32FC1);
  rimg.convertTo(fimg2, CV_32FC1);

  CudaImage cudaImg1, cudaImg2;
  cudaImg1.Allocate(fimg1.cols, fimg1.rows, iAlignUp(fimg1.cols, 128), false, NULL, (float *)fimg1.data);
  cudaImg2.Allocate(fimg2.cols, fimg2.rows, iAlignUp(fimg2.cols, 128), false, NULL, (float *)fimg2.data);
  cudaImg1.Download();
  cudaImg2.Download();

  SiftData siftData1, siftData2;
  InitSiftData(siftData1, 32768, true, true);
  InitSiftData(siftData2, 32768, true, true);

  float *mem = AllocSiftTempMemory(fimg1.cols, fimg1.rows, 5, false);
  ExtractSift(siftData1, cudaImg1, 5, 1.0f, 3.0f, 0.0f, false, mem);
  ExtractSift(siftData2, cudaImg2, 5, 1.0f, 3.0f, 0.0f, false, mem);
  FreeSiftTempMemory(mem);

  MatchSiftData(siftData1, siftData2);

  float homography[9];
  int numMatches;
  FindHomography(siftData1, homography, &numMatches, 10000, 0.00f, 0.80f, 5.0f);
  int numFit = ImproveHomography(siftData1, homography, 5, 0.00f, 0.80f, 3.0f);

  std::cout << "    RANSAC matches: " << numMatches << std::endl;
  std::cout << "    Refined inliers: " << numFit << std::endl;

  CHECK(numMatches > 20, "RANSAC found > 20 matches");
  CHECK(numFit > 10, "Refined homography has > 10 inliers");

  // Check homography is not identity
  bool notIdentity = false;
  for (int i = 0; i < 8; i++) {
    if (i == 0 || i == 4) {
      if (fabs(homography[i] - 1.0f) > 0.01f)
        notIdentity = true;
    } else if (i < 8) {
      if (fabs(homography[i]) > 0.01f)
        notIdentity = true;
    }
  }
  CHECK(notIdentity, "Homography is not identity (images differ)");

  // Check that match_error was populated
  int lowErrorCount = 0;
  for (int i = 0; i < siftData1.numPts; i++) {
    if (siftData1.h_data[i].match_error < 5.0f)
      lowErrorCount++;
  }
  CHECK(lowErrorCount > 0, "Some matches have low reprojection error");

  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
}

void TestMatchSpeed()
{
  std::cout << "\n--- Test: Matching Speed ---" << std::endl;

  cv::Mat img = cv::imread("data/img1.png", 0);
  if (img.empty()) return;
  cv::Mat fimg;
  img.convertTo(fimg, CV_32FC1);

  CudaImage cudaImg;
  cudaImg.Allocate(fimg.cols, fimg.rows, iAlignUp(fimg.cols, 128), false, NULL, (float *)fimg.data);
  cudaImg.Download();

  SiftData siftData1, siftData2;
  InitSiftData(siftData1, 32768, true, true);
  InitSiftData(siftData2, 32768, true, true);

  float *mem = AllocSiftTempMemory(fimg.cols, fimg.rows, 5, false);
  ExtractSift(siftData1, cudaImg, 5, 1.0f, 3.0f, 0.0f, false, mem);
  ExtractSift(siftData2, cudaImg, 5, 1.0f, 3.0f, 0.0f, false, mem);
  FreeSiftTempMemory(mem);

  // Warmup
  for (int i = 0; i < 10; i++)
    MatchSiftData(siftData1, siftData2);

  // Measure
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  int numRuns = 100;
  for (int i = 0; i < numRuns; i++)
    MatchSiftData(siftData1, siftData2);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  float avgMs = ms / numRuns;

  std::cout << "    Features: " << siftData1.numPts << " x " << siftData2.numPts << std::endl;
  std::cout << "    Avg matching time: " << avgMs << " ms" << std::endl;

  CHECK(avgMs < 5.0f, "Matching completes in < 5 ms");
  CHECK(avgMs < 2.0f, "Matching completes in < 2 ms (optimal)");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
}

int main(int argc, char **argv)
{
  int devNum = 0;
  if (argc > 1) devNum = std::atoi(argv[1]);

  std::cout << "=============================================" << std::endl;
  std::cout << "   CudaSift Matching Tests" << std::endl;
  std::cout << "=============================================" << std::endl;

  InitCuda(devNum);

  TestSelfMatch();
  TestCrossMatch();
  TestHomography();
  TestMatchSpeed();

  std::cout << "\n=============================================" << std::endl;
  std::cout << "  Results: " << testsPassed << " passed, " << testsFailed << " failed" << std::endl;
  std::cout << "=============================================" << std::endl;

  return testsFailed > 0 ? 1 : 0;
}
