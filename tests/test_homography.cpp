//********************************************************//
// Test: Homography Estimation with Synthetic Data        //
//********************************************************//

#include <iostream>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cudaImage.h"
#include "cudaSift.h"

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);

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

void TestTranslation()
{
  std::cout << "\n--- Test: Translation Detection ---" << std::endl;

  cv::Mat img = cv::imread("data/img1.png", 0);
  if (img.empty()) return;

  // Create translated version
  cv::Mat translated;
  cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, 30, 0, 1, 20);
  cv::warpAffine(img, translated, M, img.size());

  cv::Mat fimg1, fimg2;
  img.convertTo(fimg1, CV_32FC1);
  translated.convertTo(fimg2, CV_32FC1);

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
  ImproveHomography(siftData1, homography, 5, 0.00f, 0.80f, 3.0f);

  // Check that homography recovers the translation
  // For pure translation, h[0]~1, h[1]~0, h[2]~30, h[3]~0, h[4]~1, h[5]~20
  float h8 = homography[8];
  if (fabs(h8) > 0.001f) {
    for (int i = 0; i < 8; i++) homography[i] /= h8;
    homography[8] = 1.0f;
  }

  std::cout << "    Expected translation: (30, 20)" << std::endl;
  std::cout << "    Recovered h[2]=" << homography[2] << " h[5]=" << homography[5] << std::endl;
  std::cout << "    Matches: " << numMatches << std::endl;

  CHECK(numMatches > 20, "Found > 20 RANSAC matches for translation");
  CHECK(fabs(homography[2] - 30.0f) < 5.0f, "Translation X recovered within 5 pixels");
  CHECK(fabs(homography[5] - 20.0f) < 5.0f, "Translation Y recovered within 5 pixels");

  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
}

void TestRotation()
{
  std::cout << "\n--- Test: Rotation Detection ---" << std::endl;

  cv::Mat img = cv::imread("data/img1.png", 0);
  if (img.empty()) return;

  // Create rotated version (10 degrees)
  cv::Point2f center(img.cols / 2.0f, img.rows / 2.0f);
  cv::Mat rotMat = cv::getRotationMatrix2D(center, 10.0, 1.0);
  cv::Mat rotated;
  cv::warpAffine(img, rotated, rotMat, img.size());

  cv::Mat fimg1, fimg2;
  img.convertTo(fimg1, CV_32FC1);
  rotated.convertTo(fimg2, CV_32FC1);

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

  std::cout << "    Rotation: 10 degrees" << std::endl;
  std::cout << "    Matches found: " << numMatches << std::endl;

  CHECK(numMatches > 15, "Found > 15 RANSAC matches for rotation");

  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
}

void TestScale()
{
  std::cout << "\n--- Test: Scale Change Detection ---" << std::endl;

  cv::Mat img = cv::imread("data/img1.png", 0);
  if (img.empty()) return;

  // Create scaled version (80%)
  cv::Mat scaled;
  cv::resize(img, scaled, cv::Size(), 0.8, 0.8);

  // Pad back to original size
  cv::Mat padded = cv::Mat::zeros(img.size(), img.type());
  scaled.copyTo(padded(cv::Rect(0, 0, scaled.cols, scaled.rows)));

  cv::Mat fimg1, fimg2;
  img.convertTo(fimg1, CV_32FC1);
  padded.convertTo(fimg2, CV_32FC1);

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

  std::cout << "    Scale factor: 0.8" << std::endl;
  std::cout << "    Matches found: " << numMatches << std::endl;

  CHECK(numMatches > 10, "Found > 10 RANSAC matches for scale change");

  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
}

void TestPGMImages()
{
  std::cout << "\n--- Test: PGM Image Pair ---" << std::endl;

  cv::Mat limg, rimg;
  cv::imread("data/left.pgm", 0).convertTo(limg, CV_32FC1);
  cv::imread("data/righ.pgm", 0).convertTo(rimg, CV_32FC1);
  if (limg.empty() || rimg.empty()) {
    std::cout << "  [SKIP] PGM images not found" << std::endl;
    return;
  }

  CudaImage cudaImg1, cudaImg2;
  cudaImg1.Allocate(limg.cols, limg.rows, iAlignUp(limg.cols, 128), false, NULL, (float *)limg.data);
  cudaImg2.Allocate(rimg.cols, rimg.rows, iAlignUp(rimg.cols, 128), false, NULL, (float *)rimg.data);
  cudaImg1.Download();
  cudaImg2.Download();

  SiftData siftData1, siftData2;
  InitSiftData(siftData1, 32768, true, true);
  InitSiftData(siftData2, 32768, true, true);

  float *mem = AllocSiftTempMemory(limg.cols, limg.rows, 5, false);
  ExtractSift(siftData1, cudaImg1, 5, 1.0f, 4.5f, 0.0f, false, mem);
  ExtractSift(siftData2, cudaImg2, 5, 1.0f, 4.5f, 0.0f, false, mem);
  FreeSiftTempMemory(mem);

  std::cout << "    Left features: " << siftData1.numPts << std::endl;
  std::cout << "    Right features: " << siftData2.numPts << std::endl;

  MatchSiftData(siftData1, siftData2);

  float homography[9];
  int numMatches;
  FindHomography(siftData1, homography, &numMatches, 10000, 0.00f, 0.80f, 5.0f);
  int numFit = ImproveHomography(siftData1, homography, 5, 0.00f, 0.80f, 3.0f);

  std::cout << "    RANSAC matches: " << numMatches << std::endl;
  std::cout << "    Inliers: " << numFit << std::endl;

  CHECK(siftData1.numPts > 100, "Left image has > 100 features");
  CHECK(siftData2.numPts > 100, "Right image has > 100 features");
  CHECK(numMatches > 20, "Found > 20 matches between stereo pair");

  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
}

int main(int argc, char **argv)
{
  int devNum = 0;
  if (argc > 1) devNum = std::atoi(argv[1]);

  std::cout << "=============================================" << std::endl;
  std::cout << "   CudaSift Homography Tests" << std::endl;
  std::cout << "=============================================" << std::endl;

  InitCuda(devNum);

  TestTranslation();
  TestRotation();
  TestScale();
  TestPGMImages();

  std::cout << "\n=============================================" << std::endl;
  std::cout << "  Results: " << testsPassed << " passed, " << testsFailed << " failed" << std::endl;
  std::cout << "=============================================" << std::endl;

  return testsFailed > 0 ? 1 : 0;
}
