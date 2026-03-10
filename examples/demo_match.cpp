//********************************************************//
// Demo: SIFT Feature Matching & Homography               //
// Matches features between two images and draws results  //
//********************************************************//

#include <iostream>
#include <iomanip>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cudaImage.h"
#include "cudaSift.h"

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);

void DrawMatches(cv::Mat &result, const cv::Mat &img1, const cv::Mat &img2,
                 SiftData &siftData1, SiftData &siftData2)
{
  int w1 = img1.cols, h = img1.rows;
  result = cv::Mat(h, w1 + img2.cols, CV_8UC3);
  cv::Mat left(result, cv::Rect(0, 0, w1, h));
  cv::Mat right(result, cv::Rect(w1, 0, img2.cols, img2.rows));

  cv::Mat tmp1, tmp2;
  cv::cvtColor(img1, tmp1, cv::COLOR_GRAY2BGR);
  cv::cvtColor(img2, tmp2, cv::COLOR_GRAY2BGR);
  tmp1.copyTo(left);
  tmp2.copyTo(right);

  SiftPoint *pts1 = siftData1.h_data;
  SiftPoint *pts2 = siftData2.h_data;
  int matchCount = 0;
  for (int i = 0; i < siftData1.numPts; i++) {
    if (pts1[i].match_error < 5.0f && pts1[i].ambiguity < 0.95f) {
      int j = pts1[i].match;
      cv::Point p1((int)pts1[i].xpos, (int)pts1[i].ypos);
      cv::Point p2((int)pts2[j].xpos + w1, (int)pts2[j].ypos);
      cv::Scalar color(rand() % 200 + 55, rand() % 200 + 55, rand() % 200 + 55);
      cv::line(result, p1, p2, color, 1);
      cv::circle(result, p1, 3, cv::Scalar(0, 255, 0), -1);
      cv::circle(result, p2, 3, cv::Scalar(0, 0, 255), -1);
      matchCount++;
    }
  }
  std::cout << "Good matches drawn: " << matchCount << std::endl;
}

int main(int argc, char **argv)
{
  std::string img1Path = "data/img1.png";
  std::string img2Path = "data/img2.png";
  int devNum = 0;

  if (argc > 1) img1Path = argv[1];
  if (argc > 2) img2Path = argv[2];
  if (argc > 3) devNum = std::atoi(argv[3]);

  // Load images
  cv::Mat origImg1 = cv::imread(img1Path, cv::IMREAD_GRAYSCALE);
  cv::Mat origImg2 = cv::imread(img2Path, cv::IMREAD_GRAYSCALE);
  if (origImg1.empty() || origImg2.empty()) {
    std::cerr << "Error: Cannot load images" << std::endl;
    return -1;
  }

  cv::Mat fimg1, fimg2;
  origImg1.convertTo(fimg1, CV_32FC1);
  origImg2.convertTo(fimg2, CV_32FC1);

  std::cout << "=== CUDA SIFT Feature Matching Demo ===" << std::endl;
  std::cout << "Image 1: " << img1Path << " (" << fimg1.cols << "x" << fimg1.rows << ")" << std::endl;
  std::cout << "Image 2: " << img2Path << " (" << fimg2.cols << "x" << fimg2.rows << ")" << std::endl;

  // Initialize CUDA
  InitCuda(devNum);

  // Upload images
  CudaImage cudaImg1, cudaImg2;
  cudaImg1.Allocate(fimg1.cols, fimg1.rows, iAlignUp(fimg1.cols, 128), false, NULL, (float *)fimg1.data);
  cudaImg2.Allocate(fimg2.cols, fimg2.rows, iAlignUp(fimg2.cols, 128), false, NULL, (float *)fimg2.data);
  cudaImg1.Download();
  cudaImg2.Download();

  // Extract features
  SiftData siftData1, siftData2;
  InitSiftData(siftData1, 32768, true, true);
  InitSiftData(siftData2, 32768, true, true);

  float *memoryTmp = AllocSiftTempMemory(fimg1.cols, fimg1.rows, 5, false);
  ExtractSift(siftData1, cudaImg1, 5, 1.0f, 3.0f, 0.0f, false, memoryTmp);
  ExtractSift(siftData2, cudaImg2, 5, 1.0f, 3.0f, 0.0f, false, memoryTmp);
  FreeSiftTempMemory(memoryTmp);

  std::cout << "\n=== Extraction Results ===" << std::endl;
  std::cout << "Image 1 features: " << siftData1.numPts << std::endl;
  std::cout << "Image 2 features: " << siftData2.numPts << std::endl;

  // Match features
  MatchSiftData(siftData1, siftData2);

  // Find homography
  float homography[9];
  int numMatches;
  FindHomography(siftData1, homography, &numMatches, 10000, 0.00f, 0.80f, 5.0f);
  int numFit = ImproveHomography(siftData1, homography, 5, 0.00f, 0.80f, 3.0f);

  std::cout << "\n=== Matching Results ===" << std::endl;
  std::cout << "RANSAC matches: " << numMatches << std::endl;
  std::cout << "Inliers after refinement: " << numFit << std::endl;
  std::cout << "Inlier ratio: " << std::fixed << std::setprecision(1)
            << 100.0f * numFit / std::min(siftData1.numPts, siftData2.numPts) << "%" << std::endl;

  // Print homography matrix
  std::cout << "\nHomography matrix:" << std::endl;
  for (int i = 0; i < 3; i++) {
    std::cout << "  [";
    for (int j = 0; j < 3; j++) {
      std::cout << std::setw(12) << std::setprecision(6) << homography[i * 3 + j];
    }
    std::cout << " ]" << std::endl;
  }

  // Draw and save matches
  cv::Mat matchResult;
  DrawMatches(matchResult, origImg1, origImg2, siftData1, siftData2);
  cv::imwrite("data/matches.png", matchResult);
  std::cout << "\nMatch visualization saved to data/matches.png" << std::endl;

  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
  std::cout << "\nDone!" << std::endl;
  return 0;
}
