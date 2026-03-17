//********************************************************//
// Demo: SIFT Feature Extraction                          //
// Extracts and displays SIFT features from a single image //
//********************************************************//

#include <iostream>
#include <iomanip>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cudaImage.h"
#include "cudaSift.h"

void DrawKeypoints(cv::Mat &img, SiftData &siftData)
{
  SiftPoint *pts = siftData.h_data;
  for (int i = 0; i < siftData.numPts; i++) {
    int x = (int)(pts[i].xpos + 0.5f);
    int y = (int)(pts[i].ypos + 0.5f);
    float scale = pts[i].scale;
    int r = std::max(2, (int)(scale * 1.41f));
    cv::circle(img, cv::Point(x, y), r, cv::Scalar(0, 255, 0), 1);
    float angle = pts[i].orientation * 3.14159f / 180.0f;
    int dx = (int)(r * cosf(angle));
    int dy = (int)(r * sinf(angle));
    cv::line(img, cv::Point(x, y), cv::Point(x + dx, y + dy), cv::Scalar(0, 0, 255), 1);
  }
}

int main(int argc, char **argv)
{
  std::string imagePath = "data/img1.png";
  int devNum = 0;
  float thresh = 3.0f;
  int numOctaves = 5;

  if (argc > 1) imagePath = argv[1];
  if (argc > 2) devNum = std::atoi(argv[2]);
  if (argc > 3) thresh = std::atof(argv[3]);
  if (argc > 4) numOctaves = std::atoi(argv[4]);

  // Load image
  cv::Mat origImg = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
  if (origImg.empty()) {
    std::cerr << "Error: Cannot load image: " << imagePath << std::endl;
    return -1;
  }

  cv::Mat fimg;
  origImg.convertTo(fimg, CV_32FC1);
  int w = fimg.cols;
  int h = fimg.rows;

  std::cout << "=== CUDA SIFT Feature Extraction Demo ===" << std::endl;
  std::cout << "Image: " << imagePath << " (" << w << "x" << h << ")" << std::endl;
  std::cout << "Threshold: " << thresh << ", Octaves: " << numOctaves << std::endl;

  // Initialize CUDA
  InitCuda(devNum);

  // Upload image to GPU
  CudaImage cudaImg;
  cudaImg.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float *)fimg.data);
  cudaImg.Download();

  // Extract SIFT features
  SiftData siftData;
  InitSiftData(siftData, 32768, true, true);
  float initBlur = 1.0f;

  float *memoryTmp = AllocSiftTempMemory(w, h, numOctaves, false);
  ExtractSift(siftData, cudaImg, numOctaves, initBlur, thresh, 0.0f, false, memoryTmp);
  FreeSiftTempMemory(memoryTmp);

  std::cout << "\n=== Results ===" << std::endl;
  std::cout << "Features detected: " << siftData.numPts << std::endl;

  // Print feature statistics
  if (siftData.numPts > 0) {
    float minScale = 1e10f, maxScale = 0.0f;
    float avgScale = 0.0f;
    for (int i = 0; i < siftData.numPts; i++) {
      float s = siftData.h_data[i].scale;
      minScale = std::min(minScale, s);
      maxScale = std::max(maxScale, s);
      avgScale += s;
    }
    avgScale /= siftData.numPts;
    std::cout << "Scale range: [" << std::fixed << std::setprecision(2)
              << minScale << ", " << maxScale << "], avg: " << avgScale << std::endl;
  }

  // Draw keypoints and save
  cv::Mat colorImg;
  cv::cvtColor(origImg, colorImg, cv::COLOR_GRAY2BGR);
  DrawKeypoints(colorImg, siftData);
  cv::imwrite("data/keypoints.png", colorImg);
  std::cout << "Keypoints saved to data/keypoints.png" << std::endl;

  // Print first 10 features
  std::cout << "\nFirst 10 features:" << std::endl;
  std::cout << std::setw(5) << "ID" << std::setw(10) << "X" << std::setw(10) << "Y"
            << std::setw(10) << "Scale" << std::setw(10) << "Orient" << std::endl;
  for (int i = 0; i < std::min(10, siftData.numPts); i++) {
    SiftPoint &pt = siftData.h_data[i];
    std::cout << std::setw(5) << i
              << std::setw(10) << std::fixed << std::setprecision(1) << pt.xpos
              << std::setw(10) << pt.ypos
              << std::setw(10) << std::setprecision(2) << pt.scale
              << std::setw(10) << std::setprecision(1) << pt.orientation << std::endl;
  }

  FreeSiftData(siftData);
  std::cout << "\nDone!" << std::endl;
  return 0;
}
