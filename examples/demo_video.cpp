//********************************************************//
// Demo: Real-time SIFT on Video / Webcam                 //
// Extracts SIFT features in real-time from video input   //
//********************************************************//

#include <iostream>
#include <iomanip>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>

#include "cudaImage.h"
#include "cudaSift.h"

int main(int argc, char **argv)
{
  int devNum = 0;
  std::string source = "0"; // default: webcam
  float thresh = 5.0f;

  if (argc > 1) source = argv[1];
  if (argc > 2) devNum = std::atoi(argv[2]);
  if (argc > 3) thresh = std::atof(argv[3]);

  std::cout << "=== CUDA SIFT Real-time Video Demo ===" << std::endl;

  // Open video source
  cv::VideoCapture cap;
  if (source == "0" || source == "1" || source == "2")
    cap.open(std::atoi(source.c_str()));
  else
    cap.open(source);

  if (!cap.isOpened()) {
    std::cerr << "Error: Cannot open video source: " << source << std::endl;
    return -1;
  }

  int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  double fps = cap.get(cv::CAP_PROP_FPS);
  std::cout << "Video: " << w << "x" << h << " @ " << fps << " fps" << std::endl;
  std::cout << "Threshold: " << thresh << std::endl;
  std::cout << "Press 'q' to quit, '+'/'-' to adjust threshold" << std::endl;

  // Initialize CUDA
  InitCuda(devNum);

  // Pre-allocate SIFT data
  SiftData siftData;
  InitSiftData(siftData, 32768, true, true);

  int numOctaves = 5;
  float *memoryTmp = AllocSiftTempMemory(w, h, numOctaves, false);

  cv::Mat frame, gray, fimg;
  int frameCount = 0;
  double totalTime = 0.0;

  while (true) {
    cap >> frame;
    if (frame.empty()) break;

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(fimg, CV_32FC1);

    // Upload to GPU
    CudaImage cudaImg;
    cudaImg.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float *)fimg.data);
    cudaImg.Download();

    // Extract features
    auto t0 = cv::getTickCount();
    ExtractSift(siftData, cudaImg, numOctaves, 1.0f, thresh, 0.0f, false, memoryTmp);
    auto t1 = cv::getTickCount();
    double elapsed = (t1 - t0) / cv::getTickFrequency() * 1000.0;
    totalTime += elapsed;
    frameCount++;

    // Draw keypoints
    for (int i = 0; i < siftData.numPts; i++) {
      int x = (int)(siftData.h_data[i].xpos + 0.5f);
      int y = (int)(siftData.h_data[i].ypos + 0.5f);
      float scale = siftData.h_data[i].scale;
      int r = std::max(2, (int)(scale * 1.5f));
      cv::circle(frame, cv::Point(x, y), r, cv::Scalar(0, 255, 0), 1);
    }

    // Draw stats
    char buf[256];
    snprintf(buf, sizeof(buf), "Features: %d | SIFT: %.1f ms | Thresh: %.1f",
             siftData.numPts, elapsed, thresh);
    cv::putText(frame, buf, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 255), 2);

    cv::imshow("CUDA SIFT Real-time", frame);
    int key = cv::waitKey(1);
    if (key == 'q' || key == 27) break;
    if (key == '+' || key == '=') thresh += 0.5f;
    if (key == '-' && thresh > 0.5f) thresh -= 0.5f;
  }

  std::cout << "\n=== Statistics ===" << std::endl;
  std::cout << "Total frames: " << frameCount << std::endl;
  if (frameCount > 0)
    std::cout << "Avg SIFT time: " << std::fixed << std::setprecision(2)
              << totalTime / frameCount << " ms" << std::endl;

  FreeSiftTempMemory(memoryTmp);
  FreeSiftData(siftData);
  cv::destroyAllWindows();
  return 0;
}
