//********************************************************//
// CudaSift Performance Benchmark                         //
// Extraction, matching and homography estimation at     //
// multiple resolutions                                   //
//********************************************************//

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cuda_runtime.h>
#include "cudaImage.h"
#include "cudaSift.h"

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);

struct BenchResult {
  int width, height;
  int numFeatures;
  double extractTimeMs;
  double matchTimeMs;
  double uploadTimeMs;
  double homographyTimeMs;      // FindHomography + ImproveHomography
  double homographyGPUTimeMs;   // FindHomographyGPU + ImproveHomographyGPU
  double totalTimeMs;
};

// Run extraction benchmark at a given resolution
BenchResult RunBenchmark(const cv::Mat &srcImg, int targetW, int targetH,
                         int numOctaves, float thresh, int warmupRuns, int benchRuns)
{
  BenchResult result = {};
  result.width = targetW;
  result.height = targetH;

  // Resize image to target resolution
  cv::Mat resized, fimg;
  cv::resize(srcImg, resized, cv::Size(targetW, targetH));
  resized.convertTo(fimg, CV_32FC1);

  int w = fimg.cols;
  int h = fimg.rows;

  // Allocate
  CudaImage cudaImg;
  cudaImg.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float *)fimg.data);

  SiftData siftData;
  InitSiftData(siftData, 32768, true, true);
  float *memoryTmp = AllocSiftTempMemory(w, h, numOctaves, false);

  // Warmup
  cudaImg.Download();
  for (int i = 0; i < warmupRuns; i++) {
    ExtractSift(siftData, cudaImg, numOctaves, 1.0f, thresh, 0.0f, false, memoryTmp);
  }

  // Benchmark upload
  {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < benchRuns; i++)
      cudaImg.Download();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    result.uploadTimeMs = ms / benchRuns;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  // Benchmark extraction
  std::vector<double> times;
  for (int i = 0; i < benchRuns; i++) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    ExtractSift(siftData, cudaImg, numOctaves, 1.0f, thresh, 0.0f, false, memoryTmp);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    times.push_back(ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  // Sort and take median
  std::sort(times.begin(), times.end());
  result.extractTimeMs = times[times.size() / 2];
  result.numFeatures = siftData.numPts;

  // Benchmark matching (if we have features)
  if (siftData.numPts > 10) {
    std::vector<double> matchTimes;
    for (int i = 0; i < benchRuns; i++) {
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);
      MatchSiftData(siftData, siftData); // self-match for benchmarking
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float ms = 0;
      cudaEventElapsedTime(&ms, start, stop);
      matchTimes.push_back(ms);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
    }
    std::sort(matchTimes.begin(), matchTimes.end());
    result.matchTimeMs = matchTimes[matchTimes.size() / 2];

    // Benchmark homography estimation on the self-match (wall time, since part of
    // FindHomography and all of ImproveHomography run on the CPU)
    std::vector<double> homTimes, homGPUTimes;
    for (int i = 0; i < benchRuns; i++) {
      float homography[9];
      int numMatches, numFit;
      auto t0 = std::chrono::steady_clock::now();
      FindHomography(siftData, homography, &numMatches, 10000, 0.00f, 0.80f, 5.0f);
      ImproveHomography(siftData, homography, 5, 0.00f, 0.80f, 3.0f);
      auto t1 = std::chrono::steady_clock::now();
      FindHomographyGPU(siftData, homography, &numMatches, 10000, 0.00f, 0.80f, 5.0f, i);
      ImproveHomographyGPU(siftData, homography, &numFit, 5, 0.00f, 0.80f, 3.0f);
      auto t2 = std::chrono::steady_clock::now();
      homTimes.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      homGPUTimes.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
    std::sort(homTimes.begin(), homTimes.end());
    std::sort(homGPUTimes.begin(), homGPUTimes.end());
    result.homographyTimeMs = homTimes[homTimes.size() / 2];
    result.homographyGPUTimeMs = homGPUTimes[homGPUTimes.size() / 2];
  }

  result.totalTimeMs = result.uploadTimeMs + result.extractTimeMs + result.matchTimeMs;

  FreeSiftTempMemory(memoryTmp);
  FreeSiftData(siftData);
  return result;
}

int main(int argc, char **argv)
{
  int devNum = 0;
  int warmupRuns = 50;
  int benchRuns = 200;
  float thresh = 3.0f;
  int numOctaves = 5;

  if (argc > 1) devNum = std::atoi(argv[1]);
  if (argc > 2) benchRuns = std::atoi(argv[2]);
  if (argc > 3) thresh = std::atof(argv[3]);

  std::cout << "=============================================" << std::endl;
  std::cout << "   CudaSift Performance Benchmark" << std::endl;
  std::cout << "   Ada Lovelace (sm_89) Optimized" << std::endl;
  std::cout << "=============================================" << std::endl;

  // Initialize CUDA and print device info
  InitCuda(devNum);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, devNum);
  std::cout << "\nGPU: " << prop.name << std::endl;
  std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
  std::cout << "SM Count: " << prop.multiProcessorCount << std::endl;
  std::cout << "Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
  std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
  std::cout << "L2 Cache: " << prop.l2CacheSize / 1024 << " KB" << std::endl;
  std::cout << "Shared Mem/Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
  std::cout << "Max Threads/Block: " << prop.maxThreadsPerBlock << std::endl;

  std::cout << "\nBenchmark config: warmup=" << warmupRuns
            << " runs=" << benchRuns
            << " thresh=" << thresh
            << " octaves=" << numOctaves << std::endl;

  // Load test image
  cv::Mat srcImg = cv::imread("data/img1.png", cv::IMREAD_GRAYSCALE);
  if (srcImg.empty()) {
    std::cerr << "Error: Cannot load data/img1.png" << std::endl;
    return -1;
  }

  // Test resolutions
  struct Resolution { int w, h; const char *name; };
  std::vector<Resolution> resolutions = {
    {640, 480, "VGA"},
    {1280, 720, "720p"},
    {1280, 960, "SXGA"},
    {1920, 1080, "1080p"},
    {2560, 1440, "1440p"},
    {3840, 2160, "4K UHD"},
  };

  std::vector<BenchResult> results;

  std::cout << "\nRunning benchmarks..." << std::endl;
  std::cout << std::string(119, '-') << std::endl;
  std::cout << std::setw(10) << "Resolution"
            << std::setw(12) << "Size"
            << std::setw(10) << "Features"
            << std::setw(12) << "Upload(ms)"
            << std::setw(14) << "Extract(ms)"
            << std::setw(12) << "Match(ms)"
            << std::setw(12) << "Total(ms)"
            << std::setw(10) << "FPS"
            << std::setw(12) << "Homog(ms)"
            << std::setw(12) << "HomGPU(ms)" << std::endl;
  std::cout << std::string(119, '-') << std::endl;

  for (auto &res : resolutions) {
    auto r = RunBenchmark(srcImg, res.w, res.h, numOctaves, thresh, warmupRuns, benchRuns);
    results.push_back(r);

    char sizeStr[32];
    snprintf(sizeStr, sizeof(sizeStr), "%dx%d", res.w, res.h);

    std::cout << std::setw(10) << res.name
              << std::setw(12) << sizeStr
              << std::setw(10) << r.numFeatures
              << std::setw(12) << std::fixed << std::setprecision(2) << r.uploadTimeMs
              << std::setw(14) << r.extractTimeMs
              << std::setw(12) << r.matchTimeMs
              << std::setw(12) << r.totalTimeMs
              << std::setw(10) << std::setprecision(1) << 1000.0 / r.totalTimeMs
              << std::setw(12) << std::setprecision(2) << r.homographyTimeMs
              << std::setw(12) << r.homographyGPUTimeMs
              << std::endl;
  }

  std::cout << std::string(119, '-') << std::endl;

  // Multi-octave comparison
  std::cout << "\n=== Octave Count Comparison (1080p) ===" << std::endl;
  std::cout << std::setw(10) << "Octaves"
            << std::setw(10) << "Features"
            << std::setw(14) << "Extract(ms)" << std::endl;
  std::cout << std::string(34, '-') << std::endl;

  for (int oct = 3; oct <= 6; oct++) {
    cv::Mat resized, fimg;
    cv::resize(srcImg, resized, cv::Size(1920, 1080));
    resized.convertTo(fimg, CV_32FC1);
    CudaImage cudaImg;
    cudaImg.Allocate(1920, 1080, iAlignUp(1920, 128), false, NULL, (float *)fimg.data);
    cudaImg.Download();
    SiftData siftData;
    InitSiftData(siftData, 32768, true, true);
    float *mem = AllocSiftTempMemory(1920, 1080, oct, false);

    // Warmup
    for (int i = 0; i < 20; i++)
      ExtractSift(siftData, cudaImg, oct, 1.0f, thresh, 0.0f, false, mem);

    // Measure
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
      ExtractSift(siftData, cudaImg, oct, 1.0f, thresh, 0.0f, false, mem);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << std::setw(10) << oct
              << std::setw(10) << siftData.numPts
              << std::setw(14) << std::fixed << std::setprecision(2) << ms / 100.0
              << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    FreeSiftTempMemory(mem);
    FreeSiftData(siftData);
  }

  // Threshold comparison
  std::cout << "\n=== Threshold Comparison (1080p, 5 octaves) ===" << std::endl;
  std::cout << std::setw(10) << "Threshold"
            << std::setw(10) << "Features"
            << std::setw(14) << "Extract(ms)" << std::endl;
  std::cout << std::string(34, '-') << std::endl;

  float thresholds[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 7.0f, 10.0f};
  for (float t : thresholds) {
    cv::Mat resized, fimg;
    cv::resize(srcImg, resized, cv::Size(1920, 1080));
    resized.convertTo(fimg, CV_32FC1);
    CudaImage cudaImg;
    cudaImg.Allocate(1920, 1080, iAlignUp(1920, 128), false, NULL, (float *)fimg.data);
    cudaImg.Download();
    SiftData siftData;
    InitSiftData(siftData, 32768, true, true);
    float *mem = AllocSiftTempMemory(1920, 1080, 5, false);

    for (int i = 0; i < 20; i++)
      ExtractSift(siftData, cudaImg, 5, 1.0f, t, 0.0f, false, mem);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
      ExtractSift(siftData, cudaImg, 5, 1.0f, t, 0.0f, false, mem);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << std::setw(10) << std::fixed << std::setprecision(1) << t
              << std::setw(10) << siftData.numPts
              << std::setw(14) << std::setprecision(2) << ms / 100.0
              << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    FreeSiftTempMemory(mem);
    FreeSiftData(siftData);
  }

  // Print summary in Markdown table format
  std::cout << "\n=============================================" << std::endl;
  std::cout << "  RESULTS (Markdown table for README)" << std::endl;
  std::cout << "=============================================" << std::endl;
  std::cout << "\n| Resolution | Features | Extract (ms) | Match (ms) | Total (ms) | FPS | Homography (ms) | GPU homography (ms) |" << std::endl;
  std::cout << "|------------|----------|-------------|------------|------------|-----|-----------------|---------------------|" << std::endl;
  for (size_t i = 0; i < results.size(); i++) {
    auto &r = results[i];
    std::cout << "| " << std::setw(10) << std::left << resolutions[i].name
              << " | " << std::setw(8) << std::right << r.numFeatures
              << " | " << std::setw(11) << std::fixed << std::setprecision(2) << r.extractTimeMs
              << " | " << std::setw(10) << r.matchTimeMs
              << " | " << std::setw(10) << r.totalTimeMs
              << " | " << std::setw(3) << std::setprecision(0) << 1000.0 / r.totalTimeMs
              << " | " << std::setw(15) << std::setprecision(2) << r.homographyTimeMs
              << " | " << std::setw(19) << r.homographyGPUTimeMs
              << " |" << std::endl;
  }

  std::cout << "\nTotal: upload + extraction + matching. Homography: FindHomography + ImproveHomography;"
            << "\nGPU homography: FindHomographyGPU + ImproveHomographyGPU (10000 hypotheses, 5 iterations, on the self-match)." << std::endl;
  std::cout << "\nBenchmark complete!" << std::endl;
  return 0;
}
