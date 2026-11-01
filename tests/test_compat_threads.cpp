//********************************************************//
// ExtractSiftCompat from several host threads at once    //
//********************************************************//

// Each thread runs all the configurations on both images, in its own order; each result must be
// the same, byte for byte, as that of the same call made from one thread.
// Usage: test_compat_threads [device]
#include <cstdio>
#include <thread>
#include <vector>
#include <atomic>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "cudaImage.h"
#include "cudaSift.h"
#include "cudaSiftCompat.h"

static SiftCompatParams Config(int c)
{
  SiftCompatParams p = c%3==1 ? SiftCompatParams::VLFeat() : c%3==2 ? SiftCompatParams::COLMAP() : SiftCompatParams::OpenCV();
  if (c>=3) {
    p.firstOctave = 0;
    p.extendFirstOctave = true;
  }
  p.doublePrecision = (c==4);
  return p;
}

static unsigned long long Extract(const cv::Mat &f, int c)
{
  CudaImage img;
  img.Allocate(f.cols, f.rows, iAlignUp(f.cols, 128), false, NULL, (float *)f.data);
  img.Download();
  SiftData data;
  InitSiftData(data, 65536, true, true);
  ExtractSiftCompat(data, img, Config(c));
  unsigned long long h = 1469598103934665603ULL;
#ifdef MANAGEDMEM
  const unsigned char *b = (const unsigned char *)data.m_data;
#else
  const unsigned char *b = (const unsigned char *)data.h_data;
#endif
  for (size_t k=0;k<(size_t)data.numPts*sizeof(SiftPoint);k++)
    h = (h ^ b[k])*1099511628211ULL;
  FreeSiftData(data);
  return h;
}

int main(int argc, char **argv)
{
  const int NC = 6, NT = 4, runs = 5;
  const int device = argc>1 ? atoi(argv[1]) : 0;
  InitCuda(device);
  std::vector<cv::Mat> imgs;
  for (const char *n : {"data/img1.png", "data/img2.png"}) {
    cv::Mat g = cv::imread(n, cv::IMREAD_GRAYSCALE), f;
    g.convertTo(f, CV_32FC1);
    imgs.push_back(f);
  }
  unsigned long long ref[2][NC];
  for (int i=0;i<2;i++)
    for (int c=0;c<NC;c++)
      ref[i][c] = Extract(imgs[i], c);
  std::atomic<int> bad(0), done(0);
  std::vector<std::thread> th;
  for (int t=0;t<NT;t++)
    th.emplace_back([&, t]() {
      cudaSetDevice(device);   // each thread starts on device 0
      for (int r=0;r<runs;r++)
        for (int k=0;k<NC;k++) {
          int c = (k + t)%NC, i = (r + t)%2;
          if (Extract(imgs[i], c)!=ref[i][c])
            bad++;
          done++;
        }
    });
  for (auto &x : th)
    x.join();
  FreeSiftCompatMemory();
  printf("  [%s] %d calls from %d threads, %d differ from the single-thread results\n", bad>0 ? "FAIL" : "PASS",
         done.load(), NT, bad.load());
  return bad>0;
}
