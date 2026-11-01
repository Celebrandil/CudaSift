//********************************************************//
// Test: homography estimation on the GPU                 //
// (FindHomographySeeded, FindHomographyGPU,              //
//  ImproveHomographyGPU)                                 //
//********************************************************//

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
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

static void Project(const float *H, float x, float y, float &u, float &v)
{
  float w = H[6]*x + H[7]*y + H[8];
  u = (H[0]*x + H[1]*y + H[2])/w;
  v = (H[3]*x + H[4]*y + H[5])/w;
}

// Mean and maximum distance between the projections by two homographies of the points
// of data that are inliers of H1 (reprojection error < thresh)
static void Compare(SiftData &data, const float *H1, const float *H2, float thresh, double &mean, double &maxd)
{
  mean = maxd = 0.0;
  int n = 0;
  for (int i=0;i<data.numPts;i++) {
    const SiftPoint &p = data.h_data[i];
    float u1, v1, u2, v2;
    Project(H1, p.xpos, p.ypos, u1, v1);
    if (std::hypot(u1 - p.match_xpos, v1 - p.match_ypos)>=thresh)
      continue;
    Project(H2, p.xpos, p.ypos, u2, v2);
    double d = std::hypot(u1 - u2, v1 - v2);
    mean += d;
    maxd = std::max(maxd, d);
    n++;
  }
  if (n)
    mean /= n;
}

static bool Load(CudaImage &img, cv::Mat &fimg, const cv::Mat &src)
{
  if (src.empty())
    return false;
  src.convertTo(fimg, CV_32FC1);
  img.Allocate(fimg.cols, fimg.rows, iAlignUp(fimg.cols, 128), false, NULL, (float *)fimg.data);
  img.Download();
  return true;
}

int main(int argc, char **argv)
{
  std::cout << "=============================================" << std::endl;
  std::cout << "  CudaSift GPU Homography Tests" << std::endl;
  std::cout << "=============================================" << std::endl;
  InitCuda(argc>1 ? std::atoi(argv[1]) : 0);
  std::cout << std::setprecision(4);

  CudaImage img1, img2;
  cv::Mat f1, f2;
  CHECK(Load(img1, f1, cv::imread("data/img1.png", 0)) && Load(img2, f2, cv::imread("data/img2.png", 0)), "Images loaded");
  SiftData s1, s2;
  InitSiftData(s1, 32768, true, true);
  InitSiftData(s2, 32768, true, true);
  ExtractSift(s1, img1, 5, 1.0f, 3.0f);
  ExtractSift(s2, img2, 5, 1.0f, 3.0f);
  MatchSiftData(s1, s2);
  std::vector<SiftPoint> saved(s1.h_data, s1.h_data + s1.numPts);   // to restore match_error

  std::cout << "\n--- Test: FindHomographyGPU ---" << std::endl;
  float Ha[9], Hb[9], Hr[9];
  int na = 0, nb = 0, nr = 0;
  FindHomographyGPU(s1, Ha, &na, 10000, 0.00f, 0.80f, 5.0f, 1);
  FindHomographyGPU(s1, Hb, &nb, 10000, 0.00f, 0.80f, 5.0f, 1);
  CHECK(na==nb && !memcmp(Ha, Hb, sizeof(Ha)), "Same seed, same result");
  FindHomography(s1, Hr, &nr, 10000, 0.00f, 0.80f, 5.0f);
  double mean, maxd;
  Compare(s1, Hr, Ha, 5.0f, mean, maxd);
  std::cout << "    FindHomography: " << nr << " matches, FindHomographyGPU: " << na << " matches" << std::endl;
  std::cout << "    Distance between the projections of the inliers: mean " << mean << " px, max " << maxd << " px" << std::endl;
  CHECK(na>20, "Found > 20 matches");
  CHECK(std::abs(na - nr)<=0.03*nr, "Matches within 3% of FindHomography");
  // After refinement (on the CPU), both should give the same homography
  float Rr[9], Ra[9];
  memcpy(Rr, Hr, sizeof(Hr));
  memcpy(Ra, Ha, sizeof(Ha));
  int fr = ImproveHomography(s1, Rr, 5, 0.00f, 0.80f, 3.0f);
  int fa = ImproveHomography(s1, Ra, 5, 0.00f, 0.80f, 3.0f);
  memcpy(s1.h_data, saved.data(), sizeof(SiftPoint)*s1.numPts);
  Compare(s1, Rr, Ra, 3.0f, mean, maxd);
  std::cout << "    After ImproveHomography: " << fr << " / " << fa << " inliers, distance between the projections: mean " << mean << " px, max " << maxd << " px" << std::endl;
  CHECK(std::abs(fr - fa)<=0.01*fr, "Refined inliers within 1% of those from FindHomography");
  CHECK(mean<0.1, "Refined homography close to that from FindHomography (mean distance < 0.1 px)");
  int nmin = na, nmax = na;
  for (unsigned int seed=2;seed<=10;seed++) {
    int n;
    float H[9];
    FindHomographyGPU(s1, H, &n, 10000, 0.00f, 0.80f, 5.0f, seed);
    nmin = std::min(nmin, n);
    nmax = std::max(nmax, n);
  }
  std::cout << "    Matches over seeds 1-10: " << nmin << " - " << nmax << std::endl;
  CHECK(nmax - nmin<=0.03*nmax, "Matches within 3% over 10 seeds");

  std::cout << "\n--- Test: FindHomographySeeded ---" << std::endl;
  {
    // Both draw the same samples from the valid matches, in the order of the points
    float Hs[9], Hg[9];
    int ns = 0, ng = 0;
    FindHomographySeeded(s1, Hs, &ns, 10000, 0.00f, 0.80f, 5.0f, 7);
    FindHomographyGPU(s1, Hg, &ng, 10000, 0.00f, 0.80f, 5.0f, 7);
    std::cout << "    FindHomographySeeded: " << ns << " matches, FindHomographyGPU: " << ng << " matches" << std::endl;
    CHECK(ns==ng && !memcmp(Hs, Hg, sizeof(Hs)), "FindHomographySeeded and FindHomographyGPU: same result");
  }

  std::cout << "\n--- Test: SortSiftData ---" << std::endl;
  {
    // Two extractions of the same images, sorted: the same points in the same order, so the
    // same matches and the same homography
    SiftData t1[2], t2[2];
    float Ht[2][9];
    int nt[2] = {0, 0};
    bool samePoints = true, ordered = true;
    for (int r=0;r<2;r++) {
      InitSiftData(t1[r], 32768, true, true);
      InitSiftData(t2[r], 32768, true, true);
      ExtractSift(t1[r], img1, 5, 1.0f, 3.0f);
      ExtractSift(t2[r], img2, 5, 1.0f, 3.0f);
      SortSiftData(t1[r]);
      SortSiftData(t2[r]);
      MatchSiftData(t1[r], t2[r]);
      FindHomographyGPU(t1[r], Ht[r], &nt[r], 10000, 0.00f, 0.80f, 5.0f, 3);
    }
    for (int i=1;i<t1[0].numPts;i++) {
      const SiftPoint &p = t1[0].h_data[i-1], &q = t1[0].h_data[i];
      ordered &= (p.xpos<q.xpos || (p.xpos==q.xpos && p.ypos<=q.ypos));
    }
    samePoints = (t1[0].numPts==t1[1].numPts && t1[0].numPts>0);
    for (int i=0;samePoints && i<t1[0].numPts;i++)
      samePoints = !memcmp(&t1[0].h_data[i], &t1[1].h_data[i], sizeof(SiftPoint));
    std::cout << "    " << t1[0].numPts << " points, " << nt[0] << " and " << nt[1] << " matches" << std::endl;
    CHECK(ordered, "Points sorted by position");
    CHECK(samePoints, "Two sorted extractions: the same points in the same order, matches included");
    CHECK(nt[0]==nt[1] && nt[0]>0 && !memcmp(Ht[0], Ht[1], sizeof(Ht[0])), "Two sorted extractions: the same homography");
    for (int r=0;r<2;r++) {
      FreeSiftData(t1[r]);
      FreeSiftData(t2[r]);
    }
  }

  std::cout << "\n--- Test: ImproveHomographyGPU ---" << std::endl;
  float Hc[9], Hg[9];
  memcpy(Hc, Ha, sizeof(Ha));
  memcpy(Hg, Ha, sizeof(Ha));
  int fitCpu = ImproveHomography(s1, Hc, 5, 0.00f, 0.80f, 3.0f);
  std::vector<float> errCpu(s1.numPts);
  for (int i=0;i<s1.numPts;i++)
    errCpu[i] = s1.h_data[i].match_error;
  memcpy(s1.h_data, saved.data(), sizeof(SiftPoint)*s1.numPts);
  int fitGpu = 0;
  ImproveHomographyGPU(s1, Hg, &fitGpu, 5, 0.00f, 0.80f, 3.0f);
  double maxErr = 0.0, sumErr = 0.0;
  int nErr = 0;
  for (int i=0;i<s1.numPts;i++)
    if (errCpu[i]<3.0f) {
      double d = std::fabs(errCpu[i] - s1.h_data[i].match_error);
      maxErr = std::max(maxErr, d);
      sumErr += d;
      nErr++;
    }
  Compare(s1, Hc, Hg, 3.0f, mean, maxd);
  std::cout << "    ImproveHomography (CPU): " << fitCpu << " inliers, ImproveHomographyGPU: " << fitGpu << " inliers" << std::endl;
  std::cout << "    Distance between the projections of the inliers: mean " << mean << " px, max " << maxd << " px" << std::endl;
  std::cout << "    match_error difference of the inliers: mean " << (nErr ? sumErr/nErr : 0.0) << " px, max " << maxErr << " px" << std::endl;
  CHECK(std::abs(fitCpu - fitGpu)<=1, "Same number of inliers (+-1)");
  CHECK(maxd<0.01, "Same homography (max distance < 0.01 px)");
  CHECK(maxErr<0.01, "Same match_error (max difference < 0.01 px)");
  {
    // Fewer than 4 inliers pass the filter: the homography must be kept. maxAmbiguity 0
    // leaves none; the ambiguity of the 3rd most distinctive inlier of Ha leaves about 3.
    std::vector<float> amb;
    for (int i=0;i<s1.numPts;i++) {
      const SiftPoint &p = saved[i];
      float u, v;
      Project(Ha, p.xpos, p.ypos, u, v);
      if ((u - p.match_xpos)*(u - p.match_xpos) + (v - p.match_ypos)*(v - p.match_ypos)<9.0f)
        amb.push_back(p.ambiguity);
    }
    std::sort(amb.begin(), amb.end());
    int kept = 0, numInl = 0;
    for (float maxAmb : {0.0f, amb.size()>=3 ? amb[2] : 0.0f}) {
      int n = 0;
      for (int i=0;i<(int)amb.size();i++)
        n += (amb[i]<=maxAmb);
      float Hk[9];
      memcpy(Hk, Ha, sizeof(Ha));
      int fitKeep = 0;
      ImproveHomographyGPU(s1, Hk, &fitKeep, 5, 0.00f, maxAmb, 3.0f);
      bool same = true;
      for (int i=0;i<9;i++)
        same = same && std::fabs(Hk[i]/Hk[8] - Ha[i]/Ha[8])<=1e-6f*std::max(1.0f, std::fabs(Ha[i]/Ha[8]));
      kept += (n<4 && same && fitKeep>0);
      numInl = std::max(numInl, n);
      memcpy(s1.h_data, saved.data(), sizeof(SiftPoint)*s1.numPts);
    }
    std::cout << "    Homography kept with 0 and " << numInl << " inliers: " << kept << " of 2" << std::endl;
    CHECK(kept==2, "Homography kept with fewer than 4 inliers");
  }

  std::cout << "\n--- Test: Translation ---" << std::endl;
  {
    cv::Mat img = cv::imread("data/img1.png", 0), translated;
    cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, 30, 0, 1, 20);
    cv::warpAffine(img, translated, M, img.size());
    CudaImage t1, t2;
    cv::Mat g1, g2;
    Load(t1, g1, img);
    Load(t2, g2, translated);
    SiftData d1, d2;
    InitSiftData(d1, 32768, true, true);
    InitSiftData(d2, 32768, true, true);
    ExtractSift(d1, t1, 5, 1.0f, 3.0f);
    ExtractSift(d2, t2, 5, 1.0f, 3.0f);
    MatchSiftData(d1, d2);
    float H[9];
    int n, fit;
    FindHomographyGPU(d1, H, &n, 10000, 0.00f, 0.80f, 5.0f);
    ImproveHomographyGPU(d1, H, &fit, 5, 0.00f, 0.80f, 3.0f);
    // Displacement at the image centre: h[2], h[5] (the displacement at the origin) depend on
    // the small rotation and perspective terms of the fit, far from most of the matches
    float u, v;
    Project(H, img.cols/2.0f, img.rows/2.0f, u, v);
    u -= img.cols/2.0f;
    v -= img.rows/2.0f;
    std::cout << "    Displacement at the image centre (" << u << ", " << v << "), " << fit << " inliers" << std::endl;
    CHECK(std::fabs(u - 30.0f)<1.0f && std::fabs(v - 20.0f)<1.0f, "Translation (30, 20) recovered within 1 px at the image centre");
    FreeSiftData(d1);
    FreeSiftData(d2);
  }

  std::cout << "\n--- Test: Inliers among the valid matches, orientation check ---" << std::endl;
  {
    // As TestHomographyValidOnly and TestHomographyOrientation in test_homography.cpp: 20
    // valid matches under the identity, and either 10 valid and 40 ambiguous matches under
    // a translation, or 30 valid matches under a mirror
    for (int mirror=0;mirror<2;mirror++) {
      const int numPts = (mirror ? 50 : 70);
      SiftData d;
      InitSiftData(d, numPts, true, true);
      for (int i=0;i<numPts;i++) {
        SiftPoint &pt = d.h_data[i];
        pt.xpos = 50.0f + 37.0f*(i%7) + 3.0f*(i/7);
        pt.ypos = 40.0f + 29.0f*(i/7) + 2.0f*(i%3);
        const bool other = (i>=20);
        pt.match_xpos = (!other ? pt.xpos : mirror ? 400.0f - pt.xpos : pt.xpos + 30.0f);
        pt.match_ypos = (!other || mirror ? pt.ypos : pt.ypos + 20.0f);
        pt.score = 1.0f;
        pt.ambiguity = (!mirror && i>=30 ? 0.95f : 0.0f);
      }
      d.numPts = numPts;
      if (d.d_data!=d.h_data)
        cudaMemcpy(d.d_data, d.h_data, sizeof(SiftPoint)*numPts, cudaMemcpyHostToDevice);
      float H[9];
      int n = 0;
      FindHomographyGPU(d, H, &n, 1024, 0.0f, 0.8f, 3.0f);
      std::cout << "    " << (mirror ? "Mirror" : "Ambiguous translation") << ": " << n << " matches, H[0] " << H[0] << ", H[2] " << H[2] << std::endl;
      CHECK(n==20 && std::fabs(H[0] - 1.0f)<0.01f && std::fabs(H[2])<1.0f,
            mirror ? "Identity, not the mirror" : "Identity, inliers among the valid matches only");
      FreeSiftData(d);
    }
  }

  FreeSiftData(s1);
  FreeSiftData(s2);
  std::cout << "\n=============================================" << std::endl;
  std::cout << "  Results: " << testsPassed << " passed, " << testsFailed << " failed" << std::endl;
  std::cout << "=============================================" << std::endl;
  return testsFailed>0 ? 1 : 0;
}
