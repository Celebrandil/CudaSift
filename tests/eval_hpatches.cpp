//********************************************************//
// HPatches evaluation: matching and homography accuracy  //
//********************************************************//

// Evaluates extraction, matching and homography estimation on image sequences with
// ground-truth homographies: HPatches (https://github.com/hpatches/hpatches-dataset,
// hpatches-sequences-release: 1.ppm..6.ppm, H_1_2..H_1_6), or synthetic pairs made by
// --make-synthetic (1.png, 2.png, H_1_2), with the transformations of Lowe's synthetic tests.
// Every subdirectory with a first image is a sequence. For each pair (the first image and another one):
// - matches: points of the reference image whose match passes Lowe's ratio test (0.8 on
//   the descriptor distances), and the matching accuracy at 1..10 px (MMA): the share of
//   these matches within that distance of the ground-truth projection; for the matches
//   within 3 px, their distance (localization) and log2 of their scale over the expected
//   scale (the reference scale times the local scale change of the homography);
// - repeatability: keypoints of both images paired one to one when the projection is within
//   2.5 px and the scales agree within a factor 2^0.25, over the keypoints of the part of the
//   scene seen in both images (the detector alone); its chance level (the same with the
//   projections moved away, see Repeatability) and the chance-corrected repeatability; the
//   repeatability of the 500, 1000 and 2000 strongest keypoints of each image; matching
//   score: the matches within 3 px over the keypoints of the overlap (detector and
//   descriptors, Mikolajczyk and Schmid);
// - homography: FindHomography (10000 hypotheses, 5 px) and ImproveHomography (5
//   iterations, 3 px), and the mean distance between the 4 image corners projected by
//   the estimated and the ground-truth homographies, counted as correct below 1, 3, 5 px.
// Only the API common to all versions is used, so the same tool compares versions.
//
//   eval_hpatches <dataset dir> [--opencv] [--opencv-legacy-upscale] [--vlfeat] [--vlfeat-colmap]
//                 [--compat opencv|vlfeat|colmap [--first-octave n] [--extend-first-octave [--extend-scales n]] [--contrast-threshold t] [--double]]
//                 [--gpu-homography]
//                 [--opencv-homography] [--cudasift-homography] [--oracle px]
//                 [--init-blur sigma] [--thresh t] [--repeat-curve] [--chance-k k] [--profile] [--jobs n] [--gpu-matching] [--per-sequence] [--skip sequence]...
//                 [--csv file] [device]
//   eval_hpatches <output dir> --make-synthetic image,image,...
//
// --opencv evaluates OpenCV's SIFT instead (OpenCV 4.4 or later; default parameters, with
// enable_precise_upscale: OpenCV 4.8 or later, or --opencv-legacy-upscale to accept the
// +0.25 px shift of older versions; brute-force matching with the same ratio test,
// findHomography with RANSAC at 5 px and 10000 iterations). --vlfeat does the same with
// VLFeat's SIFT (default parameters, images in [0, 1]), and --vlfeat-colmap with COLMAP's
// settings on VLFeat (see CpuSift below), in builds with VLFeat (HAVE_VLFEAT). --compat
// extracts CudaSift's features with ExtractSiftCompat, in the given mode, instead of
// ExtractSift, in builds that have it (CUDASIFT_HAS_COMPAT); --first-octave (-1 or 0),
// --extend-first-octave (first octave 0, with the levels of octave -1 below sigma),
// --extend-scales (the number of added scales kept), --contrast-threshold (OpenCV mode) and
// --double (double precision) change its parameters. --gpu-homography uses
// FindHomographyGPU and ImproveHomographyGPU, with the same parameters, in versions that
// have them (cudaSift.h defines CUDASIFT_HAS_GPU_HOMOGRAPHY). The features and the
// homography estimator can be combined: --opencv-homography estimates the homography of
// CudaSift's matches with OpenCV's findHomography, and --cudasift-homography (or
// --gpu-homography) that of OpenCV's matches with FindHomography and ImproveHomography, to
// separate the effects of the features and of the estimator. --oracle px fits the
// homography to the matches within px of the ground truth (least squares and
// Levenberg-Marquardt), an upper bound for any estimator on these matches. --init-blur and
// --thresh set those ExtractSift parameters (1.0 and 3.0), e.g. to compare at the same
// number of features. --per-sequence also summarizes each sequence. --skip leaves out a
// sequence (MatchSiftData crashes on HPatches' i_fog without the fix of
// https://github.com/Celebrandil/CudaSift/pull/96). Times are per pair: the extraction of the
// target image, matching, and homography estimation.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <random>
#include <atomic>
#include <future>
#include <mutex>
#include <thread>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <cuda_runtime.h>
#include "cudaImage.h"
#include "cudaSift.h"
#ifdef CUDASIFT_HAS_COMPAT   // set by cudasift_lib in builds that compile ExtractSiftCompat
#include "cudaSiftCompat.h"
#define EVAL_HAS_COMPAT 1
#endif
#ifdef HAVE_VLFEAT
extern "C" {
#include <vl/sift.h>
}
#endif

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);

static const float maxAmbiguity = 0.8f;
enum Estimator { EST_DEFAULT, EST_CUDASIFT, EST_GPU, EST_OPENCV, EST_ORACLE };
static float oracleThresh = 3.0f;
static float initBlur = 1.0f, dogThresh = 3.0f;   // ExtractSift parameters
static int compatMode = -1;   // ExtractSiftCompat: 0 OpenCV, 1 VLFeat, 2 COLMAP (-1: ExtractSift)
static int compatFirstOctave = -2;            // -2: the mode's default
static bool compatExtendFirstOctave = false;  // the levels of octave -1 below sigma, undoubled
static int compatExtendScales = -1;           // added detection scales kept (-1: all)
static bool compatDouble = false;             // doublePrecision
static float compatContrastThreshold = -1.0f; // OpenCV mode; < 0: the default
static Estimator estimator = EST_DEFAULT;   // EST_DEFAULT: the one of the features

struct PairResult {
  std::string seq;
  int k;                  // target image 2..6
  int numPts1, numPts2;   // features
  int numMatches;         // passing the ratio test
  int correct[11];        // matches within t px, t = 1..10
  std::vector<float> err3;   // distances to the ground truth of the matches within 3 px
  int numInliers;         // of the RANSAC homography (before refinement), at 5 px
  double cornerError;     // of the estimated homography, 1e9 if none
  double extractMs, matchMs, homographyMs;   // per pair: target extraction, matching, homography
  int numOverlap;         // keypoints in the part of the scene seen in both images (min of both)
  int numPosRepeated;     // reference keypoints with a target keypoint within repeatPx
  int numRepeated;        // ... whose scale also agrees within a factor 2^repeatScale
  double chanceRepeat;    // repeatability by chance: with the projections moved away (R0)
  double budgetRepeat[3]; // repeatability of the strongest budgets[b] keypoints, -1 if fewer
  std::vector<double> curve;   // --repeat-curve: repeatability with the projections moved
  std::vector<float> scaleErr;   // log2(found / expected scale) of the matches within 3 px
};

// A keypoint: position, scale (sigma) and the magnitude of its response (DoG contrast)
struct Keypoint {
  float x, y, s, r;
};

static const float repeatPx = 2.5f, repeatScale = 0.25f;
// Chance level of the repeatability: the projections of the reference keypoints moved by
// max(2 repeatPx, chanceK sigma), in 8 directions (see Repeatability)
static float chanceK = 3.0f;
static const int budgets[3] = {500, 1000, 2000};
// --repeat-curve: the moves, fixed (px) and scale-adaptive (times sigma, negative here)
static bool repeatCurve = false;
static const float curveMoves[] = {1.25f, 2.5f, 3.75f, 5.0f, 7.5f, 10.0f, 15.0f, 20.0f, 30.0f, 40.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f};
static const int numCurveMoves = sizeof(curveMoves)/sizeof(curveMoves[0]);

static double Ms(std::chrono::steady_clock::time_point t0)
{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static bool ReadH(const std::string &file, double H[9])
{
  std::ifstream f(file);
  for (int i=0;i<9;i++)
    if (!(f >> H[i]))
      return false;
  return true;
}

static void Project(const double H[9], double x, double y, double &u, double &v)
{
  double z = H[6]*x + H[7]*y + H[8];
  u = (H[0]*x + H[1]*y + H[2])/z;
  v = (H[3]*x + H[4]*y + H[5])/z;
}

static double CornerError(const double Hgt[9], const double Hest[9], int w, int h)
{
  const double cx[4] = {0, (double)w, (double)w, 0}, cy[4] = {0, 0, (double)h, (double)h};
  double e = 0.0;
  for (int i=0;i<4;i++) {
    double u1, v1, u2, v2;
    Project(Hgt, cx[i], cy[i], u1, v1);
    Project(Hest, cx[i], cy[i], u2, v2);
    e += std::hypot(u1 - u2, v1 - v2);
  }
  return std::isfinite(e) ? e/4 : 1e9;
}

// Inverse of a homography
static void InvertH(const double H[9], double Hi[9])
{
  double a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7], i = H[8];
  double A = e*i - f*h, B = -(d*i - f*g), C = d*h - e*g;
  double det = a*A + b*B + c*C;
  double M[9] = {A, -(b*i - c*h), b*f - c*e, B, a*i - c*g, -(a*f - c*d), C, -(a*h - b*g), a*e - b*d};
  for (int k=0;k<9;k++)
    Hi[k] = M[k]/det;
}

// Local scale change of the homography at (x, y): sqrt of the determinant of its Jacobian
static double LocalScale(const double H[9], double x, double y)
{
  double u, v;
  Project(H, x, y, u, v);
  double z = H[6]*x + H[7]*y + H[8];
  double j00 = (H[0] - u*H[6])/z, j01 = (H[1] - u*H[7])/z, j10 = (H[3] - v*H[6])/z, j11 = (H[4] - v*H[7])/z;
  return std::sqrt(std::fabs(j00*j11 - j01*j10));
}

// Repeatability: reference and target keypoints paired one to one (nearest first), when the
// reference keypoint projects within repeatPx of the target keypoint and their scales agree,
// within a factor 2^repeatScale, after the local scale change of the homography; over the
// keypoints of the part of the scene seen in both images (the smaller of the two counts).
// numPosRepeated: the same without the scale condition.
// A dense detector also pairs keypoints by chance, so the chance level R0 is measured too:
// the same pairing with each projection moved by max(2 repeatPx, chanceK sigma) (sigma: its
// expected scale in the target), averaged over 8 directions. This keeps the clustering and
// scales of both sets but breaks the true correspondences; the chance-corrected
// repeatability is (R - R0)/(1 - R0). And the repeatability of the strongest N keypoints of
// each image (by response), at the budgets N, compares detectors at the same density.
struct RepeatSets {
  const double *H = nullptr;
  const std::vector<Keypoint> *k1 = nullptr, *k2 = nullptr;
  std::vector<int> sel1, sel2;   // the keypoints used
  int w1 = 0, h1 = 0, w2 = 0, h2 = 0;
  static constexpr float cell = 4.0f;
  int gw = 0, gh = 0;
  std::vector<int> cellStart, cellItems;   // grid of sel2 (indices into sel2), as offsets
  std::vector<float> x2, y2, ls2;          // sel2: positions and log2 scales
  std::vector<double> u, v, expected;      // projections of sel1 into the target
  std::vector<float> le;                   // log2 of the expected scales
  std::vector<char> inside;
  int n1 = 0, n2 = 0;

  void Build()
  {
    double Hi[9];
    InvertH(H, Hi);
    gw = (int)(w2/cell) + 1;
    gh = (int)(h2/cell) + 1;
    std::vector<int> cellOf(sel2.size());
    cellStart.assign(gw*gh + 1, 0);
    x2.resize(sel2.size());
    y2.resize(sel2.size());
    ls2.resize(sel2.size());
    n2 = 0;
    for (size_t t=0;t<sel2.size();t++) {
      const Keypoint &q = (*k2)[sel2[t]];
      int gx = std::min(std::max((int)(q.x/cell), 0), gw - 1), gy = std::min(std::max((int)(q.y/cell), 0), gh - 1);
      cellOf[t] = gy*gw + gx;
      cellStart[cellOf[t] + 1]++;
      x2[t] = q.x;
      y2[t] = q.y;
      ls2[t] = std::log2(q.s);
      double a, b;
      Project(Hi, q.x, q.y, a, b);
      n2 += (a>=0 && a<w1 && b>=0 && b<h1);
    }
    for (int c=0;c<gw*gh;c++)
      cellStart[c + 1] += cellStart[c];
    cellItems.resize(sel2.size());
    std::vector<int> fill(cellStart.begin(), cellStart.end() - 1);
    for (size_t t=0;t<sel2.size();t++)
      cellItems[fill[cellOf[t]]++] = (int)t;
    n1 = 0;
    u.resize(sel1.size());
    v.resize(sel1.size());
    expected.resize(sel1.size());
    le.resize(sel1.size());
    inside.resize(sel1.size());
    for (size_t t=0;t<sel1.size();t++) {
      const Keypoint &p = (*k1)[sel1[t]];
      Project(H, p.x, p.y, u[t], v[t]);
      inside[t] = (u[t]>=0 && u[t]<w2 && v[t]>=0 && v[t]<h2);
      n1 += inside[t];
      expected[t] = p.s*LocalScale(H, p.x, p.y);
      le[t] = (float)std::log2(expected[t]);
    }
  }

  // Pairs with the projections moved by (dx, dy)*move(t), nearest first, one to one;
  // returns {position only (if wanted), with scale}
  template <typename Move>
  std::pair<int, int> Count(double dx, double dy, Move move, bool wantPos) const
  {
    struct Cand { float d2; int i, j; bool scaleOk; };
    std::vector<Cand> cands;
    cands.reserve(2*sel1.size());
    const float r2 = repeatPx*repeatPx;
    for (size_t t=0;t<sel1.size();t++) {
      if (!inside[t])
        continue;
      double m = move(t);
      float uu = (float)(u[t] + dx*m), vv = (float)(v[t] + dy*m);
      int gx0 = (int)std::floor((uu - repeatPx)/cell), gx1 = (int)std::floor((uu + repeatPx)/cell);
      int gy0 = (int)std::floor((vv - repeatPx)/cell), gy1 = (int)std::floor((vv + repeatPx)/cell);
      for (int gy=std::max(gy0, 0);gy<=std::min(gy1, gh - 1);gy++)
        for (int gx=std::max(gx0, 0);gx<=std::min(gx1, gw - 1);gx++) {
          const int c = gy*gw + gx;
          for (int k=cellStart[c];k<cellStart[c + 1];k++) {
            const int j = cellItems[k];
            float ex = x2[j] - uu, ey = y2[j] - vv, d2 = ex*ex + ey*ey;
            if (d2<=r2)
              cands.push_back({d2, (int)t, j, std::fabs(ls2[j] - le[t])<repeatScale});
          }
        }
    }
    std::sort(cands.begin(), cands.end(), [](const Cand &a, const Cand &b) { return a.d2<b.d2; });
    std::vector<char> used1(sel1.size()), used2(sel2.size());
    auto Pairs = [&](bool needScale) {
      std::fill(used1.begin(), used1.end(), 0);
      std::fill(used2.begin(), used2.end(), 0);
      int n = 0;
      for (const Cand &c : cands)
        if ((!needScale || c.scaleOk) && !used1[c.i] && !used2[c.j]) {
          used1[c.i] = used2[c.j] = 1;
          n++;
        }
      return n;
    };
    int pos = (wantPos ? Pairs(false) : 0);
    return {pos, Pairs(true)};
  }

  // Mean repeatability over 8 directions with the projections moved by move(t)
  template <typename Move>
  double Moved(Move move) const
  {
    std::vector<int> n(8);
    cv::parallel_for_(cv::Range(0, 8), [&](const cv::Range &range) {
      for (int a=range.start;a<range.end;a++)
        n[a] = Count(std::cos(a*CV_PI/4), std::sin(a*CV_PI/4), move, false).second;
    });
    double sum = 0.0;
    for (int a=0;a<8;a++)   // in a fixed order
      sum += n[a];
    int overlap = std::min(n1, n2);
    return overlap ? sum/8/overlap : 0.0;
  }
};

// The indices of the n strongest keypoints (all if fewer)
static std::vector<int> Strongest(const std::vector<Keypoint> &k, int n)
{
  std::vector<int> idx(k.size());
  for (size_t i=0;i<k.size();i++)
    idx[i] = (int)i;
  if ((int)k.size()>n) {
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return k[a].r>k[b].r; });
    idx.resize(n);
  }
  return idx;
}

static void Repeatability(PairResult &r, const double Hgt[9], const std::vector<Keypoint> &k1, const std::vector<Keypoint> &k2, int w1, int h1, int w2, int h2)
{
  auto Sets = [&](int n) {
    RepeatSets R;
    R.H = Hgt;
    R.k1 = &k1;
    R.k2 = &k2;
    R.w1 = w1;
    R.h1 = h1;
    R.w2 = w2;
    R.h2 = h2;
    R.sel1 = Strongest(k1, n);
    R.sel2 = Strongest(k2, n);
    R.Build();
    return R;
  };
  const RepeatSets R = Sets((int)std::max(k1.size(), k2.size()));
  auto zero = [](size_t) { return 0.0; };
  std::pair<int, int> c = R.Count(0, 0, zero, true);
  r.numPosRepeated = c.first;
  r.numRepeated = c.second;
  r.numOverlap = std::min(R.n1, R.n2);
  r.chanceRepeat = R.Moved([&](size_t t) { return std::max(2.0*repeatPx, (double)chanceK*R.expected[t]); });
  r.curve.clear();
  if (repeatCurve)
    for (int m=0;m<numCurveMoves;m++) {
      float mv = curveMoves[m];
      r.curve.push_back(mv>0 ? R.Moved([&](size_t) { return (double)mv; }) : R.Moved([&](size_t t) { return -mv*R.expected[t]; }));
    }
  for (int b=0;b<3;b++) {
    r.budgetRepeat[b] = -1.0;
    if ((int)k1.size()<budgets[b] || (int)k2.size()<budgets[b])
      continue;
    const RepeatSets B = Sets(budgets[b]);
    int overlap = std::min(B.n1, B.n2);
    r.budgetRepeat[b] = (overlap ? (double)B.Count(0, 0, zero, false).second/overlap : 0.0);
  }
}

static void CountCorrect(PairResult &r, const double Hgt[9], const std::vector<cv::Point2f> &p1, const std::vector<cv::Point2f> &p2,
                         const std::vector<float> &s1, const std::vector<float> &s2)
{
  r.scaleErr.clear();
  r.numMatches = (int)p1.size();
  memset(r.correct, 0, sizeof(r.correct));
  r.err3.clear();
  for (size_t i=0;i<p1.size();i++) {
    double u, v;
    Project(Hgt, p1[i].x, p1[i].y, u, v);
    double d = std::hypot(u - p2[i].x, v - p2[i].y);
    for (int t=1;t<=10;t++)
      r.correct[t] += (d<=t);
    if (d<=3.0) {
      r.err3.push_back((float)d);
      r.scaleErr.push_back((float)std::log2(s2[i]/(s1[i]*LocalScale(Hgt, p1[i].x, p1[i].y))));
    }
  }
}

// The points of a SiftData on the host (with managed memory, the shared buffer)
static SiftPoint *HostPoints(SiftData &d)
{
#ifdef MANAGEDMEM
  cudaDeviceSynchronize();
  return d.m_data;
#else
  return d.h_data;
#endif
}

// Copies the host points to the device copy, if separate (FindHomography reads the device copy)
static void UploadPoints(SiftData &d)
{
#ifndef MANAGEDMEM
  if (d.d_data!=d.h_data && d.numPts>0)
    cudaMemcpy(d.d_data, d.h_data, sizeof(SiftPoint)*d.numPts, cudaMemcpyHostToDevice);
#endif
}

// Lowe's ratio of the descriptor distances, from MatchSiftData's best correlation and its
// ambiguity (the ratio of the second best to the best correlation): unit descriptors have a
// squared distance of 2 - 2*correlation
static float LoweRatio(float best, float ambiguity)
{
  const float second = ambiguity*best;
  return std::sqrt(std::max(0.0f, 1.0f - best)/std::max(1e-12f, 1.0f - second));
}

// The corner error of an OpenCV homography (none if empty)
static void ScoreHomography(PairResult &r, const double Hgt[9], int w, int h, const cv::Mat &H)
{
  if (H.empty())
    return;
  double Hd[9];
  for (int i=0;i<9;i++)
    Hd[i] = H.at<double>(i/3, i%3);
  r.cornerError = CornerError(Hgt, Hd, w, h);
}

// OpenCV's findHomography, RANSAC at 5 px, 10000 iterations at most
static void OpenCVHomography(PairResult &r, const double Hgt[9], int w, int h, const std::vector<cv::Point2f> &p1, const std::vector<cv::Point2f> &p2)
{
  cv::setRNGSeed(1);
  auto t0 = std::chrono::steady_clock::now();
  cv::Mat mask;
  cv::Mat H = cv::findHomography(p1, p2, cv::RANSAC, 5.0, mask, 10000, 0.995);
  r.homographyMs = Ms(t0);
  r.numInliers = (mask.empty() ? 0 : cv::countNonZero(mask));
  ScoreHomography(r, Hgt, w, h, H);
}

// Upper bound for any estimator: least squares and Levenberg-Marquardt (findHomography
// with method 0) on the matches within oracleThresh of the ground-truth projection
static void OracleHomography(PairResult &r, const double Hgt[9], int w, int h, const std::vector<cv::Point2f> &p1, const std::vector<cv::Point2f> &p2)
{
  std::vector<cv::Point2f> q1, q2;
  for (size_t i=0;i<p1.size();i++) {
    double u, v;
    Project(Hgt, p1[i].x, p1[i].y, u, v);
    if (std::hypot(u - p2[i].x, v - p2[i].y)<=oracleThresh) {
      q1.push_back(p1[i]);
      q2.push_back(p2[i]);
    }
  }
  r.numInliers = (int)q1.size();
  if (q1.size()<4)
    return;
  auto t0 = std::chrono::steady_clock::now();
  cv::Mat H = cv::findHomography(q1, q2, 0);
  r.homographyMs = Ms(t0);
  ScoreHomography(r, Hgt, w, h, H);
}

// FindHomography (10000 hypotheses, 5 px) and ImproveHomography (5 iterations, 3 px), or
// their GPU versions, on the matches of data (the device copy must be up to date)
static void CudaSiftHomography(PairResult &r, const double Hgt[9], int w, int h, SiftData &data)
{
  float H[9];
  int numMatches;
  srand(1);   // FindHomography samples with rand()
  auto t0 = std::chrono::steady_clock::now();
#ifdef CUDASIFT_HAS_GPU_HOMOGRAPHY
  if (estimator==EST_GPU) {
    int numFit;
    FindHomographyGPU(data, H, &numMatches, 10000, 0.0f, maxAmbiguity, 5.0f);   // default seed
    if (numMatches>0)
      ImproveHomographyGPU(data, H, &numFit, 5, 0.0f, maxAmbiguity, 3.0f);
  } else
#endif
  {
    FindHomography(data, H, &numMatches, 10000, 0.0f, maxAmbiguity, 5.0f);
    if (numMatches>0)
      ImproveHomography(data, H, 5, 0.0f, maxAmbiguity, 3.0f);
  }
  r.homographyMs = Ms(t0);
  r.numInliers = numMatches;
  if (numMatches==0)   // FindHomography found nothing (it then returns the identity)
    return;
  double Hd[9];
  for (int i=0;i<9;i++)
    Hd[i] = H[i];
  r.cornerError = CornerError(Hgt, Hd, w, h);
}

struct CudaSiftEval {
  std::vector<Keypoint> kps[2];
  CudaImage img[2];
  cv::Mat fimg[2];
  SiftData data[2];
  float *dbuf[2] = {NULL, NULL};   // device images, reused: CudaImage::Allocate doesn't free
  size_t dsize[2] = {0, 0};        // a previous allocation
  float *tmp = NULL;               // ExtractSift's temporary memory, for images of tmpW x tmpH
  int tmpW = 0, tmpH = 0;
  bool capWarned = false;
  CudaSiftEval() { for (int i=0;i<2;i++) InitSiftData(data[i], compatMode>=0 ? 65536 : 32768, true, true); }
  ~CudaSiftEval()
  {
    for (int i=0;i<2;i++) {
      FreeSiftData(data[i]);
      if (dbuf[i])
        cudaFree(dbuf[i]);
    }
    if (tmp)
      FreeSiftTempMemory(tmp);
  }
  double Extract(int i, const cv::Mat &gray)
  {
    gray.convertTo(fimg[i], CV_32FC1);
    const int pitch = iAlignUp(fimg[i].cols, 128);
    const size_t bytes = sizeof(float)*pitch*fimg[i].rows;
    if (bytes>dsize[i]) {
      if (dbuf[i])
        cudaFree(dbuf[i]);
      cudaMalloc((void **)&dbuf[i], bytes);
      dsize[i] = bytes;
    }
    img[i].Allocate(fimg[i].cols, fimg[i].rows, pitch, false, dbuf[i], (float *)fimg[i].data);
    img[i].Download();
    if (compatMode<0 && (fimg[i].cols!=tmpW || fimg[i].rows!=tmpH)) {   // allocated outside the timing
      if (tmp)
        FreeSiftTempMemory(tmp);
      tmp = AllocSiftTempMemory(fimg[i].cols, fimg[i].rows, 5, false);
      tmpW = fimg[i].cols;
      tmpH = fimg[i].rows;
    }
    auto t0 = std::chrono::steady_clock::now();
#ifdef EVAL_HAS_COMPAT
    if (compatMode>=0)
    {
      SiftCompatParams p = (compatMode==1 ? SiftCompatParams::VLFeat() :
                            compatMode==2 ? SiftCompatParams::COLMAP() : SiftCompatParams::OpenCV());
      if (compatFirstOctave>-2)
        p.firstOctave = compatFirstOctave;
      if (compatExtendFirstOctave) {
        p.firstOctave = 0;
        p.extendFirstOctave = true;
        p.extendFirstOctaveScales = compatExtendScales;
      }
      if (compatContrastThreshold>=0)
        p.contrastThreshold = compatContrastThreshold;
      p.doublePrecision = compatDouble;
      ExtractSiftCompat(data[i], img[i], p);
    }
    else
#endif
      ExtractSift(data[i], img[i], 5, initBlur, dogThresh, 0.0f, false, tmp);
    double ms = Ms(t0);
    if (data[i].numPts>=data[i].maxPts && !capWarned) {
      printf("Warning: %d points, SiftData's capacity: the points kept vary between runs\n", data[i].numPts);
      capWarned = true;
    }
    // The point order varies between runs (atomic counters), and FindHomography samples
    // the matches in that order: sort the points to make the results reproducible
    SiftPoint *pts = HostPoints(data[i]);
    std::sort(pts, pts + data[i].numPts, [](const SiftPoint &a, const SiftPoint &b) {
      if (a.ypos!=b.ypos) return a.ypos<b.ypos;
      if (a.xpos!=b.xpos) return a.xpos<b.xpos;
      if (a.scale!=b.scale) return a.scale<b.scale;
      return a.orientation<b.orientation;
    });
    UploadPoints(data[i]);
    kps[i].clear();
    for (int j=0;j<data[i].numPts;j++)
      kps[i].push_back({pts[j].xpos, pts[j].ypos, pts[j].scale, std::fabs(pts[j].sharpness)});
    return ms;
  }
  void Pair(PairResult &r, const double Hgt[9], int w, int h)
  {
    r.numPts1 = data[0].numPts;
    r.numPts2 = data[1].numPts;
    auto t0 = std::chrono::steady_clock::now();
    SiftPoint *pts0 = HostPoints(data[0]), *pts1 = HostPoints(data[1]);
    if (data[1].numPts>0)
      MatchSiftData(data[0], data[1]);
    else   // MatchSiftData would leave the previous pair's matches
      for (int i=0;i<data[0].numPts;i++)
        pts0[i].match = -1;
    r.matchMs = Ms(t0);
    // CudaSift's ambiguity is the ratio of the second best to the best correlation. For the
    // same ratio test as OpenCV, replace it by Lowe's ratio of the descriptor distances
    // (unit descriptors: squared distance = 2 - 2*correlation); FindHomography and
    // ImproveHomography then use it too.
    std::vector<cv::Point2f> p1, p2;
    std::vector<float> s1, s2;
    for (int i=0;i<data[0].numPts;i++) {
      SiftPoint &p = pts0[i];
      p.ambiguity = LoweRatio(p.score, p.ambiguity);
      if (p.ambiguity<maxAmbiguity && p.match>=0) {
        p1.push_back(cv::Point2f(p.xpos, p.ypos));
        p2.push_back(cv::Point2f(p.match_xpos, p.match_ypos));
        s1.push_back(p.scale);
        s2.push_back(pts1[p.match].scale);
      }
    }
    CountCorrect(r, Hgt, p1, p2, s1, s2);
    UploadPoints(data[0]);   // with Lowe's ratios
    r.cornerError = 1e9;
    if (r.numMatches>=4) {
      if (estimator==EST_OPENCV)
        OpenCVHomography(r, Hgt, w, h, p1, p2);
      else if (estimator==EST_ORACLE)
        OracleHomography(r, Hgt, w, h, p1, p2);
      else
        CudaSiftHomography(r, Hgt, w, h, data[0]);
    }
  }
};

// OpenCV doubles the image before the first octave. Before 4.8 (and with
// enable_precise_upscale false, the default of SIFT::create), the doubled image is
// pixel-centre aligned while the positions are then halved, which shifts every keypoint by
// +0.25 px. eval_hpatches uses the precise upscale (x -> 2x) where available, unless
// --opencv-legacy-upscale.
// cv::SIFT is in features2d from OpenCV 4.4 (before, in opencv_contrib): without it, no --opencv.
#define OPENCV_HAS_PRECISE_UPSCALE (CV_VERSION_MAJOR>4 || (CV_VERSION_MAJOR==4 && CV_VERSION_MINOR>=8))
#define OPENCV_HAS_SIFT (CV_VERSION_MAJOR>4 || (CV_VERSION_MAJOR==4 && CV_VERSION_MINOR>=4))
static bool opencvLegacyUpscale = false;

static cv::Ptr<cv::Feature2D> CreateSIFT()
{
#if OPENCV_HAS_PRECISE_UPSCALE
  return cv::SIFT::create(0, 3, 0.04, 10, 1.6, !opencvLegacyUpscale);
#elif OPENCV_HAS_SIFT
  return cv::SIFT::create();
#else
  return cv::Ptr<cv::Feature2D>();
#endif
}

// CPU references: OpenCV's SIFT, VLFeat's SIFT with its defaults, or VLFeat's SIFT with
// COLMAP's settings (first octave -1 i.e. the image doubled, 4 octaves, 3 levels, peak
// threshold 0.02/3 on images in [0, 1], edge threshold 10, at most 2 orientations, whole
// DoG levels from the coarsest until they have more than 8192 detections, and RootSIFT
// descriptors, L1-normalized then square roots, rounded to 8 bits; as COLMAP's sift.cc, but
// without its +0.5 shift of the positions)
enum CpuSift { CPU_OPENCV, CPU_VLFEAT, CPU_VLFEAT_COLMAP };
static CpuSift cpuSift = CPU_OPENCV;

#ifdef HAVE_VLFEAT
static void VlFeatExtract(const cv::Mat &gray, bool colmap, std::vector<cv::KeyPoint> &kp, cv::Mat &desc)
{
  cv::Mat img;
  // As COLMAP converts the image for VLFeat (x/255.0f; convertTo multiplies by the scale, which
  // rounds differently)
  img.create(gray.size(), CV_32F);
  for (int y=0;y<gray.rows;y++)
    for (int x=0;x<gray.cols;x++)
      img.at<float>(y, x) = gray.at<uchar>(y, x)/255.0f;
  VlSiftFilt *f = vl_sift_new(img.cols, img.rows, colmap ? 4 : -1, 3, colmap ? -1 : 0);
  if (colmap) {
    vl_sift_set_peak_thresh(f, 0.02/3);
    vl_sift_set_edge_thresh(f, 10.0);
  }
  struct Feature { cv::KeyPoint k; std::vector<float> d; };
  std::vector<std::vector<Feature>> levels;   // per DoG level
  std::vector<int> levelDetections;
  int err = vl_sift_process_first_octave(f, (const float *)img.data);
  while (err!=VL_ERR_EOF) {
    vl_sift_detect(f);
    const VlSiftKeypoint *keys = vl_sift_get_keypoints(f);
    const int nk = vl_sift_get_nkeypoints(f);
    int prevLevel = -1000;
    for (int i=0;i<nk;i++) {
      if (keys[i].is!=prevLevel) {
        levels.emplace_back();
        levelDetections.push_back(0);
        prevLevel = keys[i].is;
      }
      levelDetections.back()++;
      double angles[4];
      int na = vl_sift_calc_keypoint_orientations(f, angles, &keys[i]);
      if (colmap)
        na = std::min(na, 2);
      for (int j=0;j<na;j++) {
        Feature ft;
        ft.d.resize(128);
        vl_sift_calc_keypoint_descriptor(f, ft.d.data(), &keys[i], angles[j]);
        // Response: the DoG value at the detection (VLFeat doesn't keep the interpolated one)
        const int ow = vl_sift_get_octave_width(f);
        const float dog = vl_sift_get_octave(f, keys[i].is + 1)[keys[i].iy*ow + keys[i].ix] -
                          vl_sift_get_octave(f, keys[i].is)[keys[i].iy*ow + keys[i].ix];
        ft.k = cv::KeyPoint(keys[i].x, keys[i].y, 2.0f*keys[i].sigma, (float)(angles[j]*180.0/CV_PI), std::fabs(dog));
        levels.back().push_back(ft);
      }
    }
    err = vl_sift_process_next_octave(f);
  }
  vl_sift_delete(f);
  // COLMAP (sift.cc, VLFeat path): whole DoG levels, from the coarsest, until they have more
  // than 8192 detections; L1-root descriptors rounded to 8 bits
  size_t first = 0;
  if (colmap) {
    int num = 0;
    for (int l=(int)levels.size()-1;l>=0;l--) {
      num += levelDetections[l];
      if (num>8192) {
        first = l;
        break;
      }
    }
  }
  std::vector<Feature> feats;
  for (size_t l=first;l<levels.size();l++)
    feats.insert(feats.end(), levels[l].begin(), levels[l].end());
  if (colmap) {
    for (Feature &ft : feats) {
      double l1 = 0.0;
      for (float v : ft.d)
        l1 += std::fabs(v);
      for (float &v : ft.d)
        v = std::min(255.0f, std::round(512.0f*(float)std::sqrt(v*(1.0f/(float)l1))));
    }
  }
  kp.clear();
  desc.create((int)feats.size(), 128, CV_32F);
  for (size_t i=0;i<feats.size();i++) {
    kp.push_back(feats[i].k);
    memcpy(desc.ptr<float>((int)i), feats[i].d.data(), 128*sizeof(float));
  }
}
#endif

static bool gpuMatching = false;   // --gpu-matching: CPU references matched by MatchSiftData

struct OpenCVEval {
  std::vector<Keypoint> kps[2];
  cv::Ptr<cv::Feature2D> sift = CreateSIFT();
  std::vector<cv::KeyPoint> kp[2];
  cv::Mat desc[2];
  SiftData gdata[2];   // --gpu-matching
  int gcap[2] = {0, 0};
  bool refChanged = true;   // gdata[0] to convert again
  SiftData hdata;           // the matches, for CudaSift's estimators
  int hcap = 0;
  ~OpenCVEval()
  {
    for (int i=0;i<2;i++)
      if (gcap[i])
        FreeSiftData(gdata[i]);
    if (hcap)
      FreeSiftData(hdata);
  }
  // The features of image i as a SiftData, with unit descriptors
  void ToSiftData(int i)
  {
    const int n = (int)kp[i].size();
    if (n>gcap[i]) {
      if (gcap[i])
        FreeSiftData(gdata[i]);
      InitSiftData(gdata[i], n, true, true);
      gcap[i] = n;
    }
    gdata[i].numPts = n;
    SiftPoint *pts = HostPoints(gdata[i]);
    for (int j=0;j<n;j++) {
      SiftPoint &p = pts[j];
      memset(&p, 0, sizeof(p));
      p.xpos = kp[i][j].pt.x;
      p.ypos = kp[i][j].pt.y;
      p.scale = 0.5f*kp[i][j].size;
      const float *d = desc[i].ptr<float>(j);
      double nrm = 0.0;
      for (int k=0;k<128;k++)
        nrm += (double)d[k]*d[k];
      const float inv = (float)(1.0/std::sqrt(std::max(nrm, 1e-12)));
      for (int k=0;k<128;k++)
        p.data[k] = d[k]*inv;
    }
    UploadPoints(gdata[i]);
  }
  double Extract(int i, const cv::Mat &gray)
  {
    auto t0 = std::chrono::steady_clock::now();
#ifdef HAVE_VLFEAT
    if (cpuSift!=CPU_OPENCV)
      VlFeatExtract(gray, cpuSift==CPU_VLFEAT_COLMAP, kp[i], desc[i]);
    else
#endif
      sift->detectAndCompute(gray, cv::noArray(), kp[i], desc[i]);
    double ms = Ms(t0);
    if (i==0)
      refChanged = true;
    kps[i].clear();
    for (const cv::KeyPoint &k : kp[i])
      kps[i].push_back({k.pt.x, k.pt.y, 0.5f*k.size, std::fabs(k.response)});   // size: diameter, 2 sigma
    return ms;
  }
  void Pair(PairResult &r, const double Hgt[9], int w, int h)
  {
    r.numPts1 = (int)kp[0].size();
    r.numPts2 = (int)kp[1].size();
    std::vector<cv::Point2f> p1, p2;
    std::vector<float> s1, s2;
    auto t0 = std::chrono::steady_clock::now();
    if (gpuMatching && !desc[0].empty() && desc[1].rows>=2) {
      // MatchSiftData, then Lowe's ratio of the descriptor distances (unit descriptors:
      // squared distance = 2 - 2*correlation)
      if (refChanged) {
        ToSiftData(0);
        refChanged = false;
      }
      ToSiftData(1);
      MatchSiftData(gdata[0], gdata[1]);
      const SiftPoint *pts0 = HostPoints(gdata[0]);
      for (int i=0;i<gdata[0].numPts;i++) {
        const SiftPoint &p = pts0[i];
        float ratio = LoweRatio(p.score, p.ambiguity);
        if (ratio<maxAmbiguity && p.match>=0) {
          p1.push_back(kp[0][i].pt);
          p2.push_back(kp[1][p.match].pt);
          s1.push_back(0.5f*kp[0][i].size);
          s2.push_back(0.5f*kp[1][p.match].size);
        }
      }
    } else if (!desc[0].empty() && desc[1].rows>=2) {
      std::vector<std::vector<cv::DMatch>> knn;
      cv::BFMatcher(cv::NORM_L2).knnMatch(desc[0], desc[1], knn, 2);
      for (auto &m : knn)
        if (m.size()==2 && m[0].distance<maxAmbiguity*m[1].distance) {
          p1.push_back(kp[0][m[0].queryIdx].pt);
          p2.push_back(kp[1][m[0].trainIdx].pt);
          s1.push_back(0.5f*kp[0][m[0].queryIdx].size);
          s2.push_back(0.5f*kp[1][m[0].trainIdx].size);
        }
    }
    r.matchMs = Ms(t0);
    CountCorrect(r, Hgt, p1, p2, s1, s2);
    r.cornerError = 1e9;
    if (r.numMatches>=4) {
      if (estimator==EST_DEFAULT || estimator==EST_OPENCV)
        OpenCVHomography(r, Hgt, w, h, p1, p2);
      else if (estimator==EST_ORACLE)
        OracleHomography(r, Hgt, w, h, p1, p2);
      else {
        // The matches as a SiftData, all passing the score and ambiguity limits
        if (r.numMatches>hcap) {
          if (hcap)
            FreeSiftData(hdata);
          InitSiftData(hdata, r.numMatches, true, true);
          hcap = r.numMatches;
        }
        SiftPoint *pts = HostPoints(hdata);
        for (int i=0;i<r.numMatches;i++) {
          SiftPoint &p = pts[i];
          memset(&p, 0, sizeof(p));
          p.xpos = p1[i].x;
          p.ypos = p1[i].y;
          p.match_xpos = p2[i].x;
          p.match_ypos = p2[i].y;
          p.score = 1.0f;
          p.ambiguity = 0.0f;
        }
        hdata.numPts = r.numMatches;
        UploadPoints(hdata);
        CudaSiftHomography(r, Hgt, w, h, hdata);
      }
    }
  }
};

// Image k (1..6) of a sequence: HPatches (k.ppm) or synthetic (k.png)
static cv::Mat ReadImage(const std::string &seqDir, int k)
{
  for (const std::string &name : {std::to_string(k) + ".ppm", std::to_string(k) + ".png"}) {
    if (!std::filesystem::exists(seqDir + "/" + name))
      continue;
    cv::Mat m = cv::imread(seqDir + "/" + name, cv::IMREAD_GRAYSCALE);
    if (!m.empty())
      return m;
  }
  return cv::Mat();
}

// Homography from image 1 to image k
static bool ReadSeqH(const std::string &seqDir, int k, double H[9])
{
  return ReadH(seqDir + "/H_1_" + std::to_string(k), H);
}

// A pair to evaluate: the target image and the homography from the reference
struct Target {
  int k;
  cv::Mat img;
  double H[9];
};

// Synthetic pairs (--make-synthetic), following the synthetic tests of Lowe's paper
// (Distinctive Image Features from Scale-Invariant Keypoints, IJCV 2004, sections 3 and
// 6.3: "rotation, scaling, affine stretch, change in brightness and contrast, and addition
// of image noise"). Per pair, on canvases of the source's size:
// - reference: the source rotated about its centre by a random angle;
// - target: the source rotated by another random angle, scaled by U(0.2, 0.9) (Lowe's
//   range), and stretched by a planar tilt of U(0, 50) degrees (a scale of cos(tilt) along
//   a random direction; Lowe's range for affine change);
// - both: a gain 2^U(-0.5, 0.5) and an offset of U(-20, 20) grey levels, independently,
//   then uniform noise of +-1% (Lowe's), rounded to 8 bits.
// Both images are rendered the same way, so that they have the same resampling blur: the
// source is upsampled 4x (bicubic), blurred by an anisotropic Gaussian that keeps the blur of
// the input at 0.5 target pixel along the reduced directions, warped on the 4x grid
// (bilinear) and reduced 4x by averaging. Pixel centres are at integer coordinates, as in
// CudaSift and OpenCV. The random values come from std::mt19937 (whose output the standard
// defines), seeded with 1000 + the index of the source image, so the pairs are the same on
// every platform. The parameters of each pair are written to its params file.

// Uniform in [0, 1) from the 32-bit output of mt19937, the same on every platform
static double Uniform(std::mt19937 &rng)
{
  return (rng() >> 8)*(1.0/16777216.0);
}

// Renders the source, upsampled 4x (src4), through the affine map A (source to image,
// 2x3, in pixels of the source), with gain, offset and noise
static cv::Mat Render(const cv::Mat &src4, cv::Size size, const cv::Matx23d &A, double gain, double offset, std::mt19937 &rng)
{
  const double F = 4.0, o = 0.5*(F - 1.0);   // 4x grid: x4 = F*x + o
  // Anti-aliasing: along each principal direction v of the source with scale s < 1, a
  // Gaussian of variance F^2 (0.25 - 0.25 s^2)/s^2 (pixels of the 4x source), so that the
  // reduced image keeps a blur of 0.5 pixel
  cv::Matx22d L(A(0, 0), A(0, 1), A(1, 0), A(1, 1));
  cv::Mat w, u, vt;
  cv::SVD::compute(cv::Mat(L), w, u, vt);
  cv::Matx22d C(0, 0, 0, 0);   // covariance of the blur, 4x source pixels
  double maxSigma = 0.0;
  for (int i=0;i<2;i++) {
    double sc = w.at<double>(i);
    if (sc>=1.0)
      continue;
    double var = F*F*0.25*(1.0 - sc*sc)/(sc*sc);
    cv::Vec2d v(vt.at<double>(i, 0), vt.at<double>(i, 1));
    C += var*(v*v.t());
    maxSigma = std::max(maxSigma, std::sqrt(var));
  }
  cv::Mat blurred = src4;   // shares src4's pixels: filter into a new image below
  if (maxSigma>0.0) {
    const int r = (int)std::ceil(3.0*maxSigma);
    cv::Matx22d Ci = (C + cv::Matx22d(1e-6, 0, 0, 1e-6)).inv();
    cv::Mat k(2*r + 1, 2*r + 1, CV_32F);
    double sum = 0.0;
    for (int y=-r;y<=r;y++)
      for (int x=-r;x<=r;x++) {
        double e = std::exp(-0.5*(x*x*Ci(0, 0) + 2*x*y*Ci(0, 1) + y*y*Ci(1, 1)));
        k.at<float>(y + r, x + r) = (float)e;
        sum += e;
      }
    k /= sum;
    blurred = cv::Mat();   // not in place: src4 is used for the other images
    cv::filter2D(src4, blurred, -1, k, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
  }
  cv::Matx23d A4 = A;   // T A T^-1, with T(x) = F*x + o
  A4(0, 2) = F*A(0, 2) + o - o*(A(0, 0) + A(0, 1));
  A4(1, 2) = F*A(1, 2) + o - o*(A(1, 0) + A(1, 1));
  cv::Mat w4, out;
  cv::warpAffine(blurred, w4, cv::Mat(A4), cv::Size(4*size.width, 4*size.height), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
  cv::resize(w4, out, size, 0, 0, cv::INTER_AREA);
  cv::Mat out8(size, CV_8U);
  for (int y=0;y<size.height;y++)
    for (int x=0;x<size.width;x++) {
      double v = gain*out.at<float>(y, x) + offset + 2.55*(2.0*Uniform(rng) - 1.0);
      out8.at<uchar>(y, x) = (uchar)std::min(std::max(std::lround(v), 0L), 255L);
    }
  return out8;
}

// Rotation by deg about c, then the linear map M (about c too), as a 2x3 affine map
static cv::Matx23d AboutCentre(cv::Point2d c, const cv::Matx22d &M)
{
  cv::Vec2d t = cv::Vec2d(c.x, c.y) - M*cv::Vec2d(c.x, c.y);
  return cv::Matx23d(M(0, 0), M(0, 1), t[0], M(1, 0), M(1, 1), t[1]);
}

static cv::Matx22d Rot(double deg)
{
  double a = deg*CV_PI/180.0;
  return cv::Matx22d(std::cos(a), -std::sin(a), std::sin(a), std::cos(a));
}

// Writes pairsPerImage pairs per source image to outDir/x_<name>_<nn>/{1.png, 2.png, H_1_2, params}
static int MakeSynthetic(const std::string &outDir, const std::vector<std::string> &files, int pairsPerImage)
{
  for (size_t f=0;f<files.size();f++) {
    cv::Mat src = cv::imread(files[f], cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
      printf("Cannot read %s\n", files[f].c_str());
      return 1;
    }
    cv::Mat srcf, src4;
    src.convertTo(srcf, CV_32F);
    cv::resize(srcf, src4, cv::Size(4*src.cols, 4*src.rows), 0, 0, cv::INTER_CUBIC);
    std::mt19937 rng(1000 + (unsigned)f);
    const std::string name = std::filesystem::path(files[f]).stem().string();
    const cv::Point2d c(0.5*(src.cols - 1), 0.5*(src.rows - 1));
    for (int k=0;k<pairsPerImage;k++) {
      const double deg1 = 360.0*Uniform(rng), deg2 = 360.0*Uniform(rng);
      const double scale = 0.2 + 0.7*Uniform(rng);
      const double tilt = 50.0*Uniform(rng), tiltDir = 180.0*Uniform(rng);
      const double gain1 = std::pow(2.0, Uniform(rng) - 0.5), offset1 = 40.0*Uniform(rng) - 20.0;
      const double gain2 = std::pow(2.0, Uniform(rng) - 0.5), offset2 = 40.0*Uniform(rng) - 20.0;
      // Stretch: scale cos(tilt) along the direction tiltDir
      const cv::Matx22d S = Rot(tiltDir)*cv::Matx22d(std::cos(tilt*CV_PI/180.0), 0, 0, 1)*Rot(-tiltDir);
      const cv::Matx23d A1 = AboutCentre(c, Rot(deg1));
      const cv::Matx23d A2 = AboutCentre(c, Rot(deg2)*S*scale);
      cv::Mat img1 = Render(src4, src.size(), A1, gain1, offset1, rng);
      cv::Mat img2 = Render(src4, src.size(), A2, gain2, offset2, rng);
      cv::Matx33d H1(A1(0, 0), A1(0, 1), A1(0, 2), A1(1, 0), A1(1, 1), A1(1, 2), 0, 0, 1);
      cv::Matx33d H2(A2(0, 0), A2(0, 1), A2(0, 2), A2(1, 0), A2(1, 1), A2(1, 2), 0, 0, 1);
      cv::Matx33d H = H2*H1.inv();
      char seq[64];
      snprintf(seq, sizeof(seq), "x_%s_%02d", name.c_str(), k);
      const std::string dir = outDir + "/" + seq;
      std::filesystem::create_directories(dir);
      cv::imwrite(dir + "/1.png", img1);
      cv::imwrite(dir + "/2.png", img2);
      FILE *fh = fopen((dir + "/H_1_2").c_str(), "w");
      for (int i=0;i<3;i++)
        fprintf(fh, "%.12g %.12g %.12g\n", H(i, 0), H(i, 1), H(i, 2));
      fclose(fh);
      FILE *fp = fopen((dir + "/params").c_str(), "w");
      fprintf(fp, "rotation1 %.4f\nrotation2 %.4f\nscale %.4f\ntilt %.4f\ntilt_direction %.4f\ngain1 %.4f\noffset1 %.4f\ngain2 %.4f\noffset2 %.4f\n",
              deg1, deg2, scale, tilt, tiltDir, gain1, offset1, gain2, offset2);
      fclose(fp);
    }
    printf("%s: %d pairs\n", files[f].c_str(), pairsPerImage);
  }
  return 0;
}

// --profile: wall time of each phase over the whole run (ms)
static bool profile = false;
static int jobs = 1;   // --jobs: CPU references, sequences in parallel
static double profRead = 0, profExtract = 0, profPair = 0, profRepeat = 0;
static std::mutex profMutex;   // --jobs: several threads add to the totals
static void ProfAdd(double &total, std::chrono::steady_clock::time_point t0)
{
  double ms = Ms(t0);
  std::lock_guard<std::mutex> lock(profMutex);
  total += ms;
}

template <class Eval>
static void RunPairs(Eval &eval, const std::string &seq, const cv::Mat &ref, const std::vector<Target> &targets, std::vector<PairResult> &results)
{
  auto t0 = std::chrono::steady_clock::now();
  eval.Extract(0, ref);
  ProfAdd(profExtract, t0);
  std::vector<Keypoint> refKps = eval.kps[0];
  for (const Target &t : targets) {
    PairResult r;
    r.seq = seq;
    r.k = t.k;
    r.homographyMs = 0.0;
    r.numInliers = 0;
    t0 = std::chrono::steady_clock::now();
    r.extractMs = eval.Extract(1, t.img);
    ProfAdd(profExtract, t0);
    t0 = std::chrono::steady_clock::now();
    eval.Pair(r, t.H, ref.cols, ref.rows);
    ProfAdd(profPair, t0);
    t0 = std::chrono::steady_clock::now();
    Repeatability(r, t.H, refKps, eval.kps[1], ref.cols, ref.rows, t.img.cols, t.img.rows);
    ProfAdd(profRepeat, t0);
    results.push_back(r);
  }
}

// The images and homographies of a sequence
struct SequenceData {
  cv::Mat ref;
  std::vector<Target> targets;
};

static SequenceData ReadSequence(const std::string &seqDir)
{
  SequenceData d;
  d.ref = ReadImage(seqDir, 1);
  if (d.ref.empty())
    return d;
  for (int k=2;k<=6;k++) {
    Target t;
    t.k = k;
    t.img = ReadImage(seqDir, k);
    if (!t.img.empty() && ReadSeqH(seqDir, k, t.H))
      d.targets.push_back(t);
  }
  return d;
}

// The pairs of the sequences, in order; the next sequence is read while one is evaluated
template <class Eval>
static std::vector<PairResult> Run(Eval &eval, const std::string &dir, const std::vector<std::string> &seqs)
{
  std::vector<PairResult> results;
  std::future<SequenceData> next;
  if (!seqs.empty())
    next = std::async(std::launch::async, ReadSequence, dir + "/" + seqs[0]);
  for (size_t s=0;s<seqs.size();s++) {
    auto t0 = std::chrono::steady_clock::now();
    SequenceData d = next.get();
    ProfAdd(profRead, t0);   // the wait for the reading only
    if (s + 1<seqs.size())
      next = std::async(std::launch::async, ReadSequence, dir + "/" + seqs[s + 1]);
    if (!d.ref.empty())
      RunPairs(eval, seqs[s], d.ref, d.targets, results);
  }
  return results;
}

// The pairs of a summary: all (type 0, seq empty), those whose sequence starts with type, or
// those of sequence seq
static bool InSummary(const PairResult &r, char type, const std::string &seq)
{
  return (seq.empty() ? (!type || r.seq[0]==type) : r.seq==seq);
}

static void Summary(const char *name, const std::vector<PairResult> &all, char type, const std::string &seq = "")
{
  int n = 0;
  double pts = 0, matches = 0, mma[11] = {0}, ext = 0, mat = 0, hom = 0;
  int homOk[6] = {0};
  for (const PairResult &r : all) {
    if (!InSummary(r, type, seq))
      continue;
    n++;
    pts += 0.5*(r.numPts1 + r.numPts2);
    matches += r.numMatches;
    for (int t=1;t<=10;t++)
      mma[t] += (r.numMatches ? (double)r.correct[t]/r.numMatches : 0.0);
    for (int t : {1, 3, 5})
      homOk[t] += (r.cornerError<=t);
    ext += r.extractMs;
    mat += r.matchMs;
    hom += r.homographyMs;
  }
  if (!n)
    return;
  std::vector<float> err;
  for (const PairResult &r : all)
    if (InSummary(r, type, seq))
      err.insert(err.end(), r.err3.begin(), r.err3.end());
  std::sort(err.begin(), err.end());
  double mean = 0.0;
  for (float e : err)
    mean += e;
  auto Q = [&](double f) { return err.empty() ? 0.0 : (double)err[std::min(err.size() - 1, (size_t)(f*err.size()))]; };
  printf("%-13s %4d pairs, %6.0f features, %6.0f matches | MMA@1/3/5/10 px %5.3f %5.3f %5.3f %5.3f | homography@1/3/5 px %5.3f %5.3f %5.3f | ms: extract %.2f, match %.2f, homography %.2f, total %.2f\n",
         name, n, pts/n, matches/n, mma[1]/n, mma[3]/n, mma[5]/n, mma[10]/n,
         (double)homOk[1]/n, (double)homOk[3]/n, (double)homOk[5]/n, ext/n, mat/n, hom/n, (ext + mat + hom)/n);
  printf("%-13s matches within 3 px: %.0f per pair, error mean %.3f, p25 %.3f, median %.3f, p75 %.3f, p90 %.3f px\n",
         "", (double)err.size()/n, err.empty() ? 0.0 : mean/err.size(), Q(0.25), Q(0.5), Q(0.75), Q(0.9));
  // Repeatability and matching score (means over the pairs), scale errors (pooled)
  double rep = 0.0, posRep = 0.0, mscore = 0.0, chance = 0.0, corrected = 0.0, budget[3] = {0, 0, 0};
  int budgetN[3] = {0, 0, 0};
  std::vector<double> curve;
  std::vector<float> se;
  for (const PairResult &r : all)
    if (InSummary(r, type, seq)) {
      rep += (r.numOverlap ? (double)r.numRepeated/r.numOverlap : 0.0);
      posRep += (r.numOverlap ? (double)r.numPosRepeated/r.numOverlap : 0.0);
      mscore += (r.numOverlap ? std::min(1.0, (double)r.correct[3]/r.numOverlap) : 0.0);
      const double R = (r.numOverlap ? (double)r.numRepeated/r.numOverlap : 0.0);
      chance += r.chanceRepeat;
      corrected += (r.chanceRepeat<1.0 ? (R - r.chanceRepeat)/(1.0 - r.chanceRepeat) : 0.0);
      for (int b=0;b<3;b++)
        if (r.budgetRepeat[b]>=0) {
          budget[b] += r.budgetRepeat[b];
          budgetN[b]++;
        }
      if (curve.size()<r.curve.size())
        curve.resize(r.curve.size(), 0.0);
      for (size_t m=0;m<r.curve.size();m++)
        curve[m] += r.curve[m];
      se.insert(se.end(), r.scaleErr.begin(), r.scaleErr.end());
    }
  double sm = 0.0, ss = 0.0;
  for (float e : se) {
    sm += e;
    ss += e*e;
  }
  std::vector<float> ae;
  for (float e : se)
    ae.push_back(std::fabs(e));
  std::sort(ae.begin(), ae.end());
  const double m = (se.empty() ? 0.0 : sm/se.size());
  printf("%-13s repeatability %.3f (position only %.3f), matching score %.3f; matches within 3 px, log2(scale / expected scale): mean %+.3f, std %.3f, median |.| %.3f\n",
         "", rep/n, posRep/n, mscore/n, m, se.empty() ? 0.0 : std::sqrt(std::max(ss/se.size() - m*m, 0.0)), ae.empty() ? 0.0 : ae[ae.size()/2]);
  printf("%-13s chance-corrected repeatability %.3f (chance %.3f, moves of max(%.1f px, %.1f sigma)); strongest", "", corrected/n, chance/n, 2*repeatPx, chanceK);
  for (int b=0;b<3;b++)
    printf(" %d: %.3f (%d pairs)%s", budgets[b], budgetN[b] ? budget[b]/budgetN[b] : 0.0, budgetN[b], b<2 ? "," : "\n");
  if (!curve.empty()) {
    printf("%-13s repeatability with the projections moved:", "");
    for (size_t k=0;k<curve.size();k++)
      printf(curveMoves[k]>0 ? " %.2f px %.3f" : " %.0f sigma %.3f", std::fabs(curveMoves[k]), curve[k]/n);
    printf("\n");
  }
}

int main(int argc, char **argv)
{
  if (argc<2) {
    printf("Usage: %s <dataset dir> [--opencv] [--opencv-legacy-upscale] [--vlfeat] [--vlfeat-colmap] [--compat opencv|vlfeat|colmap [--first-octave n] [--extend-first-octave [--extend-scales n]] [--contrast-threshold t] [--double]] [--gpu-homography] [--opencv-homography] [--cudasift-homography] [--oracle px] [--init-blur sigma] [--thresh t] [--repeat-curve] [--chance-k k] [--profile] [--jobs n] [--gpu-matching] [--per-sequence] [--skip sequence]... [--csv file] [device]\n"
           "       %s <output dir> --make-synthetic image,image,...\n", argv[0], argv[0]);
    return 1;
  }
  std::string dir = argv[1], csv;
  std::vector<std::string> skip;
  bool perSeq = false;
  bool opencv = false;
  int device = 0;
  for (int i=2;i<argc;i++) {
    if (!strcmp(argv[i], "--opencv"))
      opencv = true;
    else if (!strcmp(argv[i], "--vlfeat") || !strcmp(argv[i], "--vlfeat-colmap")) {
#ifndef HAVE_VLFEAT
      printf("This build has no VLFeat (configure with VLFEAT_DIR, see the README)\n");
      return 1;
#endif
      opencv = true;   // the CPU reference path
      cpuSift = (!strcmp(argv[i], "--vlfeat") ? CPU_VLFEAT : CPU_VLFEAT_COLMAP);
    }
    else if (!strcmp(argv[i], "--compat") && i+1<argc) {
#ifndef EVAL_HAS_COMPAT
      printf("This build has no ExtractSiftCompat\n");
      return 1;
#endif
      i++;
      compatMode = (!strcmp(argv[i], "opencv") ? 0 : !strcmp(argv[i], "vlfeat") ? 1 : !strcmp(argv[i], "colmap") ? 2 : -1);
      if (compatMode<0) {
        printf("--compat: opencv, vlfeat or colmap, not %s\n", argv[i]);
        return 1;
      }
    }
    else if (!strcmp(argv[i], "--first-octave") && i+1<argc)
      compatFirstOctave = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--extend-first-octave"))
      compatExtendFirstOctave = true;
    else if (!strcmp(argv[i], "--extend-scales") && i+1<argc)
      compatExtendScales = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--double"))
      compatDouble = true;
    else if (!strcmp(argv[i], "--contrast-threshold") && i+1<argc)
      compatContrastThreshold = (float)atof(argv[++i]);
    else if (!strcmp(argv[i], "--gpu-homography")) {
#ifndef CUDASIFT_HAS_GPU_HOMOGRAPHY
      printf("This version has no FindHomographyGPU and ImproveHomographyGPU\n");
      return 1;
#endif
      estimator = EST_GPU;
    } else if (!strcmp(argv[i], "--opencv-homography"))
      estimator = EST_OPENCV;
    else if (!strcmp(argv[i], "--cudasift-homography"))
      estimator = EST_CUDASIFT;
    else if (!strcmp(argv[i], "--init-blur") && i+1<argc)
      initBlur = (float)atof(argv[++i]);
    else if (!strcmp(argv[i], "--thresh") && i+1<argc)
      dogThresh = (float)atof(argv[++i]);
    else if (!strcmp(argv[i], "--oracle") && i+1<argc) {
      estimator = EST_ORACLE;
      oracleThresh = (float)atof(argv[++i]);
    }
    else if (!strcmp(argv[i], "--skip") && i+1<argc)
      skip.push_back(argv[++i]);
    else if (!strcmp(argv[i], "--make-synthetic") && i+1<argc) {   // image files, comma-separated: write pairs to dir
      std::vector<std::string> files;
      for (char *t = strtok(argv[++i], ","); t; t = strtok(NULL, ","))
        files.push_back(t);
      return MakeSynthetic(dir, files, 40);
    }
    else if (!strcmp(argv[i], "--profile"))
      profile = true;
    else if (!strcmp(argv[i], "--gpu-matching"))
      gpuMatching = true;
    else if (!strcmp(argv[i], "--jobs") && i+1<argc)
      jobs = std::max(1, atoi(argv[++i]));
    else if (!strcmp(argv[i], "--repeat-curve"))
      repeatCurve = true;
    else if (!strcmp(argv[i], "--chance-k") && i+1<argc)
      chanceK = (float)atof(argv[++i]);
    else if (!strcmp(argv[i], "--per-sequence"))
      perSeq = true;
    else if (!strcmp(argv[i], "--opencv-legacy-upscale"))
      opencvLegacyUpscale = true;
    else if (!strcmp(argv[i], "--csv") && i+1<argc)
      csv = argv[++i];
    else if (argv[i][0]!='\0' && strspn(argv[i], "0123456789")==strlen(argv[i]))
      device = atoi(argv[i]);
    else {
      printf("Unknown option, or option without its value: %s\n", argv[i]);
      return 1;
    }
  }
  FILE *csvFile = NULL;
  if (!csv.empty() && !(csvFile = fopen(csv.c_str(), "w"))) {   // before the evaluation
    printf("Cannot write %s\n", csv.c_str());
    return 1;
  }
  // ExtractSiftCompat's parameters: --compat stops above in builds without it
  const bool compatOptions = (compatFirstOctave>-2 || compatExtendFirstOctave || compatExtendScales>=0 || compatDouble || compatContrastThreshold>=0);
  if (compatOptions && compatMode<0) {
    printf("--first-octave, --extend-first-octave, --extend-scales, --double and --contrast-threshold need --compat\n");
    return 1;
  }
  if (compatExtendScales>=0 && !compatExtendFirstOctave) {
    printf("--extend-scales needs --extend-first-octave\n");
    return 1;
  }
  if (compatExtendFirstOctave && compatFirstOctave!=-2 && compatFirstOctave!=0) {
    printf("--extend-first-octave needs first octave 0\n");
    return 1;
  }
  std::vector<std::string> seqs;
  std::error_code ec;
  for (const auto &e : std::filesystem::directory_iterator(dir, ec)) {
    std::string name = e.path().filename().string();
    if (e.is_directory() && !ReadImage(e.path().string(), 1).empty() &&
        std::find(skip.begin(), skip.end(), name)==skip.end())
      seqs.push_back(name);
  }
  std::sort(seqs.begin(), seqs.end());
  if (seqs.empty()) {
    printf("No sequences found in %s\n", dir.c_str());
    return 1;
  }

  std::vector<PairResult> results;
#if !OPENCV_HAS_SIFT
  if (opencv && cpuSift==CPU_OPENCV) {
    printf("Error: OpenCV %s has no cv::SIFT (4.4 or later), so --opencv is not available.\n", CV_VERSION);
    return 1;
  }
#elif !OPENCV_HAS_PRECISE_UPSCALE
  if (opencv && cpuSift==CPU_OPENCV && !opencvLegacyUpscale) {
    printf("Error: OpenCV %s has no precise upscale (4.8 or later), so OpenCV's SIFT keypoints would be shifted by +0.25 px.\n"
           "Build with OpenCV 4.8 or later, or pass --opencv-legacy-upscale to accept the shift.\n", CV_VERSION);
    return 1;
  }
#endif
  if (opencv) {
    if (estimator==EST_CUDASIFT || estimator==EST_GPU || gpuMatching)
      InitCuda(device);
    if (jobs>1 && estimator!=EST_CUDASIFT && estimator!=EST_GPU && !gpuMatching) {
      // Sequences in parallel, each thread with its own extractor; results in sequence order
      printf("--jobs %d: the times per pair are not comparable with single-job runs\n", jobs);
      std::vector<std::vector<PairResult>> part(seqs.size());
      std::atomic<size_t> nextSeq(0);
      std::vector<std::thread> threads;
      for (int t=0;t<jobs;t++)
        threads.emplace_back([&]() {
          OpenCVEval eval;
          for (size_t q; (q = nextSeq++)<seqs.size();) {
            SequenceData d = ReadSequence(dir + "/" + seqs[q]);
            if (!d.ref.empty())
              RunPairs(eval, seqs[q], d.ref, d.targets, part[q]);
          }
        });
      for (std::thread &t : threads)
        t.join();
      for (auto &p : part)
        results.insert(results.end(), p.begin(), p.end());
    } else {
      OpenCVEval eval;
      results = Run(eval, dir, seqs);
    }
  } else {
    InitCuda(device);
    CudaSiftEval eval;
    results = Run(eval, dir, seqs);
  }
  const char *est = (estimator==EST_GPU ? ", GPU homography" : estimator==EST_ORACLE ? ", oracle homography" : estimator==EST_OPENCV && !opencv ? ", OpenCV homography" :
                     estimator==EST_CUDASIFT && opencv ? ", CudaSift homography" : "");
  const char *cpuName = (cpuSift==CPU_VLFEAT ? "VLFeat SIFT" : cpuSift==CPU_VLFEAT_COLMAP ? "VLFeat SIFT, COLMAP settings" :
                         OPENCV_HAS_PRECISE_UPSCALE && !opencvLegacyUpscale ? "OpenCV SIFT (precise upscale)" : "OpenCV SIFT (legacy upscale)");
  const char *gpuName = (compatMode==0 ? "CudaSift, ExtractSiftCompat (OpenCV)" : compatMode==1 ? "CudaSift, ExtractSiftCompat (VLFeat)" :
                         compatMode==2 ? "CudaSift, ExtractSiftCompat (COLMAP)" : "CudaSift");
  printf("\n%s%s, %zu sequences\n", opencv ? cpuName : gpuName, est, seqs.size());
  if (profile)
    printf("Profile (ms, whole run): read %.0f, extract %.0f (incl. upload and download), match and homography %.0f, repeatability %.0f\n",
           profRead, profExtract, profPair, profRepeat);
  Summary("all", results, 0);
  bool hpatches = false;
  for (const PairResult &r : results)
    hpatches |= (r.seq.size()>1 && r.seq[1]=='_' && (r.seq[0]=='i' || r.seq[0]=='v'));
  if (hpatches) {
    Summary("illumination", results, 'i');
    Summary("viewpoint", results, 'v');
  }
  if (perSeq) {
    std::vector<std::string> names;
    for (const PairResult &r : results)
      if (std::find(names.begin(), names.end(), r.seq)==names.end())
        names.push_back(r.seq);
    for (const std::string &name : names)
      Summary(name.c_str(), results, 0, name);
  }

  if (csvFile) {
    FILE *f = csvFile;
    fprintf(f, "seq,k,pts1,pts2,matches,correct1,correct3,correct5,correct10,inliers,corner_error,overlap,pos_repeated,repeated,chance_repeat,repeat500,repeat1000,repeat2000,scale_n,scale_sum,scale_sumsq\n");
    for (const PairResult &r : results) {
      double sum = 0.0, sumsq = 0.0;
      for (float e : r.scaleErr) {
        sum += e;
        sumsq += e*e;
      }
      fprintf(f, "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.4f,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%zu,%.6f,%.6f\n", r.seq.c_str(), r.k, r.numPts1, r.numPts2, r.numMatches,
              r.correct[1], r.correct[3], r.correct[5], r.correct[10], r.numInliers, r.cornerError, r.numOverlap, r.numPosRepeated, r.numRepeated,
              r.chanceRepeat, r.budgetRepeat[0], r.budgetRepeat[1], r.budgetRepeat[2], r.scaleErr.size(), sum, sumsq);
    }
    fclose(f);
  }
  return 0;
}
