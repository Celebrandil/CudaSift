//********************************************************//
// Keypoint-by-keypoint comparison with OpenCV and VLFeat //
//********************************************************//

// Runs a reference SIFT and ExtractSiftCompat (or, with --original, ExtractSift) on the same
// images, and pairs their keypoints. The reference is OpenCV's SIFT (default parameters,
// enable_precise_upscale where available), or with --vlfeat VLFeat's (vl_sift_new defaults),
// or with --colmap VLFeat with COLMAP's settings and post-processing; the last two need
// VLFeat (CMake option VLFEAT_DIR). A reference keypoint is reproduced when a CudaSift
// keypoint lies within a position, scale and angle tolerance. Reports, per image and in
// total, the share of the reference keypoints reproduced (recall) and of CudaSift's keypoints
// that the reference has too (precision) at three tolerances, the position, scale and angle
// errors of the pairs, the L2 distance between their (unit) descriptors and the share of
// identical descriptors, the recall per reference octave, and the extraction times.
//
//   compare_sift [--vlfeat | --colmap] [--double] [--original [--thresh t] [--init-blur sigma] [--angle-offset deg]] image... [device]
//
// --double sets SiftCompatParams::doublePrecision. With --original, CudaSift's features come
// from ExtractSift: --thresh and --init-blur are its parameters; --angle-offset is added to its
// orientations before comparing (it uses another angle convention). Positions are in pixels
// of the input image, with pixel centres at integer coordinates in all of them.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>

#include <cuda_runtime.h>
#include "cudaImage.h"
#include "cudaSift.h"
#include "cudaSiftCompat.h"
#ifdef HAVE_VLFEAT
extern "C" {
#include <vl/sift.h>
}
#endif

struct Feature {
  float x, y, sigma, angle;   // angle in degrees, [0, 360)
  int octave;                 // -1 = the doubled image
  float desc[128];            // unit L2
};

enum Reference { REF_OPENCV, REF_VLFEAT, REF_COLMAP };

// Copies a descriptor to unit L2 norm
static void Normalize(float *dst, const float *src)
{
  double n = 0.0;
  for (int k=0;k<128;k++)
    n += (double)src[k]*src[k];
  for (int k=0;k<128;k++)
    dst[k] = (float)(src[k]/std::sqrt(std::max(n, 1e-12)));
}

static double AngleDiff(double a, double b)
{
  double d = std::fmod(std::fabs(a - b), 360.0);
  return std::min(d, 360.0 - d);
}

static std::vector<Feature> OpenCVFeatures(const cv::Mat &gray)
{
#if CV_VERSION_MAJOR>4 || (CV_VERSION_MAJOR==4 && CV_VERSION_MINOR>=8)
  cv::Ptr<cv::SIFT> sift = cv::SIFT::create(0, 3, 0.04, 10, 1.6, true);
#else
  cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
  printf("Warning: OpenCV %s has no precise upscale: its keypoints are shifted by +0.25 px\n", CV_VERSION);
#endif
  std::vector<cv::KeyPoint> kp;
  cv::Mat desc;
  auto t0 = std::chrono::steady_clock::now();
  sift->detectAndCompute(gray, cv::noArray(), kp, desc);
  printf("OpenCV: %.2f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  std::vector<Feature> out(kp.size());
  for (size_t i=0;i<kp.size();i++) {
    Feature &f = out[i];
    f.x = kp[i].pt.x;
    f.y = kp[i].pt.y;
    f.sigma = 0.5f*kp[i].size;
    f.angle = kp[i].angle;
    int o = kp[i].octave & 255;
    f.octave = (o>=128 ? o - 256 : o);
    Normalize(f.desc, desc.ptr<float>((int)i));
  }
  return out;
}

#ifdef HAVE_VLFEAT
// VLFeat's SIFT with its defaults or, with colmap, with COLMAP's settings and post-processing
// (sift.cc, VLFeat path): whole DoG levels kept from the coarsest until they have more than
// 8192 detections, L1-root descriptors in OpenCV's bin order and rounded to 8 bits; without
// COLMAP's +0.5 shift. The settings are written here independently of
// SiftCompatParams::COLMAP, as a reference.
static std::vector<Feature> VLFeatFeatures(const cv::Mat &gray, bool colmap)
{
  // COLMAP's conversion (x/255.0f; convertTo multiplies by the scale, which rounds differently)
  cv::Mat img(gray.size(), CV_32F);
  for (int y=0;y<gray.rows;y++)
    for (int x=0;x<gray.cols;x++)
      img.at<float>(y, x) = gray.at<uchar>(y, x)/255.0f;
  auto t0 = std::chrono::steady_clock::now();
  VlSiftFilt *f = vl_sift_new(img.cols, img.rows, colmap ? 4 : -1, 3, colmap ? -1 : 0);
  if (colmap) {
    vl_sift_set_peak_thresh(f, 0.02/3);
    vl_sift_set_edge_thresh(f, 10.0);
  }
  std::vector<std::vector<Feature>> levels;
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
        float d[128] = {0};
        vl_sift_calc_keypoint_descriptor(f, d, &keys[i], angles[j]);
        ft.x = keys[i].x;
        ft.y = keys[i].y;
        ft.sigma = keys[i].sigma;
        ft.angle = (float)(angles[j]*180.0/CV_PI);
        ft.octave = keys[i].o;
        if (colmap) {
          double l1 = 0.0;
          for (int k=0;k<128;k++)
            l1 += std::fabs(d[k]);
          for (int k=0;k<128;k++) {
            float v = (float)std::sqrt(d[k]*(l1>0 ? 1.0f/(float)l1 : 0.0f));   // no descriptor: zero, as ExtractSiftCompat
            ft.desc[k - k%8 + (8 - k%8)%8] = std::min(255.0f, std::round(512.0f*v));
          }
        } else
          memcpy(ft.desc, d, sizeof(d));
        levels.back().push_back(ft);
      }
    }
    err = vl_sift_process_next_octave(f);
  }
  vl_sift_delete(f);
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
  std::vector<Feature> out;
  for (size_t l=first;l<levels.size();l++)
    out.insert(out.end(), levels[l].begin(), levels[l].end());
  printf("VLFeat: %.2f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  for (Feature &ft : out)
    Normalize(ft.desc, ft.desc);
  return out;
}
#endif

static std::vector<Feature> CudaSiftFeatures(const cv::Mat &gray, Reference ref, bool doublePrecision, bool original,
                                             float thresh, float initBlur, float angleOffset)
{
  cv::Mat fimg;
  gray.convertTo(fimg, CV_32FC1);
  CudaImage img;
  img.Allocate(fimg.cols, fimg.rows, iAlignUp(fimg.cols, 128), false, NULL, (float *)fimg.data);
  img.Download();
  SiftData data;
  InitSiftData(data, 65536, true, true);
  for (int run=0;run<2;run++) {   // the second run is timed
    auto t0 = std::chrono::steady_clock::now();
    if (original)
      ExtractSift(data, img, 5, initBlur, thresh, 0.0f, false);
    else
    {
      SiftCompatParams p = (ref==REF_VLFEAT ? SiftCompatParams::VLFeat() :
                            ref==REF_COLMAP ? SiftCompatParams::COLMAP() : SiftCompatParams::OpenCV());
      p.doublePrecision = doublePrecision;
      ExtractSiftCompat(data, img, p);
    }
    cudaDeviceSynchronize();
    if (run==1)
      printf("CudaSift: %.2f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
#ifdef MANAGEDMEM
  const SiftPoint *pts = data.m_data;
#else
  const SiftPoint *pts = data.h_data;
#endif
  std::vector<Feature> out(data.numPts);
  for (int i=0;i<data.numPts;i++) {
    const SiftPoint &p = pts[i];
    Feature &f = out[i];
    f.x = p.xpos;
    f.y = p.ypos;
    f.sigma = p.scale;
    f.angle = (float)std::fmod(p.orientation + angleOffset + 720.0, 360.0);
    f.octave = (int)std::lrint(std::log2(p.subsampling));
    Normalize(f.desc, p.data);
  }
  FreeSiftData(data);
  return out;
}

struct Tolerance {
  float px, scale, deg;   // position (px), scale (relative), angle (degrees)
};
#define NUM_TOLERANCES 3
#define MAX_OCTAVES 16   // octaves -1 to 14
static const Tolerance tolerances[NUM_TOLERANCES] = {{0.1f, 0.01f, 1.0f}, {0.5f, 0.05f, 5.0f}, {1.0f, 0.10f, 10.0f}};

struct Totals {
  long nRef = 0, nOurs = 0;
  long recalled[NUM_TOLERANCES] = {0}, precise[NUM_TOLERANCES] = {0};
  std::vector<float> dpos, dscale, dangle, ddesc;   // pairs at the loosest tolerance
  long octRef[MAX_OCTAVES] = {0}, octRecalled[MAX_OCTAVES] = {0};     // per octave + 1 (-1 is the first), loosest tolerance
  long sameDesc = 0;                                // pairs with identical (quantized) descriptors
  void Add(const Totals &T)
  {
    nRef += T.nRef;
    nOurs += T.nOurs;
    for (int k=0;k<NUM_TOLERANCES;k++) {
      recalled[k] += T.recalled[k];
      precise[k] += T.precise[k];
    }
    dpos.insert(dpos.end(), T.dpos.begin(), T.dpos.end());
    dscale.insert(dscale.end(), T.dscale.begin(), T.dscale.end());
    dangle.insert(dangle.end(), T.dangle.begin(), T.dangle.end());
    ddesc.insert(ddesc.end(), T.ddesc.begin(), T.ddesc.end());
    sameDesc += T.sameDesc;
    for (int o=0;o<MAX_OCTAVES;o++) {
      octRef[o] += T.octRef[o];
      octRecalled[o] += T.octRecalled[o];
    }
  }
};

// Index of the closest feature of b within tolerance t of a, or -1 (a grid would be faster;
// fine for a few thousand points per image)
static int Closest(const Feature &a, const std::vector<Feature> &b, const Tolerance &t)
{
  int best = -1;
  double bestD = 1e9;
  for (size_t j=0;j<b.size();j++) {
    double d = std::hypot(a.x - b[j].x, a.y - b[j].y);
    if (d>t.px || std::fabs(b[j].sigma/a.sigma - 1.0)>t.scale || AngleDiff(a.angle, b[j].angle)>t.deg)
      continue;
    if (d<bestD) {
      bestD = d;
      best = (int)j;
    }
  }
  return best;
}

static void Compare(const std::vector<Feature> &ref, const std::vector<Feature> &ours, Totals &T)
{
  T.nRef += ref.size();
  T.nOurs += ours.size();
  for (int k=0;k<NUM_TOLERANCES;k++) {
    for (const Feature &a : ref)
      T.recalled[k] += (Closest(a, ours, tolerances[k])>=0);
    for (const Feature &b : ours)
      T.precise[k] += (Closest(b, ref, tolerances[k])>=0);
  }
  for (const Feature &a : ref) {
    int j = Closest(a, ours, tolerances[2]);
    int o = std::min(std::max(a.octave + 1, 0), 15);
    T.octRef[o]++;
    if (j<0)
      continue;
    T.octRecalled[o]++;
    const Feature &b = ours[j];
    T.dpos.push_back((float)std::hypot(a.x - b.x, a.y - b.y));
    T.dscale.push_back((float)std::fabs(b.sigma/a.sigma - 1.0));
    T.dangle.push_back((float)AngleDiff(a.angle, b.angle));
    double d2 = 0.0;
    for (int k=0;k<128;k++)
      d2 += (a.desc[k] - b.desc[k])*(a.desc[k] - b.desc[k]);
    T.ddesc.push_back((float)std::sqrt(d2));
    T.sameDesc += (d2<1e-10);
  }
}

static float Median(std::vector<float> v)
{
  if (v.empty())
    return 0.0f;
  std::sort(v.begin(), v.end());
  return v[v.size()/2];
}

static void Report(const char *name, const Totals &T)
{
  printf("%s: reference %ld keypoints, CudaSift %ld\n", name, T.nRef, T.nOurs);
  for (int k=0;k<NUM_TOLERANCES;k++)
    printf("  within %.1f px, %.0f%% scale, %.0f deg: recall %.5f (%ld missing), precision %.5f (%ld extra)\n", tolerances[k].px, 100*tolerances[k].scale, tolerances[k].deg,
           T.nRef ? (double)T.recalled[k]/T.nRef : 0.0, T.nRef - T.recalled[k], T.nOurs ? (double)T.precise[k]/T.nOurs : 0.0, T.nOurs - T.precise[k]);
  printf("  pairs (loosest tolerance): median position error %.2g px, scale error %.2g, angle error %.2g deg, descriptor L2 %.2g, identical descriptors %.3f\n",
         Median(T.dpos), Median(T.dscale), Median(T.dangle), Median(T.ddesc), T.dpos.empty() ? 0.0 : (double)T.sameDesc/T.dpos.size());
  printf("  recall per reference octave:");
  for (int o=0;o<MAX_OCTAVES;o++)
    if (T.octRef[o])
      printf(" %d: %.3f (%ld)", o - 1, (double)T.octRecalled[o]/T.octRef[o], T.octRef[o]);
  printf("\n");
}

int main(int argc, char **argv)
{
  float thresh = 3.0f, initBlur = 1.0f, angleOffset = 0.0f;
  bool original = false, doublePrecision = false;
  Reference ref = REF_OPENCV;
  int device = 0;
  std::vector<std::string> files;
  for (int i=1;i<argc;i++) {
    if (!strcmp(argv[i], "--original"))
      original = true;
    else if (!strcmp(argv[i], "--vlfeat") || !strcmp(argv[i], "--colmap")) {
#ifdef HAVE_VLFEAT
      ref = (argv[i][2]=='v' ? REF_VLFEAT : REF_COLMAP);
#else
      printf("%s needs VLFeat: build with -DVLFEAT_DIR=<prefix>\n", argv[i]);
      return 1;
#endif
    }
    else if (!strcmp(argv[i], "--double"))
      doublePrecision = true;
    else if (!strcmp(argv[i], "--thresh") && i+1<argc)
      thresh = (float)atof(argv[++i]);
    else if (!strcmp(argv[i], "--init-blur") && i+1<argc)
      initBlur = (float)atof(argv[++i]);
    else if (!strcmp(argv[i], "--angle-offset") && i+1<argc)
      angleOffset = (float)atof(argv[++i]);
    else if (argv[i][0]>='0' && argv[i][0]<='9' && !strchr(argv[i], '.'))
      device = atoi(argv[i]);
    else
      files.push_back(argv[i]);
  }
  if (files.empty()) {
    printf("Usage: %s [--vlfeat | --colmap] [--double] [--original [--thresh t] [--init-blur sigma] [--angle-offset deg]] image... [device]\n", argv[0]);
    return 1;
  }
  InitCuda(device);
  Totals all;
  for (const std::string &file : files) {
    cv::Mat gray = cv::imread(file, cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
      printf("Cannot read %s\n", file.c_str());
      continue;
    }
    Totals T;
    std::vector<Feature> refFeatures;
#ifdef HAVE_VLFEAT
    if (ref!=REF_OPENCV)
      refFeatures = VLFeatFeatures(gray, ref==REF_COLMAP);
    else
#endif
      refFeatures = OpenCVFeatures(gray);
    Compare(refFeatures, CudaSiftFeatures(gray, ref, doublePrecision, original, thresh, initBlur, angleOffset), T);
    Report(file.c_str(), T);
    all.Add(T);
  }
  if (files.size()>1)
    Report("all", all);
  FreeSiftCompatMemory();
  return 0;
}
