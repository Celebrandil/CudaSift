//********************************************************//
// OpenCV- and VLFeat-compatible SIFT extraction          //
//********************************************************//

// The host code of ExtractSiftCompat that doesn't depend on the backend, included by
// cudaSiftCompat.cu (CUDA) and metal/cudaSiftCompatMetal.mm (Metal): the validation of the
// parameters, the pyramids of OpenCV's and VLFeat's SIFT and the selection of the keypoints.
// The pyramid builders are templated on the backend, which places the levels (pointers with
// CUDA, offsets in a buffer with Metal) and launches the input, blur and downsampling kernels,
// and on its pyramid and parameter types. The includer defines COMPAT_MAX_OCTAVES,
// COMPAT_MAX_LAYERS, COMPAT_MAX_LEVELS, COMPAT_MAX_RADIUS and SIFT_INIT_SIGMA (a float).

#ifndef CUDASIFTCOMPATHOST_H
#define CUDASIFTCOMPATHOST_H

#if !defined(COMPAT_MAX_OCTAVES) || !defined(COMPAT_MAX_LAYERS) || !defined(COMPAT_MAX_LEVELS) || !defined(COMPAT_MAX_RADIUS) || !defined(SIFT_INIT_SIGMA)
#error "define the constants of ExtractSiftCompat before including cudaSiftCompatHost.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>
#include "cudaImage.h"
#include "cudaSiftCompat.h"

// Stops the program with a message if the parameters are invalid
inline void ValidateCompatParams(const SiftCompatParams &prm)
{
  if (prm.nOctaveLayers<1 || prm.nOctaveLayers>COMPAT_MAX_LAYERS) {
    fprintf(stderr, "ExtractSiftCompat: nOctaveLayers must be in [1, %d]\n", COMPAT_MAX_LAYERS);
    exit(1);
  }
  if (prm.firstOctave!=-1 && prm.firstOctave!=0) {
    fprintf(stderr, "ExtractSiftCompat: firstOctave must be -1 or 0\n");
    exit(1);
  }
  if (prm.extendFirstOctave && prm.firstOctave!=0) {
    fprintf(stderr, "ExtractSiftCompat: extendFirstOctave needs firstOctave 0\n");
    exit(1);
  }
  if (prm.numOctaves==0 || prm.numOctaves<-1) {
    fprintf(stderr, "ExtractSiftCompat: numOctaves must be -1 (the default) or at least 1\n");
    exit(1);
  }
}

// At most COMPAT_MAX_OCTAVES octaves (for images of more than 2^16 pixels across), with a message
inline int CapOctaves(int n)
{
  if (n>COMPAT_MAX_OCTAVES)
    fprintf(stderr, "ExtractSiftCompat: %d octaves, only the first %d computed\n", n, COMPAT_MAX_OCTAVES);
  return std::min(n, COMPAT_MAX_OCTAVES);
}

// The levels of the extension of the first octave (extendFirstOctave), 0 without it
inline int ExtensionLevels(const SiftCompatParams &prm)
{
  if (!prm.extendFirstOctave)
    return 0;
  return (prm.extendFirstOctaveScales<0 ? prm.nOctaveLayers : std::min(prm.extendFirstOctaveScales, prm.nOctaveLayers));
}

// getGaussianKernel(ksize, sigma, CV_32F), computed in double as OpenCV does; returns the radius
inline int GaussianKernelOpenCV(double sigma, float *kernel)
{
  int n = (int)std::lrint(sigma*4*2 + 1) | 1;
  int r = (n - 1)/2;
  if (r>COMPAT_MAX_RADIUS) {
    fprintf(stderr, "ExtractSiftCompat: Gaussian kernel radius %d > %d\n", r, COMPAT_MAX_RADIUS);
    exit(1);
  }
  double scale2X = -0.125/(sigma*sigma);
  std::vector<double> values(r);
  double sum = 0.0;
  for (int i=0,x=1-n;i<r;i++,x+=2) {
    values[i] = std::exp((double)(x*x)*scale2X);
    sum += values[i];
  }
  sum = sum*2 + 1.0;
  double mul1 = 1.0/sum;
  for (int i=0;i<r;i++)
    kernel[i] = kernel[n - 1 - i] = (float)(values[i]*mul1);
  kernel[r] = (float)mul1;
  return r;
}

// The kernel as _vl_sift_smooth computes it, in float; returns the radius
inline int GaussianKernelVLFeat(double sigma, float *kernel)
{
  int W = std::max((int)std::ceil(4.0*sigma), 1);
  if (W>COMPAT_MAX_RADIUS) {
    fprintf(stderr, "ExtractSiftCompat: Gaussian kernel radius %d > %d\n", W, COMPAT_MAX_RADIUS);
    exit(1);
  }
  float acc = 0;
  for (int j=0;j<2*W+1;j++) {
    float d = ((float)(j - W))/((float)sigma);
    kernel[j] = (float)std::exp(-0.5*(d*d));
    acc += kernel[j];
  }
  for (int j=0;j<2*W+1;j++)
    kernel[j] /= acc;
  return W;
}

// A threshold in the backend's type: as is in double, or the largest float at or below it, so
// that a float v is above it exactly when v is above the threshold in double
inline void ThresholdBelow(double v, double &out) { out = v; }
inline void ThresholdBelow(double v, float &out)
{
  out = (float)v;
  if ((double)out>v)
    out = std::nextafter(out, -1.0f);
}

// The pyramid of OpenCV's SIFT (createInitialImage, buildGaussianPyramid), and the parameters of
// findScaleSpaceExtrema. The backend B places the levels after two temporary images
// (Allocate, which returns the first; the levels are then B::At(P, o, i)) and runs the kernels
// (Input, Blur, Downsample).
template <class B, class Pyramid, class Params>
void BuildPyramidOpenCV(B &backend, Pyramid &P, Params &dp, CudaImage &img, const SiftCompatParams &prm)
{
  typedef typename B::Level Level;
  const int L = prm.nOctaveLayers;
  const int nLev = L + 3;
  const int firstOctave = prm.firstOctave;
  const bool upscale = (firstOctave<0);
  const int E = ExtensionLevels(prm);
  int bw = (upscale ? 2*img.width : img.width);
  int bh = (upscale ? 2*img.height : img.height);
  int nOct = (int)std::lrint(std::log((double)std::min(bw, bh))/std::log(2.) - 2) - firstOctave;
  nOct = std::max(1, CapOctaves(nOct));
  P.nOct = nOct;
  P.nLev = nLev;
  for (int o=0,w=bw,h=bh;o<nOct;o++,w/=2,h/=2) {
    P.w[o] = w;
    P.h[o] = h;
    P.p[o] = iAlignUp(w, 32);
    P.pix[o] = (float)(1 << o);
    P.lev[o] = (o==0 ? E : 0);
  }
  // The input image in float (upsampled or not) and a temporary image, then the levels
  const size_t baseSize = (size_t)P.p[0]*bh;
  const Level base = backend.Allocate(P, 2*baseSize), tmp = base + baseSize;
  // OpenCV passes sigma as a float to createInitialImage and to the extrema, as a double to
  // buildGaussianPyramid (and the thresholds as floats)
  const float sigmaF = (float)prm.sigma;
  float kernel[2*COMPAT_MAX_RADIUS + 1];
  auto smooth = [&](Level dst, Level src, int o, double sigma) {
    int r = GaussianKernelOpenCV(sigma, kernel);
    backend.Blur(SIFT_COMPAT_OPENCV, dst, src, tmp, P.w[o], P.h[o], P.p[o], kernel, r);
  };
  backend.Input(SIFT_COMPAT_OPENCV, base, bw, bh, P.p[0], img, upscale);
  float sig_diff;
  if (upscale)
    sig_diff = sqrtf(std::max(sigmaF*sigmaF - SIFT_INIT_SIGMA*SIFT_INIT_SIGMA*4, 0.01f));
  else
    sig_diff = sqrtf(std::max(sigmaF*sigmaF - SIFT_INIT_SIGMA*SIFT_INIT_SIGMA, 0.01f));
  smooth(B::At(P, 0, E), base, 0, sig_diff);
  std::vector<double> sig(nLev);
  sig[0] = prm.sigma;
  double k = std::pow(2., 1./L);
  for (int i=1;i<nLev;i++) {
    double sig_prev = std::pow(k, (double)(i-1))*prm.sigma;
    double sig_total = sig_prev*k;
    sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
  }
  // The extension: the E levels below the first one, at sigma * k^(i - E), each blurred directly
  // from the input (blur SIFT_INIT_SIGMA); the levels from sigma up are OpenCV's
  for (int i=0;i<E;i++) {
    float t = sigmaF*(float)std::pow(k, (double)(i - E));
    smooth(B::At(P, 0, i), base, 0, sqrtf(std::max(t*t - SIFT_INIT_SIGMA*SIFT_INIT_SIGMA, 0.01f)));
  }
  for (int o=0;o<nOct;o++) {
    auto g = [&](int i) { return B::At(P, o, P.lev[o] + i); };   // OpenCV's levels
    if (o>0)
      backend.Downsample(g(0), P.w[o], P.h[o], P.p[o], B::At(P, o-1, P.lev[o-1] + L), P.p[o-1]);
    for (int i=1;i<nLev;i++)
      smooth(g(i), g(i-1), o, sig[i]);
  }
  dp.preThreshold = (float)(int)std::floor(0.5*(float)prm.contrastThreshold/L*255);
  dp.contrastThreshold = (float)prm.contrastThreshold;
  dp.edgeThreshold = prm.edgeThreshold;
  dp.sigma = sigmaF;
  dp.S = L;
}

// The pyramid of VLFeat's SIFT (vl_sift_process_first_octave, vl_sift_process_next_octave), and
// the parameters of vl_sift_detect (the backend as for BuildPyramidOpenCV)
template <class B, class Pyramid, class Params>
void BuildPyramidVLFeat(B &backend, Pyramid &P, Params &dp, CudaImage &img, const SiftCompatParams &prm)
{
  typedef typename B::Level Level;
  const int S = prm.nOctaveLayers, s_min = -1, s_max = S + 1;
  const int nLev = s_max - s_min + 1;
  const int o_min = prm.firstOctave;
  const int E = ExtensionLevels(prm);
  int O = prm.numOctaves;
  if (O<0)
    O = std::max((int)std::floor(std::log2((double)std::min(img.width, img.height))) - o_min - 3, 1);
  O = CapOctaves(O);
  P.nLev = nLev;
  P.nOct = 0;
  for (int o=0;o<O;o++) {
    int oc = o_min + o;
    int w = (oc<0 ? img.width << -oc : img.width >> oc);
    int h = (oc<0 ? img.height << -oc : img.height >> oc);
    if (w<1 || h<1)
      break;
    P.w[o] = w;
    P.h[o] = h;
    P.p[o] = iAlignUp(w, 32);
    P.pix[o] = (oc<0 ? 1.0f/(1 << -oc) : (float)(1 << oc));
    P.lev[o] = (o==0 ? E : 0);
    P.nOct++;
  }
  O = P.nOct;
  // The unblurred base of the octave and a temporary image, then the levels
  const size_t baseSize = (size_t)P.p[0]*P.h[0];
  const Level base = backend.Allocate(P, 2*baseSize), tmp = base + baseSize;
  const double sigmak = std::pow(2.0, 1.0/S);
  const double sigma0 = 1.6*sigmak;
  const double sigman = 0.5;
  const double dsigma0 = sigma0*std::sqrt(1.0 - 1.0/(sigmak*sigmak));
  float kernel[2*COMPAT_MAX_RADIUS + 1];
  for (int o=0;o<O;o++) {
    const int w = P.w[o], h = P.h[o], p = P.p[o];
    auto smooth = [&](Level dst, Level src, double sigma) {
      int r = GaussianKernelVLFeat(sigma, kernel);
      backend.Blur(SIFT_COMPAT_VLFEAT, dst, src, tmp, w, h, p, kernel, r);
    };
    auto g = [&](int i) { return B::At(P, o, P.lev[o] + i); };   // VLFeat's levels: level s is g(s - s_min)
    // The blur of the octave's base, from (sb) and to (sa): VLFeat blurs it in place, here it is
    // built in `base` and blurred into g(0), so that the blur doesn't run in place
    double sa, sb;
    if (o==0) {
      sa = sigma0*std::pow(sigmak, s_min);
      sb = sigman*std::pow(2.0, -o_min);
    } else {
      // The blur VLFeat assumes after downsampling, computed in float as VLFeat does
      const int s_best = std::min(s_min + S, s_max);
      sa = sigma0*powf((float)sigmak, (float)s_min);
      sb = sigma0*powf((float)sigmak, (float)(s_best - S));
    }
    const Level src = (sa>sb ? base : g(0));   // sa > sb in the first octave, which the extension uses
    if (o==0) {
      backend.Input(SIFT_COMPAT_VLFEAT, src, w, h, p, img, o_min<0);
      // The extension: the E levels below s_min, each blurred directly from the input
      for (int i=0;i<P.lev[0];i++) {
        const double si = sigma0*std::pow(sigmak, s_min - P.lev[0] + i);
        smooth(B::At(P, 0, i), src, std::sqrt(si*si - sigman*sigman));
      }
    } else {
      // The level VLFeat downsamples, s_min + S (at most s_max)
      const int s_best = std::min(s_min + S, s_max);
      backend.Downsample(src, w, h, p, B::At(P, o-1, P.lev[o-1] + s_best - s_min), P.p[o-1]);
    }
    if (sa>sb)
      smooth(g(0), src, std::sqrt(sa*sa - sb*sb));
    for (int s=s_min+1;s<=s_max;s++)
      smooth(g(s - s_min), g(s - s_min - 1), dsigma0*std::pow(sigmak, s));
  }
  ThresholdBelow(prm.peakThreshold, dp.peakThreshold);
  dp.peakPrefilter = (float)(0.8*prm.peakThreshold);   // the smallest float >= 0.8*peakThreshold
  if ((double)dp.peakPrefilter<0.8*prm.peakThreshold)
    dp.peakPrefilter = std::nextafter(dp.peakPrefilter, 1.0f);
  dp.edgeThreshold = prm.edgeThreshold;
  dp.sigma0 = sigma0;
  dp.S = S;
  dp.s_min = s_min;
  dp.s_max = s_max;
}

// Keeps the k keypoints of strongest response, in their order, if there are more: with allTies,
// also all those with the k-th response (OpenCV's retainBest), else only the first of them
template <typename T>
void KeepStrongest(std::vector<T> &kpts, int k, float (*response)(const T &), bool allTies)
{
  if (k<=0) {
    kpts.clear();
    return;
  }
  if (kpts.size()<=(size_t)k)
    return;
  std::vector<float> resp(kpts.size());
  for (size_t j=0;j<kpts.size();j++)
    resp[j] = response(kpts[j]);
  std::nth_element(resp.begin(), resp.begin() + k - 1, resp.end(), std::greater<float>());
  const float thr = resp[k - 1];
  int ties = k;   // how many of those at the threshold fit, after those above it
  for (const T &p : kpts)
    ties -= (response(p)>thr);
  size_t m = 0;
  for (size_t j=0;j<kpts.size();j++) {
    float r = response(kpts[j]);
    if (r>thr || (r==thr && (allTies || ties-->0)))
      kpts[m++] = kpts[j];
  }
  kpts.resize(m);
}

// The keypoints that fit in SiftData::maxPts, with a message if some don't
template <typename T>
void KeepMaxPts(std::vector<T> &kpts, int maxPts, float (*response)(const T &))
{
  if (kpts.size()>(size_t)std::max(maxPts, 0))
    fprintf(stderr, "ExtractSiftCompat: %d keypoints, only the %d strongest kept (SiftData.maxPts)\n", (int)kpts.size(), maxPts);
  KeepStrongest(kpts, maxPts, response, false);
}

// OpenCV's keypoints kept: KeyPoint12_LessThan's order, duplicates removed and the nfeatures
// strongest (removeDuplicatedSorted, retainBest), then those that fit in maxPts
template <typename K>
void SelectKeypointsOpenCV(std::vector<K> &kpts, const SiftCompatParams &prm, int maxPts)
{
  std::sort(kpts.begin(), kpts.end(), [](const K &a, const K &b) {
    if (a.x!=b.x)
      return a.x<b.x;
    if (a.y!=b.y)
      return a.y<b.y;
    if (a.size!=b.size)
      return a.size>b.size;
    if (a.angle!=b.angle)
      return a.angle<b.angle;
    if (a.response!=b.response)
      return a.response>b.response;
    return a.packedOctave>b.packedOctave;
  });
  size_t m = 0;
  for (size_t j=0;j<kpts.size();j++)
    if (m==0 || kpts[m-1].x!=kpts[j].x || kpts[m-1].y!=kpts[j].y || kpts[m-1].size!=kpts[j].size || kpts[m-1].angle!=kpts[j].angle)
      kpts[m++] = kpts[j];
  kpts.resize(m);
  float (*response)(const K &) = [](const K &k) { return k.response; };
  if (prm.nfeatures>0)
    KeepStrongest<K>(kpts, prm.nfeatures, response, true);
  KeepMaxPts<K>(kpts, maxPts, response);
}

// VLFeat's keypoints kept: VLFeat's order (octave, level, then the position where the extremum
// was detected), COLMAP's selection of whole levels from the coarsest (maxFeatures, counting the
// numCands candidates cands), then those that fit in maxPts. level(k): the index of the
// keypoint's level among those of all the octaves, from the finest.
template <typename K, typename F>
void SelectKeypointsVLFeat(std::vector<K> &kpts, const SiftCompatParams &prm, F level, const K *cands, unsigned int numCands, int maxPts)
{
  std::sort(kpts.begin(), kpts.end(), [](const K &a, const K &b) {
    if (a.o!=b.o)
      return a.o<b.o;
    if (a.is!=b.is)
      return a.is<b.is;
    if (a.raster!=b.raster)
      return a.raster<b.raster;
    return a.ori<b.ori;
  });
  if (prm.maxFeatures>0 && numCands>0) {
    std::vector<int> levelCount(COMPAT_MAX_OCTAVES*COMPAT_MAX_LEVELS, 0);
    for (unsigned int j=0;j<numCands;j++)
      levelCount[level(cands[j])]++;
    int num = 0, firstLevel = 0;
    for (int l=(int)levelCount.size()-1;l>=0;l--) {
      num += levelCount[l];
      if (num>prm.maxFeatures) {
        firstLevel = l;
        break;
      }
    }
    size_t m = 0;
    for (size_t j=0;j<kpts.size();j++)
      if (level(kpts[j])>=firstLevel)
        kpts[m++] = kpts[j];
    kpts.resize(m);
  }
  KeepMaxPts<K>(kpts, maxPts, [](const K &k) { return fabsf(k.val); });
}

#endif
