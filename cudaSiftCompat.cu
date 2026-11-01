//********************************************************//
// OpenCV- and VLFeat-compatible SIFT extraction          //
//********************************************************//

// GPU versions of OpenCV's SIFT (modules/features2d/src/sift.dispatch.cpp and sift.simd.hpp,
// OpenCV 4.8 to 4.13) and of VLFeat's (vl/sift.c, VLFeat 0.9.21), step by step and with the
// same arithmetic where it matters: the same scale space (upsampling, Gaussian kernels and
// borders), extrema, interpolation, orientation histograms and descriptors. Results differ
// from theirs only by floating-point rounding (summation orders, fused multiply-adds), and
// don't vary between runs. Shares nothing with ExtractSift but the data structures. Both modes
// (COLMAP's settings are VLFeat's mode with other parameters) run the same kernels, templated on
// the mode and on the precision of the refinement, orientations and descriptors (float, or
// double as VLFeat).

#include <cstdio>
#include <cmath>
#include <cfloat>
#include <climits>
#include <vector>
#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include "cudautils.h"
#include "cudaImage.h"
#include "cudaSift.h"
#include "cudaSiftCompat.h"

#define COMPAT_MAX_OCTAVES 14
#define COMPAT_MAX_LAYERS  6     // nOctaveLayers
#define COMPAT_MAX_LEVELS  (2*COMPAT_MAX_LAYERS + 3)   // with extendFirstOctave
#define COMPAT_MAX_RADIUS  64    // of the Gaussian kernels

// OpenCV's constants (sift.simd.hpp)
#define SIFT_DESCR_WIDTH      4
#define SIFT_DESCR_HIST_BINS  8
#define SIFT_INIT_SIGMA       0.5f
#define SIFT_IMG_BORDER       5
#define SIFT_MAX_INTERP_STEPS 5
#define SIFT_ORI_HIST_BINS    36
#define SIFT_ORI_SIG_FCTR     1.5f
#define SIFT_ORI_RADIUS       4.5f
#define SIFT_ORI_PEAK_RATIO   0.8f
#define SIFT_DESCR_SCL_FCTR   3.f
#define SIFT_DESCR_MAG_THR    0.2f
#define SIFT_INT_DESCR_FCTR   512.f

// VLFeat's constants (vl/sift.c)
#define VL_EXPN_SZ  256
#define VL_EXPN_MAX 25.0
#define VL_PI 3.141592653589793
#define VL_EPSILON_F 1.19209290E-07F
#define VL_EPSILON_D 2.220446049250313e-16
#define VL_MAX_W 128            // largest window radius of the orientations and descriptors
#define VL_MAX_ORIENTATION_PEAKS 4   // peaks scanned per orientation histogram
#define VL_MAX_REFINE_STEPS 5        // of the refinement of the extrema

// The scale space: nLev = S + 3 Gaussian levels per octave (S = nOctaveLayers), plus lev[o]
// levels at the bottom of the first octave with extendFirstOctave. The DoG levels are computed
// where they are read (DogView). The levels of an octave are indexed from its lowest one:
//   gauss[o][i]: OpenCV's level i - lev[o] (sigma * 2^((i - lev[o])/S) in octave pixels),
//                VLFeat's level s = i + s_min - lev[o]
// so that the extrema are searched in the DoG levels 1..S + lev[o] in both modes.
struct CompatPyramid {
  int nOct, nLev;
  int w[COMPAT_MAX_OCTAVES], h[COMPAT_MAX_OCTAVES], p[COMPAT_MAX_OCTAVES];
  float pix[COMPAT_MAX_OCTAVES];   // pixel size of the octave in keypoint coordinates
                                   // (OpenCV: those of the first octave; VLFeat: of the input)
  int lev[COMPAT_MAX_OCTAVES];     // levels below the mode's first one (the extension)
  float *gauss[COMPAT_MAX_OCTAVES][COMPAT_MAX_LEVELS];
};

// An OpenCV keypoint: the fields of cv::KeyPoint in the coordinates of the first octave,
// before OpenCV rescales them to the input image
struct OpenCVKeypoint {
  float x, y, size, angle, response;
  int packedOctave;      // as KeyPoint::octave: octave + (layer << 8) + (round((xi + 0.5)*255) << 16),
                         // with the level index of CompatPyramid (used to sort only)
  float contr, edge;     // signed interpolated DoG value, trace^2/det of the Hessian
  int o, layer, r, c;    // octave (0 = first), level and integer position in it
};

// A VLFeat keypoint (VlSiftKeypoint) with its orientation and detection values
struct VLFeatKeypoint {
  float x, y, sigma;       // position and sigma in input-image pixels, in float as VLFeat's
                           // VlSiftKeypoint (the orientations and descriptors start from them)
  double angle;            // radians
  float val, score;        // interpolated DoG value, trace^2/det of the Hessian
  int o, is;               // octave and level of the detection
  int ori;                 // index of the orientation among the keypoint's peaks
  int raster;              // y*width + x where the extremum was detected (VLFeat's order)
};

// What differs between the modes besides the code under `if constexpr (M==...)`: the keypoint
// type, the border without extrema, the warps (keypoints) per block of OrientationsCompat and
// the threads per block (keypoint) of DescriptorsCompat
template<SiftCompatMode M> struct CompatModeTraits;
template<> struct CompatModeTraits<SIFT_COMPAT_OPENCV> {
  typedef OpenCVKeypoint Keypoint;
  static constexpr int border = SIFT_IMG_BORDER, oriWarps = 4, descThreads = 256;
};
template<> struct CompatModeTraits<SIFT_COMPAT_VLFEAT> {
  typedef VLFeatKeypoint Keypoint;
  static constexpr int border = 1, oriWarps = 2, descThreads = 128;
};

static_assert(CompatModeTraits<SIFT_COMPAT_OPENCV>::descThreads>=128 && CompatModeTraits<SIFT_COMPAT_VLFEAT>::descThreads>=128,
              "DescriptorsCompat uses a thread per value of the descriptor");

// Parameters of the extrema detection and refinement
struct CompatDetectParams {
  int S, s_min, s_max;     // levels per octave; VLFeat's first and last levels (-1, S + 1)
  double edgeThreshold;   // as VLFeat keeps it (OpenCV converts it to float)
  // OpenCV
  float preThreshold;      // OpenCV's integer threshold on the DoG value before refinement
  float contrastThreshold, sigma;
  // VLFeat
  double peakThreshold, sigma0;
  float peakPrefilter;     // smallest float >= 0.8*peakThreshold: v >= peakPrefilter exactly
                           // when v >= 0.8*peakThreshold (VLFeat's test in double)
};

// The level index in CompatPyramid of a VLFeat keypoint (whose VLFeat level is is)
__host__ __device__ __forceinline__ int LevelIndexVLFeat(const CompatPyramid &P, const CompatDetectParams &dp, const VLFeatKeypoint &kp)
{
  return kp.is - dp.s_min + P.lev[kp.o];
}

__constant__ float d_CompatKernel[2*COMPAT_MAX_RADIUS + 1];

// A DoG level, computed where it is read from the two Gaussian levels instead of stored:
// dog[i] = upper[i] - lower[i], the same subtraction as buildDoGPyramid's and vl_sift_detect's
struct DogView {
  const float *lower, *upper;
  __device__ __forceinline__ float operator[](int i) const { return upper[i] - lower[i]; }
  __device__ __forceinline__ DogView operator+(int i) const { return {lower + i, upper + i}; }
};

__device__ __forceinline__ DogView Dog(const CompatPyramid &P, int o, int l)
{
  return {P.gauss[o][l], P.gauss[o][l+1]};
}

// OpenCV's borderInterpolate for BORDER_REFLECT_101
__device__ __forceinline__ int Reflect101(int i, int n)
{
  if (n==1)
    return 0;
  do {
    if (i<0)
      i = -i;
    else
      i = n - 1 - (i - n) - 1;
  } while ((unsigned)i>=(unsigned)n);
  return i;
}

// The index of a sample of the borders: OpenCV's BORDER_REFLECT_101, VLFeat's continuity
template<SiftCompatMode M>
__device__ __forceinline__ int BorderIndex(int i, int n)
{
  if (i>=0 && i<n)
    return i;
  return (M==SIFT_COMPAT_VLFEAT ? min(max(i, 0), n - 1) : Reflect101(i, n));
}

// Adds v to hist[bin] for every lane of the warp with bin >= 0, in an order that doesn't vary
// between runs (unlike atomicAdd): the lanes with the same bin are summed in lane order by
// the first of them. All lanes must call it; buf holds 32 floats per warp.
// The lanes of the warp whose v is the lane's: __match_any_sync (compute capability 7.0 and
// later), or the same mask from 32 shuffles
__device__ __forceinline__ unsigned int MatchAny(int v)
{
#if __CUDA_ARCH__>=700
  return __match_any_sync(0xffffffff, v);
#else
  unsigned int m = 0;
  for (int l=0;l<32;l++)
    m |= (unsigned int)(__shfl_sync(0xffffffff, v, l)==v) << l;
  return m;
#endif
}

__device__ __forceinline__ void WarpHistAdd(float *hist, float *buf, int bin, float v)
{
  const int lane = threadIdx.x & 31;
  const unsigned int group = MatchAny(bin);
  buf[lane] = v;
  __syncwarp();
  if (bin>=0 && lane==__ffs(group) - 1) {
    float s = 0.0f;
    for (unsigned int m=group;m;m&=m-1)
      s += buf[__ffs(m) - 1];
    hist[bin] += s;
  }
  __syncwarp();
}

// Divisions rounded to nearest (with --use_fast_math, float divisions aren't), and rounding
// to the nearest integer
template<typename R> __device__ __forceinline__ R DivRN(R a, R b);
template<> __device__ __forceinline__ float DivRN<float>(float a, float b) { return __fdiv_rn(a, b); }
template<> __device__ __forceinline__ double DivRN<double>(double a, double b) { return a/b; }
template<typename R> __device__ __forceinline__ int RoundRN(R a);
template<> __device__ __forceinline__ int RoundRN<float>(float a) { return __float2int_rn(a); }
template<> __device__ __forceinline__ int RoundRN<double>(double a) { return __double2int_rn(a); }

// Separable Gaussian blurs, on tiles in shared memory. OpenCV (sepFilter2D on floats,
// BORDER_REFLECT_101): rows then columns, the row filter sums the taps in order, the column
// filter pairs the symmetric taps, with fused multiply-adds. VLFeat (vl_imconvcol_vf):
// columns then rows, borders by continuity, products and sums rounded separately.
#define ROW_W 128
#define ROW_H 4
#define COL_W 32
#define COL_H 32
#define COL_THREADS_Y 8

template <SiftCompatMode M>
__global__ void BlurRowsTiled(float *dst, const float *src, int w, int h, int p, int radius)
{
  constexpr bool VL = (M==SIFT_COMPAT_VLFEAT);
  extern __shared__ float tile[];
  const int tw = ROW_W + 2*radius;
  const int x0 = blockIdx.x*ROW_W, y0 = blockIdx.y*ROW_H;
  const float *srow = src + min(y0 + (int)threadIdx.y, h - 1)*p;
  for (int tx=threadIdx.x;tx<tw;tx+=ROW_W) {
    const int x = BorderIndex<M>(x0 + tx - radius, w);
    tile[threadIdx.y*tw + tx] = srow[x];
  }
  __syncthreads();
  const int x = x0 + threadIdx.x, y = y0 + threadIdx.y;
  if (x>=w || y>=h)
    return;
  const float *t = tile + threadIdx.y*tw + threadIdx.x;
  float s;
  if constexpr (VL) {
    s = 0.0f;
    for (int k=0;k<=2*radius;k++)
      s = __fadd_rn(s, __fmul_rn(t[k], d_CompatKernel[2*radius - k]));
  } else {
    s = t[0]*d_CompatKernel[0];
    for (int k=1;k<=2*radius;k++)
      s = fmaf(t[k], d_CompatKernel[k], s);
  }
  dst[y*p + x] = s;
}

template <SiftCompatMode M>
__global__ void BlurColumnsTiled(float *dst, const float *src, int w, int h, int p, int radius)
{
  constexpr bool VL = (M==SIFT_COMPAT_VLFEAT);
  extern __shared__ float tile[];
  const int th = COL_H + 2*radius;
  const int x0 = blockIdx.x*COL_W, y0 = blockIdx.y*COL_H;
  const int xs = min(x0 + (int)threadIdx.x, w - 1);
  for (int ty=threadIdx.y;ty<th;ty+=COL_THREADS_Y) {
    const int y = BorderIndex<M>(y0 + ty - radius, h);
    tile[ty*COL_W + threadIdx.x] = src[y*p + xs];
  }
  __syncthreads();
  const int x = x0 + threadIdx.x;
  if (x>=w)
    return;
  for (int yy=threadIdx.y;yy<COL_H;yy+=COL_THREADS_Y) {
    const int y = y0 + yy;
    if (y>=h)
      break;
    const float *t = tile + (yy + radius)*COL_W + threadIdx.x;
    float s;
    if constexpr (VL) {
      s = 0.0f;
      for (int k=-radius;k<=radius;k++)
        s = __fadd_rn(s, __fmul_rn(t[k*COL_W], d_CompatKernel[radius - k]));
    } else {
      s = t[0]*d_CompatKernel[radius];
      for (int k=1;k<=radius;k++)
        s = fmaf(t[k*COL_W] + t[-k*COL_W], d_CompatKernel[radius + k], s);
    }
    dst[y*p + x] = s;
  }
}

template <SiftCompatMode M>
static void BlurRows(float *dst, const float *src, int w, int h, int p, int radius)
{
  dim3 blocks(iDivUp(w, ROW_W), iDivUp(h, ROW_H)), threads(ROW_W, ROW_H);
  BlurRowsTiled<M><<<blocks, threads, ROW_H*(ROW_W + 2*radius)*sizeof(float)>>>(dst, src, w, h, p, radius);
}

template <SiftCompatMode M>
static void BlurColumns(float *dst, const float *src, int w, int h, int p, int radius)
{
  dim3 blocks(iDivUp(w, COL_W), iDivUp(h, COL_H)), threads(COL_W, COL_THREADS_Y);
  BlurColumnsTiled<M><<<blocks, threads, COL_W*(COL_H + 2*radius)*sizeof(float)>>>(dst, src, w, h, p, radius);
}

// Both passes in one kernel, on a tile and its borders in shared memory. OpenCV: rows then
// columns; VLFeat: columns then rows. The same operations per output as the separate passes.
#define FUSED_W 32         // one warp per row of the tile (the kernel assumes 32 threads in x)
#define FUSED_H 32
#define FUSED_THREADS_Y 8
static_assert(FUSED_W==32, "BlurFused uses one warp per row of the tile");

// The window of 4 + 2*RAD samples of a row of a tile, with 16-byte loads (row 16-byte aligned)
template <int RAD>
__device__ __forceinline__ void LoadWindow(float *win, const float *row)
{
#pragma unroll
  for (int m=0;m<(4+2*RAD+3)/4;m++) {
    float4 v = *(const float4 *)&row[4*m];
    if (4*m<4+2*RAD) win[4*m] = v.x;
    if (4*m+1<4+2*RAD) win[4*m+1] = v.y;
    if (4*m+2<4+2*RAD) win[4*m+2] = v.z;
    if (4*m+3<4+2*RAD) win[4*m+3] = v.w;
  }
}

template <SiftCompatMode M, int RAD>
__global__ void BlurFused(float *dst, const float *src, int w, int h, int p)
{
  constexpr bool VL = (M==SIFT_COMPAT_VLFEAT);
  // Each thread computes 4 consecutive outputs of a pass from a window of 4 + 2R samples in
  // registers: along rows with 16-byte loads (rows padded to a multiple of 4 floats), along
  // columns with one load per sample
  constexpr int SW = (FUSED_W + 2*RAD + 3)/4*4;
  __shared__ __align__(16) float S[FUSED_H + 2*RAD][SW];
  __shared__ __align__(16) float I[VL ? FUSED_H : FUSED_H + 2*RAD][VL ? SW : FUSED_W];
  const int x0 = blockIdx.x*FUSED_W, y0 = blockIdx.y*FUSED_H;
  const int tx = threadIdx.x, tid = threadIdx.y*32 + tx;
  for (int ty=threadIdx.y;ty<FUSED_H+2*RAD;ty+=FUSED_THREADS_Y) {
    const int y = BorderIndex<M>(y0 + ty - RAD, h);
    for (int sx=tx;sx<FUSED_W+2*RAD;sx+=32) {
      const int x = BorderIndex<M>(x0 + sx - RAD, w);
      S[ty][sx] = src[y*p + x];
    }
  }
  __syncthreads();
  float win[4 + 2*RAD];
  if constexpr (VL) {
    // Columns over the tile and its left and right borders: 4 rows per thread
    for (int it=tid;it<(FUSED_H/4)*(FUSED_W + 2*RAD);it+=32*FUSED_THREADS_Y) {
      const int sx = it%(FUSED_W + 2*RAD), y4 = (it/(FUSED_W + 2*RAD))*4;
#pragma unroll
      for (int m=0;m<4+2*RAD;m++)
        win[m] = S[y4 + m][sx];
#pragma unroll
      for (int q=0;q<4;q++) {
        float s = 0.0f;
#pragma unroll
        for (int k=0;k<=2*RAD;k++)
          s = __fadd_rn(s, __fmul_rn(win[q + k], d_CompatKernel[2*RAD - k]));
        I[y4 + q][sx] = s;
      }
    }
  } else {
    // Rows over the tile and its top and bottom borders: 4 columns per thread
    for (int it=tid;it<(FUSED_H + 2*RAD)*(FUSED_W/4);it+=32*FUSED_THREADS_Y) {
      const int ty = it/(FUSED_W/4), x4 = (it%(FUSED_W/4))*4;
      LoadWindow<RAD>(win, &S[ty][x4]);
      float4 o;
      float *op = &o.x;
#pragma unroll
      for (int q=0;q<4;q++) {
        float s = win[q]*d_CompatKernel[0];
#pragma unroll
        for (int k=1;k<=2*RAD;k++)
          s = fmaf(win[q + k], d_CompatKernel[k], s);
        op[q] = s;
      }
      *(float4 *)&I[ty][x4] = o;
    }
  }
  __syncthreads();
  // Second pass: 32 rows x 32 columns of outputs
  if constexpr (VL) {
    // Rows: 4 columns per thread
    const int ty = tid/(FUSED_W/4), x4 = (tid%(FUSED_W/4))*4;
    const int y = y0 + ty;
    LoadWindow<RAD>(win, &I[ty][x4]);
#pragma unroll
    for (int q=0;q<4;q++) {
      float s = 0.0f;
#pragma unroll
      for (int k=0;k<=2*RAD;k++)
        s = __fadd_rn(s, __fmul_rn(win[q + k], d_CompatKernel[2*RAD - k]));
      const int x = x0 + x4 + q;
      if (x<w && y<h)
        dst[y*p + x] = s;
    }
  } else {
    // Columns: 4 rows per thread
    const int x = x0 + tx, y4 = threadIdx.y*4;
#pragma unroll
    for (int m=0;m<4+2*RAD;m++)
      win[m] = I[y4 + m][tx];
#pragma unroll
    for (int q=0;q<4;q++) {
      float s = win[q + RAD]*d_CompatKernel[RAD];
#pragma unroll
      for (int k=1;k<=RAD;k++)
        s = fmaf(win[q + RAD + k] + win[q + RAD - k], d_CompatKernel[RAD + k], s);
      const int y = y0 + y4 + q;
      if (x<w && y<h)
        dst[y*p + x] = s;
    }
  }
}

// The radii of the fused blurs (BlurFused), 1 to FUSED_MAX_RADIUS
#define FUSED_MAX_RADIUS 16
#define FUSED_RADII(F) F(1) F(2) F(3) F(4) F(5) F(6) F(7) F(8) F(9) F(10) F(11) F(12) F(13) F(14) F(15) F(16)
static_assert(FUSED_MAX_RADIUS==16, "FUSED_RADII must list the radii 1 to FUSED_MAX_RADIUS");

// Gaussian blur of src into dst, both passes in one kernel for radii up to FUSED_MAX_RADIUS,
// separate passes through tmp otherwise or in place (other blocks still read the borders of
// a fused block's tile)
template <SiftCompatMode M>
static void BlurLevel(float *dst, const float *src, float *tmp, int w, int h, int p, int radius)
{
  constexpr bool VL = (M==SIFT_COMPAT_VLFEAT);
  if (radius>=1 && radius<=FUSED_MAX_RADIUS && dst!=src) {
    dim3 blocks(iDivUp(w, FUSED_W), iDivUp(h, FUSED_H)), threads(32, FUSED_THREADS_Y);
    switch (radius) {
#define FUSED_CASE(RAD) case RAD: BlurFused<M, RAD><<<blocks, threads>>>(dst, src, w, h, p); break;
      FUSED_RADII(FUSED_CASE)
#undef FUSED_CASE
    }
  } else if constexpr (VL) {
    BlurColumns<SIFT_COMPAT_VLFEAT>(tmp, src, w, h, p, radius);
    BlurRows<SIFT_COMPAT_VLFEAT>(dst, tmp, w, h, p, radius);
  } else {
    BlurRows<SIFT_COMPAT_OPENCV>(tmp, src, w, h, p, radius);
    BlurColumns<SIFT_COMPAT_OPENCV>(dst, tmp, w, h, p, radius);
  }
}

// ---------------------------------------------------------------------------------------
// OpenCV's arithmetic: cv::fastAtan2, expf, the 3x3 solve and the refinement of the extrema

// cv::fastAtan2 (degrees in [0, 360))
__device__ __forceinline__ float FastAtan2OpenCV(float y, float x)
{
  const float p1 = 0.9997878412794807f*(float)(180/M_PI);
  const float p3 = -0.3258083974640975f*(float)(180/M_PI);
  const float p5 = 0.1555786518463281f*(float)(180/M_PI);
  const float p7 = -0.04432655554792128f*(float)(180/M_PI);
  float ax = fabsf(x), ay = fabsf(y);
  float a, c, c2;
  if (ax>=ay) {
    c = __fdiv_rn(ay, ax + (float)DBL_EPSILON);
    c2 = c*c;
    a = (((p7*c2 + p5)*c2 + p3)*c2 + p1)*c;
  } else {
    c = __fdiv_rn(ax, ay + (float)DBL_EPSILON);
    c2 = c*c;
    a = 90.f - (((p7*c2 + p5)*c2 + p3)*c2 + p1)*c;
  }
  if (x<0)
    a = 180.f - a;
  if (y<0)
    a = 360.f - a;
  return a;
}

// expf within about 1 ulp (Cephes' expf), where --use_fast_math would substitute __expf
__device__ __forceinline__ float ExpfOpenCV(float x)
{
  float t = rintf(x*1.44269504088896341f);
  float f = fmaf(t, -0.693359375f, x);
  f = fmaf(t, 2.12194440e-4f, f);
  float p = 1.9875691500E-4f;
  p = fmaf(p, f, 1.3981999507E-3f);
  p = fmaf(p, f, 8.3334519073E-3f);
  p = fmaf(p, f, 4.1665795894E-2f);
  p = fmaf(p, f, 1.6666665459E-1f);
  p = fmaf(p, f, 5.0000001201E-1f);
  p = fmaf(p, f*f, f) + 1.0f;
  int i = (int)t;
  return (i<-126 ? 0.0f : p*__int_as_float((i + 127) << 23));
}

// 3x3 solve as cv::Matx<float, 3, 3>::solve(DECOMP_LU) (Cramer's rule; zero if singular)
template<typename R>
__device__ void Solve3OpenCV(const R a[3][3], const R b[3], R x[3])
{
  R d = a[0][0]*(a[1][1]*a[2][2] - a[2][1]*a[1][2]) -
    a[0][1]*(a[1][0]*a[2][2] - a[2][0]*a[1][2]) +
    a[0][2]*(a[1][0]*a[2][1] - a[2][0]*a[1][1]);
  if (d==0) {
    x[0] = x[1] = x[2] = R(0);
    return;
  }
  d = DivRN<R>(R(1), d);
  x[0] = d*(b[0]*(a[1][1]*a[2][2] - a[1][2]*a[2][1]) -
            a[0][1]*(b[1]*a[2][2] - a[1][2]*b[2]) +
            a[0][2]*(b[1]*a[2][1] - a[1][1]*b[2]));
  x[1] = d*(a[0][0]*(b[1]*a[2][2] - a[1][2]*b[2]) -
            b[0]*(a[1][0]*a[2][2] - a[1][2]*a[2][0]) +
            a[0][2]*(a[1][0]*b[2] - b[1]*a[2][0]));
  x[2] = d*(a[0][0]*(a[1][1]*b[2] - b[1]*a[2][1]) -
            a[0][1]*(a[1][0]*b[2] - b[1]*a[2][0]) +
            b[0]*(a[1][0]*a[2][1] - a[1][1]*a[2][0]));
}

// OpenCV's adjustLocalExtrema: up to 5 interpolation steps, then the contrast and edge tests.
// layer is the level index in CompatPyramid (OpenCV's layer + lev[o]).
template<typename R>
__device__ bool RefineOpenCV(const CompatPyramid &P, int o, int layer, int r, int c,
                             const CompatDetectParams &dp, OpenCVKeypoint &kp)
{
  const R img_scale = R(1.)/255;
  const R deriv_scale = img_scale*R(0.5);
  const R second_deriv_scale = img_scale;
  const R cross_deriv_scale = img_scale*R(0.25);
  const int w = P.w[o], h = P.h[o], p = P.p[o];
  const int L = dp.S;
  R xi = 0, xr = 0, xc = 0, contr = 0;
  int i = 0;
  for (;i<SIFT_MAX_INTERP_STEPS;i++) {
    const DogView prev = Dog(P, o, layer-1), cur = Dog(P, o, layer), next = Dog(P, o, layer+1);
    const DogView ir = cur + r*p;
    R dD[3] = {(ir[c+1] - ir[c-1])*deriv_scale, (ir[c+p] - ir[c-p])*deriv_scale,
                   (next[r*p+c] - prev[r*p+c])*deriv_scale};
    R v2 = ir[c]*2;
    R dxx = (ir[c+1] + ir[c-1] - v2)*second_deriv_scale;
    R dyy = (ir[c+p] + ir[c-p] - v2)*second_deriv_scale;
    R dss = (next[r*p+c] + prev[r*p+c] - v2)*second_deriv_scale;
    R dxy = (ir[c+p+1] - ir[c+p-1] - ir[c-p+1] + ir[c-p-1])*cross_deriv_scale;
    R dxs = (next[r*p+c+1] - next[r*p+c-1] - prev[r*p+c+1] + prev[r*p+c-1])*cross_deriv_scale;
    R dys = (next[(r+1)*p+c] - next[(r-1)*p+c] - prev[(r+1)*p+c] + prev[(r-1)*p+c])*cross_deriv_scale;
    R H[3][3] = {{dxx, dxy, dxs}, {dxy, dyy, dys}, {dxs, dys, dss}};
    R X[3];
    Solve3OpenCV(H, dD, X);
    xi = -X[2];
    xr = -X[1];
    xc = -X[0];
    if (fabs(xi)<R(0.5) && fabs(xr)<R(0.5) && fabs(xc)<R(0.5))
      break;
    if (fabs(xi)>(R)(INT_MAX/3) || fabs(xr)>(R)(INT_MAX/3) || fabs(xc)>(R)(INT_MAX/3))
      return false;
    c += RoundRN<R>(xc);
    r += RoundRN<R>(xr);
    layer += RoundRN<R>(xi);
    if (layer<1 || layer>L + P.lev[o] || c<SIFT_IMG_BORDER || c>=w - SIFT_IMG_BORDER ||
        r<SIFT_IMG_BORDER || r>=h - SIFT_IMG_BORDER)
      return false;
  }
  if (i>=SIFT_MAX_INTERP_STEPS)
    return false;
  const DogView prev = Dog(P, o, layer-1), cur = Dog(P, o, layer), next = Dog(P, o, layer+1);
  const DogView ir = cur + r*p;
  R dD[3] = {(ir[c+1] - ir[c-1])*deriv_scale, (ir[c+p] - ir[c-p])*deriv_scale,
                 (next[r*p+c] - prev[r*p+c])*deriv_scale};
  R t = dD[0]*xc + dD[1]*xr + dD[2]*xi;
  contr = ir[c]*img_scale + t*R(0.5);
  if (fabs(contr)*L<dp.contrastThreshold)
    return false;
  R v2 = ir[c]*R(2.);
  R dxx = (ir[c+1] + ir[c-1] - v2)*second_deriv_scale;
  R dyy = (ir[c+p] + ir[c-p] - v2)*second_deriv_scale;
  R dxy = (ir[c+p+1] - ir[c+p-1] - ir[c-p+1] + ir[c-p-1])*cross_deriv_scale;
  R tr = dxx + dyy;
  R det = dxx*dyy - dxy*dxy;
  const R et = (float)dp.edgeThreshold;
  if (det<=0 || tr*tr*et>=(et + 1)*(et + 1)*det)
    return false;
  kp.x = (float)((c + xc)*P.pix[o]);
  kp.y = (float)((r + xr)*P.pix[o]);
  kp.packedOctave = o + (layer << 8) + (RoundRN<R>((xi + R(0.5))*255) << 16);
  kp.size = (float)(dp.sigma*(R)pow(2.0, (double)DivRN<R>(layer - P.lev[o] + xi, (R)L))*P.pix[o]*2);
  kp.response = (float)fabs(contr);
  kp.contr = (float)contr;
  kp.edge = (float)DivRN<R>(tr*tr, det);
  kp.angle = 0.0f;
  kp.o = o;
  kp.layer = layer;
  kp.r = r;
  kp.c = c;
  return true;
}

// ---------------------------------------------------------------------------------------
// VLFeat's arithmetic (vl/sift.c of VLFeat 0.9.21). Where VLFeat rounds a product and a sum
// separately (its x86 builds use no fused multiply-adds), the _rn intrinsics keep them apart.

// The arithmetic of VLFeat's refinement, orientations and descriptors: in double, rounded as
// VLFeat rounds it, or in float (the default)
template<typename R> struct VLFeatArith;
template<> struct VLFeatArith<double> {
  static constexpr double pi = VL_PI, eps = VL_EPSILON_D;
  static __device__ __forceinline__ double add(double a, double b) { return __dadd_rn(a, b); }
  static __device__ __forceinline__ double sub(double a, double b) { return __dsub_rn(a, b); }
  static __device__ __forceinline__ double mul(double a, double b) { return __dmul_rn(a, b); }
};
template<> struct VLFeatArith<float> {
  static constexpr float pi = 3.14159265f, eps = VL_EPSILON_F;
  static __device__ __forceinline__ float add(float a, float b) { return a + b; }
  static __device__ __forceinline__ float sub(float a, float b) { return a - b; }
  static __device__ __forceinline__ float mul(float a, float b) { return a*b; }
};

// In global memory: each lane reads its own entry, which constant memory would serialize
__device__ double d_ExpnTab[VL_EXPN_SZ + 1];   // exp(-x) at x = k*VL_EXPN_MAX/VL_EXPN_SZ
__device__ float d_ExpnTabF[VL_EXPN_SZ + 1];

// fast_expn: exp(-x) by linear interpolation in a table (d_ExpnTab, d_ExpnTabF)
template<typename R> __device__ __forceinline__ R FastExpnVLFeat(R x);
template<> __device__ __forceinline__ double FastExpnVLFeat<double>(double x)
{
  if (x>VL_EXPN_MAX)
    return 0.0;
  x *= VL_EXPN_SZ/VL_EXPN_MAX;
  int i = (int)floor(x);
  double r = __dsub_rn(x, (double)i);
  double a = __ldg(&d_ExpnTab[i]), b = __ldg(&d_ExpnTab[i + 1]);
  return __dadd_rn(a, __dmul_rn(r, __dsub_rn(b, a)));
}
template<> __device__ __forceinline__ float FastExpnVLFeat<float>(float x)
{
  if (x>(float)VL_EXPN_MAX)
    return 0.0f;
  x *= (float)(VL_EXPN_SZ/VL_EXPN_MAX);
  int i = (int)floorf(x);
  float r = __fsub_rn(x, (float)i);
  float a = __ldg(&d_ExpnTabF[i]), b = __ldg(&d_ExpnTabF[i + 1]);
  return __fadd_rn(a, __fmul_rn(r, __fsub_rn(b, a)));
}

// vl_fast_atan2_f (with a fast division, within 2 ulps: faster, and the features stay as close)
__device__ __forceinline__ float FastAtan2VLFeat(float y, float x)
{
  const float c3 = 0.1821F, c1 = 0.9675F;
  float abs_y = fabsf(y) + VL_EPSILON_F;
  float angle, r;
  if (x>=0) {
    r = __fdividef(__fsub_rn(x, abs_y), __fadd_rn(x, abs_y));
    angle = (float)(VL_PI/4);
  } else {
    r = __fdividef(__fadd_rn(x, abs_y), __fsub_rn(abs_y, x));
    angle = (float)(3*VL_PI/4);
  }
  angle = __fadd_rn(angle, __fmul_rn(__fsub_rn(__fmul_rn(__fmul_rn(c3, r), r), c1), r));
  return (y<0) ? -angle : angle;
}

// vl_fast_sqrt_f
__device__ __forceinline__ float FastSqrtVLFeat(float x)
{
  if (x<1e-8)
    return 0;
  float xhalf = 0.5f*x;
  float u = __int_as_float(0x5f3759df - (__float_as_int(x) >> 1));
  u = __fmul_rn(u, __fsub_rn(1.5f, __fmul_rn(__fmul_rn(xhalf, u), u)));
  u = __fmul_rn(u, __fsub_rn(1.5f, __fmul_rn(__fmul_rn(xhalf, u), u)));
  return __fmul_rn(x, u);
}

// vl_mod_2pi_f
__device__ __forceinline__ float Mod2PiVLFeat(float x)
{
  while (x>(float)(2*VL_PI))
    x -= (float)(2*VL_PI);
  while (x<0.0F)
    x += (float)(2*VL_PI);
  return x;
}

// A sample of a row of copy_and_upsample_rows: odd samples average two pixels, except the last
// The input in [0, 1], as COLMAP converts it for VLFeat (x/255.0f)
__device__ __forceinline__ float UnitVLFeat(float v)
{
  return __fdiv_rn(v, 255.0f);
}

__device__ __forceinline__ float UpsampleRowVLFeat(const float *row, int x0, int odd, int sw)
{
  if (odd && x0<sw - 1)
    return 0.5f*__fadd_rn(UnitVLFeat(row[x0]), UnitVLFeat(row[x0 + 1]));
  return UnitVLFeat(row[x0]);
}

// update_gradient's magnitude and angle at (x, y), computed where needed instead of for whole
// levels
__device__ __forceinline__ float2 GradientVLFeat(const float *src, int x, int y, int w, int h, int p)
{
  const float *s = src + y*p + x;
  float gx, gy;
  if (x==0)
    gx = s[1] - s[0];
  else if (x==w - 1)
    gx = s[0] - s[-1];
  else
    gx = 0.5f*(s[1] - s[-1]);
  if (y==0)
    gy = s[p] - s[0];
  else if (y==h - 1)
    gy = s[0] - s[-p];
  else
    gy = 0.5f*(s[p] - s[-p]);
  float2 g;
  g.x = FastSqrtVLFeat(__fadd_rn(__fmul_rn(gx, gx), __fmul_rn(gy, gy)));
  g.y = Mod2PiVLFeat((float)((double)FastAtan2VLFeat(gy, gx) + 2*VL_PI));
  return g;
}

// vl_sift_detect's refinement of an extremum at (x, y) of level index si of octave o (VLFeat's
// level s = si + s_min - lev[o]): up to 5 steps of Gauss elimination with moves of +-1 pixel,
// then the peak and edge tests
template<typename R>
__device__ bool RefineVLFeat(const CompatPyramid &P, int o, int layer, int r, int c,
                             const CompatDetectParams &dp, VLFeatKeypoint &kp)
{
  using V = VLFeatArith<R>;
  const int w = P.w[o], h = P.h[o], p = P.p[o];
  const int s = layer + dp.s_min - P.lev[o];   // VLFeat's level
  const int raster = r*w + c;
  int x = c, y = r;
  R Dx=0, Dy=0, Ds=0, Dxx=0, Dyy=0, Dss=0, Dxy=0, Dxs=0, Dys=0;
  R A[9], b[3];
  int ddx = 0, ddy = 0;
  const DogView prev = Dog(P, o, layer-1), cur = Dog(P, o, layer), next = Dog(P, o, layer+1);
  for (int iter=0;iter<VL_MAX_REFINE_STEPS;iter++) {
    x += ddx;
    y += ddy;
    const int c = y*p + x;
#define AT1(dx, dy) cur[c + (dy)*p + (dx)]
    Dx = R(0.5)*(R)(AT1(1, 0) - AT1(-1, 0));
    Dy = R(0.5)*(R)(AT1(0, 1) - AT1(0, -1));
    Ds = R(0.5)*(R)(next[c] - prev[c]);
    Dxx = V::sub((R)(AT1(1, 0) + AT1(-1, 0)), R(2.0)*(R)AT1(0, 0));
    Dyy = V::sub((R)(AT1(0, 1) + AT1(0, -1)), R(2.0)*(R)AT1(0, 0));
    Dss = V::sub((R)(next[c] + prev[c]), R(2.0)*(R)AT1(0, 0));
    Dxy = R(0.25)*(R)(AT1(1, 1) + AT1(-1, -1) - AT1(-1, 1) - AT1(1, -1));
    Dxs = R(0.25)*(R)(next[c + 1] + prev[c - 1] - next[c - 1] - prev[c + 1]);
    Dys = R(0.25)*(R)(next[c + p] + prev[c - p] - next[c - p] - prev[c + p]);
#undef AT1
#define Aat(i, j) (A[(i) + (j)*3])
    Aat(0, 0) = Dxx;
    Aat(1, 1) = Dyy;
    Aat(2, 2) = Dss;
    Aat(0, 1) = Aat(1, 0) = Dxy;
    Aat(0, 2) = Aat(2, 0) = Dxs;
    Aat(1, 2) = Aat(2, 1) = Dys;
    b[0] = -Dx;
    b[1] = -Dy;
    b[2] = -Ds;
    // Gauss elimination with partial pivoting, as VLFeat
    for (int j=0;j<3;j++) {
      R maxa = 0, maxabsa = 0;
      int maxi = -1;
      for (int i=j;i<3;i++) {
        R a = Aat(i, j), absa = fabs(a);
        if (absa>maxabsa) {
          maxa = a;
          maxabsa = absa;
          maxi = i;
        }
      }
      if (maxabsa<1e-10f) {
        b[0] = b[1] = b[2] = 0;
        break;
      }
      int i = maxi;
      const R inv = R(1.0)/maxa;   // VLFeat divides by the pivot: the reciprocal is faster
      for (int jj=j;jj<3;jj++) {
        R tmp = Aat(i, jj);
        Aat(i, jj) = Aat(j, jj);
        Aat(j, jj) = tmp;
        Aat(j, jj) = V::mul(Aat(j, jj), inv);
      }
      R tmp = b[j];
      b[j] = b[i];
      b[i] = tmp;
      b[j] = V::mul(b[j], inv);
      for (int ii=j+1;ii<3;ii++) {
        R xx = Aat(ii, j);
        for (int jj=j;jj<3;jj++)
          Aat(ii, jj) = V::sub(Aat(ii, jj), V::mul(xx, Aat(j, jj)));
        b[ii] = V::sub(b[ii], V::mul(xx, b[j]));
      }
    }
    for (int i=2;i>0;i--) {
      R xx = b[i];
      for (int ii=i-1;ii>=0;ii--)
        b[ii] = V::sub(b[ii], V::mul(xx, Aat(ii, i)));
    }
#undef Aat
    ddx = ((b[0]>R(0.6) && x<w - 2) ? 1 : 0) + ((b[0]<-R(0.6) && x>1) ? -1 : 0);
    ddy = ((b[1]>R(0.6) && y<h - 2) ? 1 : 0) + ((b[1]<-R(0.6) && y>1) ? -1 : 0);
    if (ddx==0 && ddy==0)
      break;
  }
  R val = cur[y*p + x] + R(0.5)*(V::add(V::add(V::mul(Dx, b[0]), V::mul(Dy, b[1])), V::mul(Ds, b[2])));
  R score = DivRN<R>((Dxx + Dyy)*(Dxx + Dyy), V::sub(V::mul(Dxx, Dyy), V::mul(Dxy, Dxy)));
  R xn = x + b[0], yn = y + b[1], sn = s + b[2];
  const R te = dp.edgeThreshold;
  // The level is bounded by the lowest one, s_min (lower with the extension), and s_max
  bool good = fabs(val)>dp.peakThreshold && score<DivRN<R>((te + 1)*(te + 1), te) && score>=0 &&
    fabs(b[0])<R(1.5) && fabs(b[1])<R(1.5) && fabs(b[2])<R(1.5) &&
    xn>=0 && xn<=w - 1 && yn>=0 && yn<=h - 1 && sn>=dp.s_min - P.lev[o] && sn<=dp.s_max;
  if (!good)
    return false;
  const R xper = P.pix[o];
  kp.raster = raster;
  kp.o = o;
  kp.is = s;
  kp.x = (float)(xn*xper);
  kp.y = (float)(yn*xper);
  kp.sigma = (float)(dp.sigma0*pow(2.0, (double)DivRN<R>(sn, (R)dp.S))*(double)xper);
  kp.angle = R(0.0);
  kp.val = (float)val;
  kp.score = (float)score;
  kp.ori = 0;
  return true;
}

// ---------------------------------------------------------------------------------------
// The kernels of the pipeline and their per-mode steps: the scale space (the input image, the
// downsampling), the extrema, the orientations and the descriptors

// The first octave's input image, upsampled by 2 or not. OpenCV: bilinear (warpAffine with
// INTER_LINEAR), values 0 to 255. VLFeat (copy_and_upsample_rows): rows first, then columns,
// odd samples average two pixels except the last one that repeats the last pixel, values 0 to 1.
template<SiftCompatMode M>
__global__ void InputCompat(float *dst, int dw, int dh, int dp, const float *src, int sw, int sh, int sp, int upsample)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x>=dw || y>=dh)
    return;
  if constexpr (M==SIFT_COMPAT_VLFEAT) {
    if (!upsample) {
      dst[y*dp + x] = UnitVLFeat(src[y*sp + x]);
      return;
    }
    int x0 = x>>1, y0 = y>>1;
    float h0 = UpsampleRowVLFeat(src + y0*sp, x0, x&1, sw);
    if ((y&1) && y0<sh - 1)
      h0 = 0.5f*__fadd_rn(h0, UpsampleRowVLFeat(src + (y0 + 1)*sp, x0, x&1, sw));
    dst[y*dp + x] = h0;
  } else {
    if (!upsample) {
      dst[y*dp + x] = src[y*sp + x];
      return;
    }
    int x0 = x>>1, y0 = y>>1;
    int x1 = (x&1) ? min(x0 + 1, sw - 1) : x0;
    int y1 = (y&1) ? min(y0 + 1, sh - 1) : y0;
    float fx = (x&1) ? 0.5f : 0.0f, fy = (y&1) ? 0.5f : 0.0f;
    float v00 = src[y0*sp + x0], v01 = src[y0*sp + x1];
    float v10 = src[y1*sp + x0], v11 = src[y1*sp + x1];
    dst[y*dp + x] = v00*((1.0f - fx)*(1.0f - fy)) + v01*(fx*(1.0f - fy)) + v10*((1.0f - fx)*fy) + v11*(fx*fy);
  }
}

// Next octave's first level: every other pixel (resize INTER_NEAREST to half size)
__global__ void DownsampleCompat(float *dst, int dw, int dh, int dp, const float *src, int sp)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x>=dw || y>=dh)
    return;
  dst[y*dp + x] = src[2*y*sp + 2*x];
}

// Extrema of level `layer` (array index) of octave o over the 26 neighbours, refined and
// tested. OpenCV: above its threshold, and not smaller (larger for minima) than any
// neighbour. VLFeat: at least 0.8 times the peak threshold, and strictly larger (smaller)
// than all the neighbours (with a peak threshold of 0, a value of 0 is tested as both a
// maximum and a minimum). The level itself is tested first: most pixels fail there.
template<SiftCompatMode M, typename R>
__global__ void FindExtremaCompat(CompatPyramid P, int o, int layer, CompatDetectParams dp,
                                  typename CompatModeTraits<M>::Keypoint *cands, unsigned int *counter, int maxCands)
{
  constexpr int B = CompatModeTraits<M>::border;
  const int w = P.w[o], h = P.h[o], p = P.p[o];
  int c = blockIdx.x*blockDim.x + threadIdx.x + B;
  int r = blockIdx.y*blockDim.y + threadIdx.y + B;
  if (c>=w - B || r>=h - B)
    return;
  const DogView prev = Dog(P, o, layer-1) + (r*p + c);
  const DogView cur = Dog(P, o, layer) + (r*p + c);
  const DogView next = Dog(P, o, layer+1) + (r*p + c);
  const float v = cur[0];
  bool isMax, isMin;
  if constexpr (M==SIFT_COMPAT_VLFEAT) {
    isMax = (v>=dp.peakPrefilter);
    isMin = (v<=-dp.peakPrefilter);
  } else {
    if (!(fabsf(v)>dp.preThreshold))
      return;
    isMax = (v>0);
    isMin = !isMax;
  }
  const int offs[9] = {-p-1, -p, -p+1, -1, 1, p-1, p, p+1, 0};
  bool found = false;
  for (int pass=0;pass<2 && !found;pass++) {
    if (!(pass==0 ? isMax : isMin))
      continue;
    const float sgn = (pass==0 ? 1.0f : -1.0f);
    const float vs = sgn*v;
    found = true;
    if constexpr (M==SIFT_COMPAT_VLFEAT) {
      for (int k=0;k<8 && found;k++)
        found = (vs>sgn*cur[offs[k]]);
      for (int k=0;k<9 && found;k++)
        found = (vs>sgn*prev[offs[k]]) && (vs>sgn*next[offs[k]]);
    } else {
      for (int k=0;k<8 && found;k++)
        found = !(vs<sgn*cur[offs[k]]);
      for (int k=0;k<9 && found;k++)
        found = !(vs<sgn*prev[offs[k]] || vs<sgn*next[offs[k]]);
    }
  }
  if (!found)
    return;
  typename CompatModeTraits<M>::Keypoint kp;
  bool ok;
  if constexpr (M==SIFT_COMPAT_VLFEAT)
    ok = RefineVLFeat<R>(P, o, layer, r, c, dp, kp);
  else
    ok = RefineOpenCV<R>(P, o, layer, r, c, dp, kp);
  if (!ok)
    return;
  unsigned int k = atomicAdd(counter, 1);
  if (k<(unsigned int)maxCands)
    cands[k] = kp;
}

// The orientation histogram of one keypoint, one partial histogram per lane: OpenCV's
// (calcOrientationHist: gradients in a square window, Gaussian weights, nearest bin) and
// VLFeat's (vl_sift_calc_keypoint_orientations: gradients in a disc, fast_expn weights,
// bilinear binning; false if the keypoint is outside its level)
template<typename R>
__device__ void OrientationHistOpenCV(const CompatPyramid &P, const OpenCVKeypoint &kp, R *hist, int lane)
{
  constexpr int n = SIFT_ORI_HIST_BINS;
  const int o = kp.o;
  const float *img = P.gauss[o][kp.layer];
  const int rows = P.h[o], cols = P.w[o], p = P.p[o];
  float scl_octv = kp.size*0.5f*__frcp_rn(P.pix[o]);
  int radius = __float2int_rn(SIFT_ORI_RADIUS*scl_octv);
  float sigma = SIFT_ORI_SIG_FCTR*scl_octv;
  float expf_scale = __fdiv_rn(-1.f, 2.f*sigma*sigma);
  int side = 2*radius + 1;
  for (int k=lane;k<side*side;k+=32) {
    int i = k/side - radius, j = k%side - radius;
    int y = kp.r + i, x = kp.c + j;
    if (y<=0 || y>=rows - 1 || x<=0 || x>=cols - 1)
      continue;
    float dx = img[y*p + x + 1] - img[y*p + x - 1];
    float dy = img[(y - 1)*p + x] - img[(y + 1)*p + x];
    float wgt = ExpfOpenCV((i*i + j*j)*expf_scale);
    float ori = FastAtan2OpenCV(dy, dx);
    float mag = __fsqrt_rn(fmaf(dx, dx, dy*dy));
    int bin = __float2int_rn((n/360.f)*ori);
    if (bin>=n)
      bin -= n;
    if (bin<0)
      bin += n;
    hist[bin] += wgt*mag;
  }
}

template<typename R>
__device__ bool OrientationHistVLFeat(const CompatPyramid &P, const CompatDetectParams &dp, const VLFeatKeypoint &kp,
                                      R *hist, R *dx2, R *dy2, int lane)
{
  using V = VLFeatArith<R>;
  constexpr int n = SIFT_ORI_HIST_BINS;
  const int o = kp.o;
  const int w = P.w[o], h = P.h[o], p = P.p[o];
  const float *gauss = P.gauss[o][LevelIndexVLFeat(P, dp, kp)];
  const R xper = P.pix[o];
  R x = kp.x/xper, y = kp.y/xper, sigma = kp.sigma/xper;
  int xi = (int)(x + R(0.5)), yi = (int)(y + R(0.5));
  const R sigmaw = R(1.5)*sigma;
  int W = min(max((int)floor(R(3.0)*sigmaw), 1), VL_MAX_W);
  if (xi<0 || xi>w - 1 || yi<0 || yi>h - 1)
    return false;
  // Squared offsets, as VLFeat rounds them
  for (int t=lane;t<2*W+1;t+=32) {
    R dx = (R)(xi + t - W) - x, dy = (R)(yi + t - W) - y;
    dx2[t] = V::mul(dx, dx);
    dy2[t] = V::mul(dy, dy);
  }
  __syncwarp();
  int ys0 = max(-W, -yi), ys1 = min(W, h - 1 - yi);
  int xs0 = max(-W, -xi), xs1 = min(W, w - 1 - xi);
  int nx = xs1 - xs0 + 1, ny = ys1 - ys0 + 1;
  // Here and below, reciprocals where VLFeat divides: faster, and the features stay as close
  const R invw = 1/(2*sigmaw*sigmaw);
  for (int t=lane;t<nx*ny;t+=32) {
    int ys = ys0 + t/nx, xs = xs0 + t%nx;
    R r2 = V::add(dx2[xs + W], dy2[ys + W]);
    if (r2>=W*W + R(0.6))
      continue;
    R wgt = FastExpnVLFeat<R>(r2*invw);
    float2 g = GradientVLFeat(gauss, xi + xs, yi + ys, w, h, p);
    R mod = g.x, ang = g.y;
    R fbin = n*ang*(1/(2*V::pi));
    int bin = (int)floor(fbin - R(0.5));   // VL_SIFT_BILINEAR_ORIENTATIONS, set in sift.c
    R rbin = fbin - bin - R(0.5);
    hist[(bin + n)%n] += V::mul(V::mul(1 - rbin, mod), wgt);
    hist[(bin + 1)%n] += V::mul(V::mul(rbin, mod), wgt);
  }
  return true;
}

// The keypoints of the peaks of the summed histogram t (2 bins of padding on each side):
// OpenCV's ([1 4 6 4 1] smoothing into smooth, all the peaks of at least 0.8 times the
// maximum) and VLFeat's (six box filters, the first VL_MAX_ORIENTATION_PEAKS peaks above 0.8
// times the maximum, of which the first maxOrientations are kept)
template<typename R>
__device__ void OrientationPeaksOpenCV(R *t, const OpenCVKeypoint &kp, int lane, R *smooth, OpenCVKeypoint *kpts,
                                       unsigned int *counter, int maxKpts)
{
  constexpr int n = SIFT_ORI_HIST_BINS;
  if (lane==0) {
    t[-1] = t[n-1];
    t[-2] = t[n-2];
    t[n] = t[0];
    t[n+1] = t[1];
  }
  __syncwarp();
  for (int b=lane;b<n;b+=32)
    smooth[b] = fma(t[b-2] + t[b+2], R(1)/R(16), fma(t[b-1] + t[b+1], R(4)/R(16), t[b]*(R(6)/R(16))));
  __syncwarp();
  if (lane==0) {
    R omax = smooth[0];
    for (int b=1;b<n;b++)
      omax = fmax(omax, smooth[b]);
    R mag_thr = omax*(R)SIFT_ORI_PEAK_RATIO;
    for (int j=0;j<n;j++) {
      int l = j>0 ? j - 1 : n - 1;
      int r2 = j<n-1 ? j + 1 : 0;
      if (smooth[j]>smooth[l] && smooth[j]>smooth[r2] && smooth[j]>=mag_thr) {
        R bin = j + DivRN<R>(R(0.5)*(smooth[l] - smooth[r2]), smooth[l] - 2*smooth[j] + smooth[r2]);
        bin = bin<0 ? n + bin : bin>=n ? bin - n : bin;
        OpenCVKeypoint out = kp;
        out.angle = 360.f - (float)((R(360)/n)*bin);
        if (fabsf(out.angle - 360.f)<FLT_EPSILON)
          out.angle = 0.f;
        unsigned int m = atomicAdd(counter, 1);
        if (m<(unsigned int)maxKpts)
          kpts[m] = out;
      }
    }
  }
}

template<typename R>
__device__ void OrientationPeaksVLFeat(R *t, const VLFeatKeypoint &kp, int lane, int maxOrientations, VLFeatKeypoint *kpts,
                                       unsigned int *counter, int maxKpts)
{
  using V = VLFeatArith<R>;
  constexpr int n = SIFT_ORI_HIST_BINS;
  // Six circular box filters: each bin from the previous values of its neighbours, as VLFeat
  // (its loop keeps the previous value of the left neighbour)
  for (int iter=0;iter<6;iter++) {
    R nh[2];
    for (int k=0;k<2;k++) {
      int i = lane + 32*k;
      if (i<n)
        nh[k] = V::mul(V::add(V::add(t[(i + n - 1)%n], t[i]), t[(i + 1)%n]), R(1.0)/R(3.0));
    }
    __syncwarp();
    for (int k=0;k<2;k++)
      if (lane + 32*k<n)
        t[lane + 32*k] = nh[k];
    __syncwarp();
  }
  if (lane==0) {
    R maxh = 0;
    for (int i=0;i<n;i++)
      maxh = fmax(maxh, t[i]);
    int nangles = 0;
    for (int i=0;i<n && nangles<VL_MAX_ORIENTATION_PEAKS;i++) {
      R h0 = t[i], hm = t[(i - 1 + n)%n], hp = t[(i + 1 + n)%n];
      if (h0>R(0.8)*maxh && h0>hm && h0>hp) {
        R di = DivRN<R>(-R(0.5)*V::sub(hp, hm), V::sub(V::add(hp, hm), 2*h0));
        R th = DivRN<R>(2*V::pi*(i + di + R(0.5)), (R)n);
        if (nangles<maxOrientations) {
          VLFeatKeypoint out = kp;
          out.angle = th;
          out.ori = nangles;
          unsigned int m = atomicAdd(counter, 1);
          if (m<(unsigned int)maxKpts)
            kpts[m] = out;
        }
        nangles++;
      }
    }
  }
}

// Orientations, one warp per candidate and one keypoint per peak of its histogram of 36
// bins. R: the precision of the histograms (and, with VLFeat, of the weights and offsets).
template<SiftCompatMode M, typename R>
__global__ void OrientationsCompat(CompatPyramid P, CompatDetectParams dp, int maxOrientations,
                                   const typename CompatModeTraits<M>::Keypoint *cands, int numCands,
                                   typename CompatModeTraits<M>::Keypoint *kpts, unsigned int *counter, int maxKpts)
{
  constexpr bool VL = (M==SIFT_COMPAT_VLFEAT);
  constexpr int NW = CompatModeTraits<M>::oriWarps;
  constexpr int n = SIFT_ORI_HIST_BINS;
  constexpr int TW = (VL ? 2*VL_MAX_W + 1 : 1);   // VLFeat: squared offsets
  __shared__ R lanehist[NW][32][n + 1];
  __shared__ R hist[NW][n + 4];
  __shared__ R smooth[NW][n];
  __shared__ R dx2[NW][TW], dy2[NW][TW];
  const int warp = threadIdx.x/32, lane = threadIdx.x%32;
  const int idx = blockIdx.x*NW + warp;
  if (idx>=numCands)
    return;
  const typename CompatModeTraits<M>::Keypoint kp = cands[idx];
  for (int b=0;b<n;b++)
    lanehist[warp][lane][b] = R(0.0);
  if constexpr (VL) {
    if (!OrientationHistVLFeat<R>(P, dp, kp, lanehist[warp][lane], dx2[warp], dy2[warp], lane))
      return;
  } else
    OrientationHistOpenCV<R>(P, kp, lanehist[warp][lane], lane);
  __syncwarp();
  for (int b=lane;b<n;b+=32) {
    R s = R(0.0);
    for (int l=0;l<32;l++)
      s += lanehist[warp][l][b];
    hist[warp][b + 2] = s;
  }
  __syncwarp();
  if constexpr (VL)
    OrientationPeaksVLFeat<R>(hist[warp] + 2, kp, lane, maxOrientations, kpts, counter, maxKpts);
  else
    OrientationPeaksOpenCV<R>(hist[warp] + 2, kp, lane, smooth[warp], kpts, counter, maxKpts);
}

// VLFeat's descriptor window, row by row (DescriptorsCompat): in row r, the columns [rowLo[r],
// rowLo[r] + rowLen[r]) where the rotated window can be (WindowRange), and the prefix sums
// rowStart of the row lengths (WindowPrefix), to find the row of the t-th sample (WindowRow)
#define WINDOW_MAX_ROWS (2*VL_MAX_W + 1)

// Range of the integers j in [jmin, jmax] where a*j + b can be in (vlo, vhi), widened by one
// on each side (the exact test of each sample follows)
__device__ __forceinline__ void WindowRange(float a, float b, float vlo, float vhi, int &jmin, int &jmax)
{
  if (fabsf(a)<1e-6f)
    return;
  float j1 = (vlo - b)/a, j2 = (vhi - b)/a;
  if (j1>j2) {
    float t = j1;
    j1 = j2;
    j2 = t;
  }
  jmin = max(jmin, (int)floorf(j1) - 1);
  jmax = min(jmax, (int)ceilf(j2) + 1);
}

// Prefix sums of the row lengths (one thread)
__device__ __forceinline__ void WindowPrefix(const int *rowLen, int *rowStart, int numRows)
{
  int s = 0;
  for (int r=0;r<numRows;r++) {
    rowStart[r] = s;
    s += rowLen[r];
  }
  rowStart[numRows] = s;
}

// The row of the t-th sample
__device__ __forceinline__ int WindowRow(const int *rowStart, int numRows, int t)
{
  int lo = 0, hi = numRows - 1;
  while (lo<hi) {
    int mid = (lo + hi + 1)/2;
    if (rowStart[mid]<=t)
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo;
}

// The shared memory of DescriptorHistVLFeat: the terms of the rotated offsets, and the rows of
// the samples near the rotated window
template<typename R>
struct VLFeatWindow {
  R *tabA, *tabB, *tabC, *tabD;
  int *rowLo, *rowLen, *rowStart;
};

// The descriptor histograms of one keypoint, one per warp (hist), from the gradients in a
// rotated window, in an order that doesn't vary between runs. OpenCV (calcSIFTDescriptor):
// Gaussian weights, trilinear interpolation, histograms of (d + 2)^2 x (n + 2) bins with
// borders. VLFeat (vl_sift_calc_keypoint_descriptor): fast_expn weights, its bilinear spatial
// and orientation interpolation, the samples near the rotated window only (WindowRange).
// OpenCV's descriptor orientation (360 - angle, 0 for 360) and its cosine and sine, as
// calcSIFTDescriptor's cosf and sinf (correctly rounded here from double)
__device__ __forceinline__ float2 DescriptorRotationOpenCV(const OpenCVKeypoint &kp)
{
  float ori = 360.f - kp.angle;
  if (fabsf(ori - 360.f)<FLT_EPSILON)
    ori = 0.f;
  return make_float2((float)cos((double)(ori*(float)(M_PI/180))), (float)sin((double)(ori*(float)(M_PI/180))));
}

__device__ void DescriptorHistOpenCV(const CompatPyramid &P, const OpenCVKeypoint &kp, float2 rot, float *hist, float *wbuf)
{
  const int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;
  const int o = kp.o;
  const float *img = P.gauss[o][kp.layer];
  const int rows = P.h[o], cols = P.w[o], p = P.p[o];
  // Position and size in the octave, as OpenCV computes them from the final keypoint
  const float inv = __frcp_rn(P.pix[o]);
  float ptx = kp.x*inv, pty = kp.y*inv;
  int ptxi = __float2int_rn(ptx), ptyi = __float2int_rn(pty);
  float scl = kp.size*inv*0.5f;
  float ori = 360.f - kp.angle;
  if (fabsf(ori - 360.f)<FLT_EPSILON)
    ori = 0.f;
  float cos_t = rot.x, sin_t = rot.y;
  float bins_per_rad = n/360.f;
  float exp_scale = -1.f/(d*d*0.5f);
  float hist_width = SIFT_DESCR_SCL_FCTR*scl;
  int radius = __float2int_rn(hist_width*1.4142135623730951f*(d + 1)*0.5f);
  radius = min(radius, (int)sqrt((double)cols*cols + (double)rows*rows));
  cos_t = __fdiv_rn(cos_t, hist_width);
  sin_t = __fdiv_rn(sin_t, hist_width);
  const int side = 2*radius + 1, total = side*side;
  const int hoffs[8] = {0, 1, n + 2, n + 3, (d + 2)*(n + 2), (d + 2)*(n + 2) + 1, (d + 3)*(n + 2), (d + 3)*(n + 2) + 1};
  for (int base=0;base<total;base+=blockDim.x) {   // the same number of rounds for all warps
    const int k = base + threadIdx.x;
    int i = k/side - radius, j = k%side - radius;
    float c_rot = j*cos_t - i*sin_t;
    float r_rot = j*sin_t + i*cos_t;
    float rbin = r_rot + d/2 - 0.5f;
    float cbin = c_rot + d/2 - 0.5f;
    int r = ptyi + i, c = ptxi + j;
    float contrib[8];
    int h0 = -1;
    if (k<total && rbin>-1 && rbin<d && cbin>-1 && cbin<d && r>0 && r<rows - 1 && c>0 && c<cols - 1) {
      float dx = img[r*p + c + 1] - img[r*p + c - 1];
      float dy = img[(r - 1)*p + c] - img[(r + 1)*p + c];
      float ang = FastAtan2OpenCV(dy, dx);
      float mag = __fsqrt_rn(fmaf(dx, dx, dy*dy))*ExpfOpenCV((c_rot*c_rot + r_rot*r_rot)*exp_scale);
      float obin = (ang - ori)*bins_per_rad;
      int r0 = (int)floorf(rbin);
      int c0 = (int)floorf(cbin);
      int o0 = (int)floorf(obin);
      rbin -= r0;
      cbin -= c0;
      obin -= o0;
      if (o0<0)
        o0 += n;
      if (o0>=n)
        o0 -= n;
      float v_r1 = mag*rbin, v_r0 = mag - v_r1;
      float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
      float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
      float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
      float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
      float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
      float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;
      h0 = ((r0 + 1)*(d + 2) + c0 + 1)*(n + 2) + o0;
      contrib[0] = v_rco000;
      contrib[1] = v_rco001;
      contrib[2] = v_rco010;
      contrib[3] = v_rco011;
      contrib[4] = v_rco100;
      contrib[5] = v_rco101;
      contrib[6] = v_rco110;
      contrib[7] = v_rco111;
    }
    if (__any_sync(0xffffffff, h0>=0))   // most warps of the square's corners have nothing to add
      for (int q=0;q<8;q++)
        WarpHistAdd(hist, wbuf, h0>=0 ? h0 + hoffs[q] : -1, h0>=0 ? contrib[q] : 0.0f);
  }
}

template<typename R>
__device__ void DescriptorHistVLFeat(const CompatPyramid &P, const CompatDetectParams &dp, const VLFeatKeypoint &kp,
                                     R st0, R ct0, float *hist, float *wbuf, VLFeatWindow<R> win)
{
  R *tabA = win.tabA, *tabB = win.tabB, *tabC = win.tabC, *tabD = win.tabD;
  int *rowLo = win.rowLo, *rowLen = win.rowLen, *rowStart = win.rowStart;
  using V = VLFeatArith<R>;
  const int NBO = SIFT_DESCR_HIST_BINS, NBP = SIFT_DESCR_WIDTH;
  const int o = kp.o;
  const int w = P.w[o], h = P.h[o], p = P.p[o];
  const float *gauss = P.gauss[o][LevelIndexVLFeat(P, dp, kp)];
  const R xper = P.pix[o];
  const R x = kp.x/xper, y = kp.y/xper, sigma = kp.sigma/xper;
  const int xi = (int)(x + R(0.5)), yi = (int)(y + R(0.5));
  const R SBP = R(3.0)*sigma + V::eps;
  const int W = min((int)floor(sqrt(2.0)*(double)SBP*(NBP + 1)/2.0 + 0.5), VL_MAX_W);
  // The rotated offsets' terms, as VLFeat rounds them: nx = (A[dx] + B[dy])/SBP, ny = (C[dx] + D[dy])/SBP
  for (int t=threadIdx.x;t<2*W+1;t+=blockDim.x) {
    float dx = (float)((R)(xi + t - W) - x);
    float dy = (float)((R)(yi + t - W) - y);
    tabA[t] = V::mul(ct0, (R)dx);
    tabC[t] = V::mul(-st0, (R)dx);
    tabB[t] = V::mul(st0, (R)dy);
    tabD[t] = V::mul(ct0, (R)dy);
  }
  __syncthreads();
  if (xi<0 || xi>=w || yi<0 || yi>=h - 1)   // the same for the whole block
    return;
  int dy0 = max(-W, 1 - yi), dy1 = min(W, h - yi - 2);
  int dx0 = max(-W, 1 - xi), dx1 = min(W, w - xi - 2);
  const int ny = dy1 - dy0 + 1;   // at most 2*VL_MAX_W + 1 = WINDOW_MAX_ROWS
  const R invSBP = 1/SBP;   // VLFeat divides by SBP: the reciprocal is faster
  // Only the samples near the rotated window (|nx|, |ny| < 2.5): about half of the square
  const float cs = (float)DivRN<R>(ct0, SBP), sn = (float)DivRN<R>(st0, SBP), ox = (float)(xi - x);
  for (int r=threadIdx.x;r<ny;r+=blockDim.x) {
    const float dyf = (float)((R)(yi + dy0 + r) - y);
    int jmin = dx0, jmax = dx1;
    WindowRange(cs, cs*ox + sn*dyf, -2.5f, 2.5f, jmin, jmax);
    WindowRange(-sn, -sn*ox + cs*dyf, -2.5f, 2.5f, jmin, jmax);
    rowLo[r] = jmin;
    rowLen[r] = (jmax>=jmin ? jmax - jmin + 1 : 0);
  }
  __syncthreads();
  if (threadIdx.x==0)
    WindowPrefix(rowLen, rowStart, ny);
  __syncthreads();
  const int total = rowStart[ny];
  for (int base=0;base<total;base+=blockDim.x) {   // the same number of rounds for all warps
    const int t = base + threadIdx.x;
    const bool valid = (t<total);
    const int row = WindowRow(rowStart, ny, min(t, total - 1));
    const int dyi = dy0 + row;
    const int dxi = rowLo[row] + min(t, total - 1) - rowStart[row];
    float fnx = (float)(V::add(tabA[dxi + W], tabB[dyi + W])*invSBP);
    float fny = (float)(V::add(tabC[dxi + W], tabD[dyi + W])*invSBP);
    // Samples whose 8 bins are all outside the descriptor add nothing: skip their gradient
    // and their weights, and whole warps of them
    const bool inside = valid && fnx>=-2.5f && fnx<2.5f && fny>=-2.5f && fny<2.5f;
    if (!__any_sync(0xffffffff, inside))
      continue;
    float2 g = (inside ? GradientVLFeat(gauss, xi + dxi, yi + dyi, w, h, p) : make_float2(0.0f, 0.0f));
    float mod = g.x, angle = g.y;
    float theta = Mod2PiVLFeat((float)((R)angle - kp.angle));
    float nt = (float)((R)__fmul_rn((float)NBO, theta)*(1/(2*V::pi)));
    const float wsigma = (float)(NBP/2);
    float win = (float)FastExpnVLFeat<R>((R)__fadd_rn(__fmul_rn(fnx, fnx), __fmul_rn(fny, fny))*(1/(R(2.0)*wsigma*wsigma)));
    // VLFeat subtracts in double here, exactly: the float operations round the same way
    int binx = (int)floorf(__fsub_rn(fnx, 0.5f));
    int biny = (int)floorf(__fsub_rn(fny, 0.5f));
    int bint = (int)floorf(nt);
    float rbinx = __fsub_rn(fnx, (float)binx + 0.5f);
    float rbiny = __fsub_rn(fny, (float)biny + 0.5f);
    float rbint = __fsub_rn(nt, (float)bint);
    for (int dbinx=0;dbinx<2;dbinx++)
      for (int dbiny=0;dbiny<2;dbiny++)
        for (int dbint=0;dbint<2;dbint++) {
          int bin = -1;
          float weight = 0.0f;
          if (valid && binx + dbinx>=-(NBP/2) && binx + dbinx<(NBP/2) && biny + dbiny>=-(NBP/2) && biny + dbiny<(NBP/2)) {
            weight = __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(win, mod),
                                                   fabsf(__fsub_rn((float)(1 - dbinx), rbinx))),
                                         fabsf(__fsub_rn((float)(1 - dbiny), rbiny))),
                               fabsf(__fsub_rn((float)(1 - dbint), rbint)));
            bin = (biny + dbiny + NBP/2)*NBO*NBP + (binx + dbinx + NBP/2)*NBO + (bint + dbint)%NBO;
          }
          WarpHistAdd(hist, wbuf, bin, weight);
        }
  }
}

// The descriptor raw (128 values) from the warps' histograms: OpenCV's (wrap the orientation
// bins, clip at 0.2 of the norm, renormalize, quantize to 8 bits) and VLFeat's
// (normalize_histogram) or, with rootSift, COLMAP's (L1-root, 8 bits)
template<int NW>
__device__ void DescriptorOpenCV(const float (*whist)[(SIFT_DESCR_WIDTH + 2)*(SIFT_DESCR_WIDTH + 2)*(SIFT_DESCR_HIST_BINS + 2)],
                                 float *raw, float *norm)
{
  const int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;
  // Sum the warps' histograms, wrap the orientation bins, clip at 0.2 of the norm,
  // renormalize and quantize as OpenCV
  for (int k=threadIdx.x;k<d*d*n;k+=blockDim.x) {
    int i = k/(d*n), j = (k/n)%d, b = k%n;
    int h0 = ((i + 1)*(d + 2) + (j + 1))*(n + 2) + b;
    float v = 0.0f, v2 = 0.0f;
    for (int wp=0;wp<NW;wp++) {
      v += whist[wp][h0];
      if (b<2)
        v2 += whist[wp][h0 + n];
    }
    raw[k] = (b<2 ? v + v2 : v);
  }
  __syncthreads();
  if (threadIdx.x==0) {
    float nrm2 = 0;
    for (int k=0;k<d*d*n;k++)
      nrm2 = fmaf(raw[k], raw[k], nrm2);
    *norm = __fsqrt_rn(nrm2)*SIFT_DESCR_MAG_THR;
  }
  __syncthreads();
  if (threadIdx.x<d*d*n)
    raw[threadIdx.x] = fminf(raw[threadIdx.x], *norm);
  __syncthreads();
  if (threadIdx.x==0) {
    float nrm2 = 0;
    for (int k=0;k<d*d*n;k++)
      nrm2 += raw[k]*raw[k];
    *norm = __fdiv_rn(SIFT_INT_DESCR_FCTR, fmaxf(__fsqrt_rn(nrm2), FLT_EPSILON));
  }
  __syncthreads();
  // OpenCV's 8-bit values, scaled to unit length for MatchSiftData
  for (int k=threadIdx.x;k<d*d*n;k+=blockDim.x)
    raw[k] = fminf(fmaxf(rintf(raw[k]*(*norm)), 0.0f), 255.0f);
}

template<int NW>
__device__ void DescriptorVLFeat(const float (*whist)[SIFT_DESCR_HIST_BINS*SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH], float *raw,
                                 float *norm, bool rootSift)
{
  const int NBO = SIFT_DESCR_HIST_BINS, NBP = SIFT_DESCR_WIDTH;
  for (int t=threadIdx.x;t<NBO*NBP*NBP;t+=blockDim.x) {
    float v = 0.0f;
    for (int wp=0;wp<NW;wp++)
      v += whist[wp][t];
    raw[t] = v;
  }
  __syncthreads();
  // Normalize, truncate at 0.2 and normalize again (normalize_histogram): the sums in
  // VLFeat's order, the rest in parallel
  const int t = threadIdx.x;
  for (int pass=0;pass<2;pass++) {
    if (t==0) {
      float n = 0.0f;
      for (int k=0;k<NBO*NBP*NBP;k++)
        n = __fadd_rn(n, __fmul_rn(raw[k], raw[k]));
      *norm = __fadd_rn(FastSqrtVLFeat(n), VL_EPSILON_F);
    }
    __syncthreads();
    if (t<NBO*NBP*NBP) {
      float v = __fdiv_rn(raw[t], *norm);
      raw[t] = (pass==0 && v>0.2 ? 0.2f : v);
    }
    __syncthreads();
  }
  if (rootSift) {
    // COLMAP: L1RootNormalizeFeatureDescriptors, TransformVLFeatToUBCFeatureDescriptors,
    // FeatureDescriptorsToUnsignedByte
    if (t==0) {
      float l1 = 0.0f;
      for (int k=0;k<NBO*NBP*NBP;k++)
        l1 += fabsf(raw[k]);
      *norm = (l1>0 ? __fdiv_rn(1.0f, l1) : 0.0f);   // no descriptor (keypoint at the border): zero, not COLMAP's NaN
    }
    __syncthreads();
    // COLMAP's 8-bit values, in OpenCV's bin order, scaled to unit length for MatchSiftData
    if (t<NBO*NBP*NBP) {
      float v = __fsqrt_rn(__fmul_rn(raw[t], *norm));
      raw[t] = fminf(255.0f, roundf(512.0f*v));
    }
    __syncthreads();
  }
}

// 1 / the norm of the 128 8-bit values of raw (summed in order by one thread), to scale them to
// unit length for MatchSiftData
__device__ float InvNorm(const float *raw, float *norm)
{
  if (threadIdx.x==0) {
    float n2 = 0.0f;
    for (int k=0;k<128;k++)
      n2 += raw[k]*raw[k];
    *norm = (n2>0 ? __frsqrt_rn(n2) : 0.0f);
  }
  __syncthreads();
  return *norm;
}

// The fields of a SiftPoint, its descriptor apart
__device__ void WritePoint(SiftPoint &pt, float x, float y, float scale, float sharpness, float edgeness,
                           float orientation, float subsampling)
{
  pt.xpos = x;
  pt.ypos = y;
  pt.scale = scale;
  pt.sharpness = sharpness;
  pt.edgeness = edgeness;
  pt.orientation = orientation;
  pt.score = 0.0f;
  pt.ambiguity = 0.0f;
  pt.match = -1;
  pt.match_xpos = 0.0f;
  pt.match_ypos = 0.0f;
  pt.match_error = 0.0f;
  pt.subsampling = subsampling;
  pt.empty[0] = pt.empty[1] = pt.empty[2] = 0.0f;
}

// Descriptors, one block per keypoint: 4x4 cells of 8 orientation bins, of unit length.
// outScale: from OpenCV's keypoint coordinates (those of the first octave) to the input's.
template<SiftCompatMode M, typename R>
__global__ void DescriptorsCompat(CompatPyramid P, CompatDetectParams dp, float outScale, bool rootSift,
                                  const typename CompatModeTraits<M>::Keypoint *kpts, int numKpts, SiftPoint *out)
{
  constexpr bool VL = (M==SIFT_COMPAT_VLFEAT);
  constexpr int NW = CompatModeTraits<M>::descThreads/32;
  constexpr int HS = (VL ? SIFT_DESCR_HIST_BINS*SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH :
                      (SIFT_DESCR_WIDTH + 2)*(SIFT_DESCR_WIDTH + 2)*(SIFT_DESCR_HIST_BINS + 2));
  constexpr int TW = (VL ? 2*VL_MAX_W + 1 : 1), WR = (VL ? WINDOW_MAX_ROWS : 0);   // VLFeat's window
  __shared__ float whist[NW][HS];
  __shared__ float wbuf[NW][32];
  __shared__ float raw[128];
  __shared__ float norm;
  __shared__ R tabA[TW], tabB[TW], tabC[TW], tabD[TW];
  __shared__ int rowLo[WR + 1], rowLen[WR + 1], rowStart[WR + 1];
  __shared__ R rot[2];   // the cosine and sine of the descriptor's orientation
  const int idx = blockIdx.x;
  if (idx>=numKpts)
    return;
  const typename CompatModeTraits<M>::Keypoint kp = kpts[idx];
  for (int k=threadIdx.x;k<NW*HS;k+=blockDim.x)
    (&whist[0][0])[k] = 0.0f;
  if (threadIdx.x==0) {
    if constexpr (VL) {
      rot[0] = (R)cos((double)kp.angle);   // in double, as VLFeat
      rot[1] = (R)sin((double)kp.angle);
    } else {
      const float2 cs = DescriptorRotationOpenCV(kp);
      rot[0] = cs.x;
      rot[1] = cs.y;
    }
  }
  __syncthreads();
  float *hist = whist[threadIdx.x/32];
  SiftPoint &pt = out[idx];
  const int t = threadIdx.x;
  if constexpr (VL) {
    DescriptorHistVLFeat<R>(P, dp, kp, rot[1], rot[0], hist, wbuf[threadIdx.x/32], {tabA, tabB, tabC, tabD, rowLo, rowLen, rowStart});
    __syncthreads();
    DescriptorVLFeat<NW>(whist, raw, &norm, rootSift);
    __syncthreads();
    if (rootSift) {
      const float inv = InvNorm(raw, &norm);
      if (t<128)
        pt.data[t - t%8 + (8 - t%8)%8] = raw[t]*inv;   // OpenCV's bin order
    } else if (t<128)
      pt.data[t] = raw[t];
    if (t==0) {
      float deg = (float)(kp.angle*R(180.0)/VLFeatArith<R>::pi);   // VLFeat's angles are in [0, 2 pi]
      WritePoint(pt, (float)kp.x, (float)kp.y, (float)kp.sigma, kp.val, kp.score, deg>=360.0f ? deg - 360.0f : deg, P.pix[kp.o]);
    }
  } else {
    DescriptorHistOpenCV(P, kp, make_float2((float)rot[0], (float)rot[1]), hist, wbuf[threadIdx.x/32]);
    __syncthreads();
    DescriptorOpenCV<NW>(whist, raw, &norm);
    __syncthreads();
    const float inv = InvNorm(raw, &norm);
    for (int k=t;k<128;k+=blockDim.x)
      pt.data[k] = raw[k]*inv;
    if (t==0)
      WritePoint(pt, kp.x*outScale, kp.y*outScale, kp.size*outScale*0.5f, kp.contr, kp.edge, kp.angle,
                 P.pix[kp.o]*outScale);
  }
}

// ---------------------------------------------------------------------------------------
// Host code: temporary memory, the backend of the pyramids and the pipeline (the rest, which
// doesn't depend on CUDA, is in cudaSiftCompatHost.h)

#include "cudaSiftCompatHost.h"

// Temporary device memory, kept between calls (FreeSiftCompatMemory releases it): the scale
// space, and the candidates and keypoints, each list with its counter
struct CompatBuffer {
  char *ptr = NULL;
  size_t size = 0;
  void *Get(size_t bytes)
  {
    if (bytes>size) {
      if (ptr)
        safeCall(cudaFree(ptr));
      safeCall(cudaMalloc((void **)&ptr, bytes));
      size = bytes;
    }
    return ptr;
  }
  void Free()
  {
    if (ptr)
      safeCall(cudaFree(ptr));
    ptr = NULL;
    size = 0;
  }
};
// The state of ExtractSiftCompat on one device: its temporary memory, kept between calls, and
// a lock, which serializes the calls on the device (the blurs' kernels are in its constant
// memory, d_CompatKernel)
struct CompatDeviceState {
  std::mutex lock;
  CompatBuffer scaleSpace, cands, kpts;
  int maxCands = 1 << 16, maxKpts = 1 << 16;   // grown when full
  bool tablesReady = false;                    // d_ExpnTab and d_ExpnTabF set
};

// The state of the current device
static CompatDeviceState &CompatState()
{
  static std::mutex statesLock;
  static std::map<int, std::unique_ptr<CompatDeviceState>> states;
  int device = 0;
  safeCall(cudaGetDevice(&device));
  std::lock_guard<std::mutex> guard(statesLock);
  std::unique_ptr<CompatDeviceState> &st = states[device];
  if (!st)
    st.reset(new CompatDeviceState);
  return *st;
}

void FreeSiftCompatMemory()
{
  CompatDeviceState &st = CompatState();
  std::lock_guard<std::mutex> guard(st.lock);
  st.scaleSpace.Free();
  st.cands.Free();
  st.kpts.Free();
  st.tablesReady = false;
}

// Runs launch(items, counter, maxItems), the kernels that append items (candidates or
// keypoints) to a list, growing the list and running them again until all the items fit;
// returns their number. The counter comes first in the buffer, then the list.
#define COMPAT_COUNTER_BYTES 256   // keeps the list aligned
template <typename T, typename F>
static unsigned int CollectCompat(CompatBuffer &buf, int &maxItems, T *&items, unsigned int *&counter, F launch)
{
  for (;;) {
    char *mem = (char *)buf.Get(sizeof(T)*(size_t)maxItems + COMPAT_COUNTER_BYTES);
    counter = (unsigned int *)mem;
    items = (T *)(mem + COMPAT_COUNTER_BYTES);
    safeCall(cudaMemsetAsync(counter, 0, sizeof(unsigned int)));
    launch(items, counter, maxItems);
    unsigned int n = 0;
    safeCall(cudaMemcpy(&n, counter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    if (n<=(unsigned int)maxItems)
      return n;
    maxItems = (int)(n + n/4);
  }
}

// Places the levels of the octaves of P (its sizes, nLev and lev set) in the scale-space memory,
// after `extra` floats of temporary images, which it returns
static float *AllocatePyramid(CompatPyramid &P, size_t extra, CompatBuffer &scaleSpace)
{
  size_t total = extra;
  for (int o=0;o<P.nOct;o++)
    total += (size_t)(P.nLev + P.lev[o])*P.p[o]*P.h[o];
  float *mem = (float *)scaleSpace.Get(total*sizeof(float));
  float *next = mem + extra;
  for (int o=0;o<P.nOct;o++) {
    size_t sz = (size_t)P.p[o]*P.h[o];
    for (int i=0;i<P.nLev + P.lev[o];i++, next+=sz)
      P.gauss[o][i] = next;
  }
  return mem;
}

// The CUDA backend of the pyramid builders of cudaSiftCompatHost.h: the levels are pointers in
// the scale-space memory, the blurs' kernels go to constant memory
struct CompatPyramidBackend {
  typedef float *Level;
  CompatBuffer &scaleSpace;
  Level Allocate(CompatPyramid &P, size_t extra) { return AllocatePyramid(P, extra, scaleSpace); }
  static Level At(const CompatPyramid &P, int o, int i) { return P.gauss[o][i]; }
  // The first octave's input image, upsampled or not
  void Input(int mode, Level dst, int w, int h, int p, CudaImage &img, bool upsample)
  {
    dim3 threads(32, 8), blocks(iDivUp(w, 32), iDivUp(h, 8));
    if (mode==SIFT_COMPAT_VLFEAT)
      InputCompat<SIFT_COMPAT_VLFEAT><<<blocks, threads>>>(dst, w, h, p, img.d_data, img.width, img.height, img.pitch, upsample);
    else if (upsample)
      InputCompat<SIFT_COMPAT_OPENCV><<<blocks, threads>>>(dst, w, h, p, img.d_data, img.width, img.height, img.pitch, 1);
    else
      safeCall(cudaMemcpy2DAsync(dst, p*sizeof(float), img.d_data, img.pitch*sizeof(float), w*sizeof(float), h, cudaMemcpyDeviceToDevice));
    checkMsg("InputCompat() execution failed\n");
  }
  void Blur(int mode, Level dst, Level src, Level tmp, int w, int h, int p, const float *kernel, int radius)
  {
    safeCall(cudaMemcpyToSymbolAsync(d_CompatKernel, kernel, (2*radius + 1)*sizeof(float)));
    if (mode==SIFT_COMPAT_VLFEAT)
      BlurLevel<SIFT_COMPAT_VLFEAT>(dst, src, tmp, w, h, p, radius);
    else
      BlurLevel<SIFT_COMPAT_OPENCV>(dst, src, tmp, w, h, p, radius);
    checkMsg("BlurLevel() execution failed\n");
  }
  // Every other pixel of src (pitch sp)
  void Downsample(Level dst, int w, int h, int p, Level src, int sp)
  {
    DownsampleCompat<<<dim3(iDivUp(w, 32), iDivUp(h, 8)), dim3(32, 8)>>>(dst, w, h, p, src, sp);
    checkMsg("DownsampleCompat() execution failed\n");
  }
};

// The keypoints kept, on the host (cudaSiftCompatHost.h)
static void SelectKeypoints(std::vector<OpenCVKeypoint> &kpts, const CompatPyramid &, const CompatDetectParams &,
                            const SiftCompatParams &prm, const OpenCVKeypoint *, unsigned int, int maxPts)
{
  SelectKeypointsOpenCV(kpts, prm, maxPts);
}

static void SelectKeypoints(std::vector<VLFeatKeypoint> &kpts, const CompatPyramid &P, const CompatDetectParams &dp,
                            const SiftCompatParams &prm, const VLFeatKeypoint *d_cands, unsigned int numCands, int maxPts)
{
  std::vector<VLFeatKeypoint> cands;
  if (prm.maxFeatures>0 && numCands>0) {   // COLMAP's selection counts the candidates per level
    cands.resize(numCands);
    safeCall(cudaMemcpy(cands.data(), d_cands, sizeof(VLFeatKeypoint)*numCands, cudaMemcpyDeviceToHost));
  }
  auto level = [&](const VLFeatKeypoint &k) { return k.o*COMPAT_MAX_LEVELS + LevelIndexVLFeat(P, dp, k); };
  SelectKeypointsVLFeat(kpts, prm, level, cands.data(), (unsigned int)cands.size(), maxPts);
}

// The pipeline, the same for both modes: the pyramid, the extrema, the orientations, the
// selection on the host, the descriptors
template<SiftCompatMode M>
static void ExtractSiftCompatT(SiftData &siftData, CudaImage &img, const SiftCompatParams &prm, CompatDeviceState &st)
{
  typedef typename CompatModeTraits<M>::Keypoint Keypoint;
  constexpr int B = CompatModeTraits<M>::border;
#ifdef VERBOSE
  TimerGPU timer(0);
#endif
  CompatPyramid P;
  CompatDetectParams dp = {};
  CompatPyramidBackend backend = {st.scaleSpace};
  if constexpr (M==SIFT_COMPAT_VLFEAT)
    BuildPyramidVLFeat(backend, P, dp, img, prm);
  else
    BuildPyramidOpenCV(backend, P, dp, img, prm);
  const bool dbl = prm.doublePrecision;
  constexpr int NWO = CompatModeTraits<M>::oriWarps;
  dim3 threads(32, 8);
  Keypoint *d_cands = NULL, *d_kpts = NULL;
  unsigned int *d_candCounter = NULL, *d_kptCounter = NULL;
  unsigned int numCands = CollectCompat(st.cands, st.maxCands, d_cands, d_candCounter,
    [&](Keypoint *cands, unsigned int *counter, int maxCands) {
      for (int o=0;o<P.nOct;o++) {
        int iw = P.w[o] - 2*B, ih = P.h[o] - 2*B;
        if (iw<=0 || ih<=0)
          continue;
        for (int layer=1;layer<=dp.S + P.lev[o];layer++)
          (dbl ? FindExtremaCompat<M, double> : FindExtremaCompat<M, float>)<<<dim3(iDivUp(iw, 32), iDivUp(ih, 8)), threads>>>(P, o, layer, dp, cands, counter, maxCands);
      }
      checkMsg("FindExtremaCompat() execution failed\n");
    });
  unsigned int numKpts = CollectCompat(st.kpts, st.maxKpts, d_kpts, d_kptCounter,
    [&](Keypoint *kpts, unsigned int *counter, int maxKpts) {
      if (numCands>0)
        (dbl ? OrientationsCompat<M, double> : OrientationsCompat<M, float>)<<<iDivUp(numCands, NWO), 32*NWO>>>(
          P, dp, prm.maxOrientations, d_cands, numCands, kpts, counter, maxKpts);
      checkMsg("OrientationsCompat() execution failed\n");
    });
  std::vector<Keypoint> kpts(numKpts);
  if (numKpts>0)
    safeCall(cudaMemcpy(kpts.data(), d_kpts, sizeof(Keypoint)*numKpts, cudaMemcpyDeviceToHost));
  SelectKeypoints(kpts, P, dp, prm, d_cands, numCands, siftData.maxPts);
  int numPts = (int)kpts.size();
#ifdef MANAGEDMEM
  SiftPoint *d_out = siftData.m_data;
#else
  SiftPoint *d_out = siftData.d_data;
#endif
  if (numPts>0) {
    safeCall(cudaMemcpy(d_kpts, kpts.data(), sizeof(Keypoint)*numPts, cudaMemcpyHostToDevice));
    constexpr int NTD = CompatModeTraits<M>::descThreads;
    const float outScale = (prm.firstOctave<0 ? 1.0f/(1 << -prm.firstOctave) : (float)(1 << prm.firstOctave));
    (dbl ? DescriptorsCompat<M, double> : DescriptorsCompat<M, float>)<<<numPts, NTD>>>(P, dp, outScale, prm.rootSift, d_kpts, numPts, d_out);
    checkMsg("DescriptorsCompat() execution failed\n");
  }
  siftData.numPts = numPts;
#ifdef MANAGEDMEM
  safeCall(cudaDeviceSynchronize());
#else
  if (siftData.h_data && numPts>0)
    safeCall(cudaMemcpy(siftData.h_data, siftData.d_data, sizeof(SiftPoint)*numPts, cudaMemcpyDeviceToHost));
  safeCall(cudaStreamSynchronize(0));   // the kernels are done with the temporary memory before the lock is released
#endif
#ifdef VERBOSE
  printf("ExtractSiftCompat time =      %.2f ms, %d keypoints (%u candidates)\n", timer.read(), numPts, numCands);
#endif
}

// VLFeat's fast_expn tables (d_ExpnTab in double, d_ExpnTabF in float), on the current device
static void InitExpnTables()
{
  double tab[VL_EXPN_SZ + 1];
  float tabf[VL_EXPN_SZ + 1];
  for (int k=0;k<VL_EXPN_SZ+1;k++) {
    tab[k] = std::exp(-(double)k*(VL_EXPN_MAX/VL_EXPN_SZ));
    tabf[k] = (float)tab[k];
  }
  safeCall(cudaMemcpyToSymbol(d_ExpnTab, tab, sizeof(tab)));
  safeCall(cudaMemcpyToSymbol(d_ExpnTabF, tabf, sizeof(tabf)));
}

void ExtractSiftCompat(SiftData &siftData, CudaImage &img, const SiftCompatParams &prm)
{
  ValidateCompatParams(prm);
  CompatDeviceState &st = CompatState();
  std::lock_guard<std::mutex> guard(st.lock);
  if (!st.tablesReady) {
    InitExpnTables();
    st.tablesReady = true;
  }
  if (prm.mode==SIFT_COMPAT_VLFEAT)
    ExtractSiftCompatT<SIFT_COMPAT_VLFEAT>(siftData, img, prm, st);
  else
    ExtractSiftCompatT<SIFT_COMPAT_OPENCV>(siftData, img, prm, st);
}
