//********************************************************//
// OpenCV- and VLFeat-compatible SIFT extraction          //
//              Apple Metal backend                       //
//********************************************************//

// Metal versions of the kernels of cudaSiftCompat.cu, line by line where possible, in float
// only (Apple GPUs have no double precision). cudaSiftCompatShared.h is prepended to this file,
// which is compiled at run time without fast math, in its own library: there, divisions are
// correctly rounded, and products and sums written as separate operations (MulRN, AddRN, SubRN)
// are rounded separately, as with CUDA's _rn intrinsics; precise::sqrt replaces __fsqrt_rn.
// Both modes run the same kernels, templated on the mode (instantiated under [[host_name]]); each
// library's version of a step is an overload on its keypoint type.
//
// CUDA -> Metal: __syncthreads -> threadgroup_barrier, __syncwarp -> simdgroup_barrier, __any_sync
// -> simd_any, __match_any_sync and the per-warp histograms -> smaller histograms shared by groups
// of lanes (GroupHistAdd), atomicAdd -> atomic_fetch_add_explicit, __constant__ memory -> a constant buffer, pointers to
// the levels -> offsets in the scale-space buffer.

#include <metal_stdlib>
using namespace metal;

struct SiftPoint {
  float xpos;
  float ypos;
  float scale;
  float sharpness;
  float edgeness;
  float orientation;
  float score;
  float ambiguity;
  int match;
  float match_xpos;
  float match_ypos;
  float match_error;
  float subsampling;
  float empty[3];
  float data[128];
};
static_assert(sizeof(SiftPoint)==576, "SiftPoint as in cudaSift.h");

constant float VL_PI_F = 3.14159265f;

// Products and sums rounded separately, never fused into multiply-adds by the compiler (CUDA's
// __fmul_rn, __fadd_rn, __fsub_rn)
inline float MulRN(float a, float b)
{
#pragma METAL fp contract(off)
  return a*b;
}
inline float AddRN(float a, float b)
{
#pragma METAL fp contract(off)
  return a + b;
}
inline float SubRN(float a, float b)
{
#pragma METAL fp contract(off)
  return a - b;
}

// (float)((double)a + 2*pi), the sum of a float and a double that VLFeat rounds to float: a
// two-sum with 2*pi split in two floats
inline float AddTwoPiVLFeat(float a)
{
  const float hi = 6.28318548f, lo = -1.74845553e-07f;   // hi + lo = 2*pi
  float s = AddRN(a, hi);
  float bb = SubRN(s, a);
  float e = AddRN(SubRN(a, SubRN(s, bb)), SubRN(hi, bb));
  return AddRN(s, AddRN(e, lo));
}

// The pyramid's levels: offsets in the scale-space buffer
struct Levels {
  device float *ss;
  constant CompatPyramidM *P;
  device const float *Get(int o, int l) const { return ss + P->off[o][l]; }
};

// A DoG level, computed where it is read from the two Gaussian levels instead of stored
struct DogView {
  device const float *lower, *upper;
  float operator[](int i) const { return upper[i] - lower[i]; }
  DogView operator+(int i) const { return {lower + i, upper + i}; }
};

inline DogView Dog(Levels L, int o, int l)
{
  return {L.Get(o, l), L.Get(o, l + 1)};
}

// OpenCV's borderInterpolate for BORDER_REFLECT_101
inline int Reflect101(int i, int n)
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

// The modes: 0 OpenCV, 1 VLFeat (SiftCompatMode)
#define MODE_OPENCV 0
#define MODE_VLFEAT 1

// The index of a sample of the borders: OpenCV's BORDER_REFLECT_101, VLFeat's continuity
template<int M>
inline int BorderIndex(int i, int n)
{
  if (i>=0 && i<n)
    return i;
  return (M==MODE_VLFEAT ? min(max(i, 0), n - 1) : Reflect101(i, n));
}

// Adds v to hist[bin] for every lane with bin >= 0, the DESC_GROUP lanes of a group sharing hist
// adding one after the other, in lane order: an order that doesn't vary between runs. (The CUDA
// code sums the lanes of a warp with the same bin, with __match_any_sync; emulated with SIMD
// group votes, that is slower than these serial adds into smaller shared histograms.)
inline void GroupHistAdd(threadgroup float *hist, int bin, float v, uint lane)
{
  for (uint j=0;j<DESC_GROUP;j++) {
    if (lane%DESC_GROUP==j && bin>=0)
      hist[bin] += v;
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ---------------------------------------------------------------------------------------
// Separable Gaussian blurs, on tiles in threadgroup memory, as BlurRowsTiled and
// BlurColumnsTiled in cudaSiftCompat.cu, templated on the radius RAD (1 to TILED_MAX_RADIUS, or
// 0 for any radius): on Metal, these two passes with the loops unrolled are faster than the
// CUDA code's fused blur (BlurFused), whose larger tiles leave fewer threadgroups per GPU core.
// OpenCV: rows then columns, the row filter sums the taps in order, the column filter pairs the
// symmetric taps, with fused multiply-adds. VLFeat: columns then rows, borders by continuity,
// products and sums rounded separately.

template<int M, int RAD>
kernel void BlurRowsTiled(device float *ss [[buffer(0)]], constant CompatImageArgs &a [[buffer(1)]],
                          constant float *k [[buffer(2)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]])
{
  const bool VL = (M==MODE_VLFEAT);
  threadgroup float tile[ROW_H*(ROW_W + 2*(RAD ? RAD : COMPAT_MAX_RADIUS))];
  device float *dst = ss + a.dst;
  device const float *src = ss + a.src;
  const int radius = (RAD ? RAD : a.radius), w = a.w, h = a.h, p = a.p;
  const int tw = ROW_W + 2*radius;
  const int x0 = tg.x*ROW_W, y0 = tg.y*ROW_H;
  device const float *srow = src + min(y0 + (int)tid.y, h - 1)*p;
  for (int tx=tid.x;tx<tw;tx+=ROW_W)
    tile[tid.y*tw + tx] = srow[BorderIndex<M>(x0 + tx - radius, w)];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int x = x0 + tid.x, y = y0 + tid.y;
  if (x>=w || y>=h)
    return;
  threadgroup const float *t = tile + tid.y*tw + tid.x;
  float s;
  if (VL) {
    s = 0.0f;
    for (int j=0;j<=2*radius;j++)
      s = AddRN(s, MulRN(t[j], k[2*radius - j]));
  } else {
    s = t[0]*k[0];
    for (int j=1;j<=2*radius;j++)
      s = fma(t[j], k[j], s);
  }
  dst[y*p + x] = s;
}

template<int M, int RAD>
kernel void BlurColumnsTiled(device float *ss [[buffer(0)]], constant CompatImageArgs &a [[buffer(1)]],
                             constant float *k [[buffer(2)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]])
{
  const bool VL = (M==MODE_VLFEAT);
  threadgroup float tile[COL_W*(COL_H + 2*(RAD ? RAD : COMPAT_MAX_RADIUS))];
  device float *dst = ss + a.dst;
  device const float *src = ss + a.src;
  const int radius = (RAD ? RAD : a.radius), w = a.w, h = a.h, p = a.p;
  const int th = COL_H + 2*radius;
  const int x0 = tg.x*COL_W, y0 = tg.y*COL_H;
  const int xs = min(x0 + (int)tid.x, w - 1);
  for (int ty=tid.y;ty<th;ty+=COL_THREADS_Y)
    tile[ty*COL_W + tid.x] = src[BorderIndex<M>(y0 + ty - radius, h)*p + xs];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int x = x0 + tid.x;
  if (x>=w)
    return;
  for (int yy=tid.y;yy<COL_H;yy+=COL_THREADS_Y) {
    const int y = y0 + yy;
    if (y>=h)
      break;
    threadgroup const float *t = tile + (yy + radius)*COL_W + tid.x;
    float s;
    if (VL) {
      s = 0.0f;
      for (int j=-radius;j<=radius;j++)
        s = AddRN(s, MulRN(t[j*COL_W], k[radius - j]));
    } else {
      s = t[0]*k[radius];
      for (int j=1;j<=radius;j++)
        s = fma(t[j*COL_W] + t[-j*COL_W], k[radius + j], s);
    }
    dst[y*p + x] = s;
  }
}

#define TILED_INST(M, RAD) \
  template [[host_name("BlurRowsTiled_" #M "_" #RAD)]] kernel void BlurRowsTiled<M, RAD>(device float *, constant CompatImageArgs &, constant float *, uint2, uint2); \
  template [[host_name("BlurColumnsTiled_" #M "_" #RAD)]] kernel void BlurColumnsTiled<M, RAD>(device float *, constant CompatImageArgs &, constant float *, uint2, uint2);
#define TILED_RADII(F, M) F(M, 0) F(M, 1) F(M, 2) F(M, 3) F(M, 4) F(M, 5) F(M, 6) F(M, 7) F(M, 8) F(M, 9) F(M, 10) F(M, 11) F(M, 12) F(M, 13) F(M, 14) F(M, 15) F(M, 16)
TILED_RADII(TILED_INST, 0)
TILED_RADII(TILED_INST, 1)

// ---------------------------------------------------------------------------------------
// OpenCV's arithmetic: cv::fastAtan2, expf, the 3x3 solve and the refinement of the extrema

// cv::fastAtan2 (degrees in [0, 360))
inline float FastAtan2OpenCV(float y, float x)
{
  const float deg = 57.295780181884766f;   // (float)(180/CV_PI)
  const float p1 = 0.9997878412794807f*deg;
  const float p3 = -0.3258083974640975f*deg;
  const float p5 = 0.1555786518463281f*deg;
  const float p7 = -0.04432655554792128f*deg;
  const float eps = 2.220446049250313e-16f;   // (float)DBL_EPSILON
  float ax = fabs(x), ay = fabs(y);
  float r, c, c2;
  if (ax>=ay) {
    c = ay/(ax + eps);
    c2 = c*c;
    r = (((p7*c2 + p5)*c2 + p3)*c2 + p1)*c;
  } else {
    c = ax/(ay + eps);
    c2 = c*c;
    r = 90.f - (((p7*c2 + p5)*c2 + p3)*c2 + p1)*c;
  }
  if (x<0)
    r = 180.f - r;
  if (y<0)
    r = 360.f - r;
  return r;
}

// expf within about 1 ulp (Cephes' expf)
inline float ExpfOpenCV(float x)
{
  float t = rint(x*1.44269504088896341f);
  float f = fma(t, -0.693359375f, x);
  f = fma(t, 2.12194440e-4f, f);
  float p = 1.9875691500E-4f;
  p = fma(p, f, 1.3981999507E-3f);
  p = fma(p, f, 8.3334519073E-3f);
  p = fma(p, f, 4.1665795894E-2f);
  p = fma(p, f, 1.6666665459E-1f);
  p = fma(p, f, 5.0000001201E-1f);
  p = fma(p, f*f, f) + 1.0f;
  int i = (int)t;
  return (i<-126 ? 0.0f : p*as_type<float>((i + 127) << 23));
}

// 3x3 solve as cv::Matx<float, 3, 3>::solve(DECOMP_LU) (Cramer's rule; zero if singular)
inline void Solve3OpenCV(thread const float a[3][3], thread const float b[3], thread float x[3])
{
  float d = a[0][0]*(a[1][1]*a[2][2] - a[2][1]*a[1][2]) -
    a[0][1]*(a[1][0]*a[2][2] - a[2][0]*a[1][2]) +
    a[0][2]*(a[1][0]*a[2][1] - a[2][0]*a[1][1]);
  if (d==0) {
    x[0] = x[1] = x[2] = 0.0f;
    return;
  }
  d = 1.0f/d;
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
// layer is the level index in the pyramid (OpenCV's layer + lev[o]).
inline bool Refine(Levels Lv, int o, int layer, int r, int c, constant CompatDetectParamsM &dp, thread OpenCVKeypoint &kp)
{
  constant CompatPyramidM &P = *Lv.P;
  const float img_scale = 1.f/255;
  const float deriv_scale = img_scale*0.5f;
  const float second_deriv_scale = img_scale;
  const float cross_deriv_scale = img_scale*0.25f;
  const int w = P.w[o], h = P.h[o], p = P.p[o];
  const int L = dp.S;
  float xi = 0, xr = 0, xc = 0, contr = 0;
  int i = 0;
  for (;i<SIFT_MAX_INTERP_STEPS;i++) {
    const DogView prev = Dog(Lv, o, layer-1), cur = Dog(Lv, o, layer), next = Dog(Lv, o, layer+1);
    const DogView ir = cur + r*p;
    float dD[3] = {(ir[c+1] - ir[c-1])*deriv_scale, (ir[c+p] - ir[c-p])*deriv_scale,
                   (next[r*p+c] - prev[r*p+c])*deriv_scale};
    float v2 = ir[c]*2;
    float dxx = (ir[c+1] + ir[c-1] - v2)*second_deriv_scale;
    float dyy = (ir[c+p] + ir[c-p] - v2)*second_deriv_scale;
    float dss = (next[r*p+c] + prev[r*p+c] - v2)*second_deriv_scale;
    float dxy = (ir[c+p+1] - ir[c+p-1] - ir[c-p+1] + ir[c-p-1])*cross_deriv_scale;
    float dxs = (next[r*p+c+1] - next[r*p+c-1] - prev[r*p+c+1] + prev[r*p+c-1])*cross_deriv_scale;
    float dys = (next[(r+1)*p+c] - next[(r-1)*p+c] - prev[(r+1)*p+c] + prev[(r-1)*p+c])*cross_deriv_scale;
    float H[3][3] = {{dxx, dxy, dxs}, {dxy, dyy, dys}, {dxs, dys, dss}};
    float X[3];
    Solve3OpenCV(H, dD, X);
    xi = -X[2];
    xr = -X[1];
    xc = -X[0];
    if (fabs(xi)<0.5f && fabs(xr)<0.5f && fabs(xc)<0.5f)
      break;
    if (fabs(xi)>(float)(2147483647/3) || fabs(xr)>(float)(2147483647/3) || fabs(xc)>(float)(2147483647/3))   // INT_MAX/3
      return false;
    c += (int)rint(xc);
    r += (int)rint(xr);
    layer += (int)rint(xi);
    if (layer<1 || layer>L + P.lev[o] || c<SIFT_IMG_BORDER || c>=w - SIFT_IMG_BORDER ||
        r<SIFT_IMG_BORDER || r>=h - SIFT_IMG_BORDER)
      return false;
  }
  if (i>=SIFT_MAX_INTERP_STEPS)
    return false;
  const DogView prev = Dog(Lv, o, layer-1), cur = Dog(Lv, o, layer), next = Dog(Lv, o, layer+1);
  const DogView ir = cur + r*p;
  float dD[3] = {(ir[c+1] - ir[c-1])*deriv_scale, (ir[c+p] - ir[c-p])*deriv_scale,
                 (next[r*p+c] - prev[r*p+c])*deriv_scale};
  float t = dD[0]*xc + dD[1]*xr + dD[2]*xi;
  contr = ir[c]*img_scale + t*0.5f;
  if (fabs(contr)*L<dp.contrastThreshold)
    return false;
  float v2 = ir[c]*2.f;
  float dxx = (ir[c+1] + ir[c-1] - v2)*second_deriv_scale;
  float dyy = (ir[c+p] + ir[c-p] - v2)*second_deriv_scale;
  float dxy = (ir[c+p+1] - ir[c+p-1] - ir[c-p+1] + ir[c-p-1])*cross_deriv_scale;
  float tr = dxx + dyy;
  float det = dxx*dyy - dxy*dxy;
  const float et = dp.edgeThreshold;
  if (det<=0 || tr*tr*et>=(et + 1)*(et + 1)*det)
    return false;
  kp.x = (c + xc)*P.pix[o];
  kp.y = (r + xr)*P.pix[o];
  kp.packedOctave = o + (layer << 8) + ((int)rint((xi + 0.5f)*255) << 16);
  kp.size = dp.sigma*precise::exp2((layer - P.lev[o] + xi)/(float)L)*P.pix[o]*2;
  kp.response = fabs(contr);
  kp.contr = contr;
  kp.edge = tr*tr/det;
  kp.angle = 0.0f;
  kp.o = o;
  kp.layer = layer;
  kp.r = r;
  kp.c = c;
  return true;
}

// ---------------------------------------------------------------------------------------
// VLFeat's arithmetic (vl/sift.c of VLFeat 0.9.21), as the float path of cudaSiftCompat.cu

// fast_expn: exp(-x) by linear interpolation in a table
inline float FastExpnVLFeat(float x, device const float *tab)
{
  if (x>VL_EXPN_MAX)
    return 0.0f;
  x *= (float)(VL_EXPN_SZ/VL_EXPN_MAX);
  int i = (int)floor(x);
  float r = SubRN(x, (float)i);
  float a = tab[i], b = tab[i + 1];
  return AddRN(a, MulRN(r, SubRN(b, a)));
}

// vl_fast_atan2_f (with a fast division: faster, and the features stay as close)
inline float FastAtan2VLFeat(float y, float x)
{
  const float c3 = 0.1821F, c1 = 0.9675F;
  float abs_y = fabs(y) + VL_EPSILON_F;
  float angle, r;
  if (x>=0) {
    r = fast::divide(SubRN(x, abs_y), AddRN(x, abs_y));
    angle = 0.785398185f;   // (float)(pi/4)
  } else {
    r = fast::divide(AddRN(x, abs_y), SubRN(abs_y, x));
    angle = 2.35619449f;   // (float)(3*pi/4)
  }
  angle = AddRN(angle, MulRN(SubRN(MulRN(MulRN(c3, r), r), c1), r));
  return (y<0) ? -angle : angle;
}

// vl_fast_sqrt_f
inline float FastSqrtVLFeat(float x)
{
  if (x<1e-8)
    return 0;
  float xhalf = 0.5f*x;
  float u = as_type<float>(0x5f3759df - (as_type<int>(x) >> 1));
  u = MulRN(u, SubRN(1.5f, MulRN(MulRN(xhalf, u), u)));
  u = MulRN(u, SubRN(1.5f, MulRN(MulRN(xhalf, u), u)));
  return MulRN(x, u);
}

// vl_mod_2pi_f
inline float Mod2PiVLFeat(float x)
{
  const float twoPi = 6.28318548f;   // (float)(2*VL_PI)
  while (x>twoPi)
    x -= twoPi;
  while (x<0.0F)
    x += twoPi;
  return x;
}

// The input in [0, 1], as COLMAP converts it for VLFeat (x/255.0f)
inline float UnitVLFeat(float v)
{
  return v/255.0f;
}

// A sample of a row of copy_and_upsample_rows: odd samples average two pixels, except the last
inline float UpsampleRowVLFeat(device const float *row, int x0, int odd, int sw)
{
  if (odd && x0<sw - 1)
    return 0.5f*AddRN(UnitVLFeat(row[x0]), UnitVLFeat(row[x0 + 1]));
  return UnitVLFeat(row[x0]);
}

// update_gradient's magnitude and angle at (x, y), computed where needed
inline float2 GradientVLFeat(device const float *src, int x, int y, int w, int h, int p)
{
  device const float *s = src + y*p + x;
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
  g.x = FastSqrtVLFeat(AddRN(MulRN(gx, gx), MulRN(gy, gy)));
  g.y = Mod2PiVLFeat(AddTwoPiVLFeat(FastAtan2VLFeat(gy, gx)));
  return g;
}

// The level index of a VLFeat keypoint (whose VLFeat level is is)
inline int LevelIndexVLFeat(constant CompatPyramidM &P, constant CompatDetectParamsM &dp, thread const VLFeatKeypointM &kp)
{
  return kp.is - dp.s_min + P.lev[kp.o];
}

// vl_sift_detect's refinement of an extremum: up to 5 steps of Gauss elimination with moves of
// +-1 pixel, then the peak and edge tests
inline bool Refine(Levels Lv, int o, int layer, int r, int c, constant CompatDetectParamsM &dp, thread VLFeatKeypointM &kp)
{
  constant CompatPyramidM &P = *Lv.P;
  const int w = P.w[o], h = P.h[o], p = P.p[o];
  const int s = layer + dp.s_min - P.lev[o];   // VLFeat's level
  const int raster = r*w + c;
  int x = c, y = r;
  float Dx=0, Dy=0, Ds=0, Dxx=0, Dyy=0, Dss=0, Dxy=0, Dxs=0, Dys=0;
  float A[9], b[3];
  int ddx = 0, ddy = 0;
  const DogView prev = Dog(Lv, o, layer-1), cur = Dog(Lv, o, layer), next = Dog(Lv, o, layer+1);
  for (int iter=0;iter<VL_MAX_REFINE_STEPS;iter++) {
    x += ddx;
    y += ddy;
    const int q = y*p + x;
#define AT1(dx, dy) cur[q + (dy)*p + (dx)]
    Dx = 0.5f*(AT1(1, 0) - AT1(-1, 0));
    Dy = 0.5f*(AT1(0, 1) - AT1(0, -1));
    Ds = 0.5f*(next[q] - prev[q]);
    Dxx = SubRN(AT1(1, 0) + AT1(-1, 0), 2.0f*AT1(0, 0));
    Dyy = SubRN(AT1(0, 1) + AT1(0, -1), 2.0f*AT1(0, 0));
    Dss = SubRN(next[q] + prev[q], 2.0f*AT1(0, 0));
    Dxy = 0.25f*(AT1(1, 1) + AT1(-1, -1) - AT1(-1, 1) - AT1(1, -1));
    Dxs = 0.25f*(next[q + 1] + prev[q - 1] - next[q - 1] - prev[q + 1]);
    Dys = 0.25f*(next[q + p] + prev[q - p] - next[q - p] - prev[q + p]);
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
      float maxa = 0, maxabsa = 0;
      int maxi = -1;
      for (int i=j;i<3;i++) {
        float av = Aat(i, j), absa = fabs(av);
        if (absa>maxabsa) {
          maxa = av;
          maxabsa = absa;
          maxi = i;
        }
      }
      if (maxabsa<1e-10f) {
        b[0] = b[1] = b[2] = 0;
        break;
      }
      int i = maxi;
      const float inv = 1.0f/maxa;   // VLFeat divides by the pivot: the reciprocal is faster
      for (int jj=j;jj<3;jj++) {
        float tmp = Aat(i, jj);
        Aat(i, jj) = Aat(j, jj);
        Aat(j, jj) = tmp;
        Aat(j, jj) = MulRN(Aat(j, jj), inv);
      }
      float tmp = b[j];
      b[j] = b[i];
      b[i] = tmp;
      b[j] = MulRN(b[j], inv);
      for (int ii=j+1;ii<3;ii++) {
        float xx = Aat(ii, j);
        for (int jj=j;jj<3;jj++)
          Aat(ii, jj) = SubRN(Aat(ii, jj), MulRN(xx, Aat(j, jj)));
        b[ii] = SubRN(b[ii], MulRN(xx, b[j]));
      }
    }
    for (int i=2;i>0;i--) {
      float xx = b[i];
      for (int ii=i-1;ii>=0;ii--)
        b[ii] = SubRN(b[ii], MulRN(xx, Aat(ii, i)));
    }
#undef Aat
    ddx = ((b[0]>0.6f && x<w - 2) ? 1 : 0) + ((b[0]<-0.6f && x>1) ? -1 : 0);
    ddy = ((b[1]>0.6f && y<h - 2) ? 1 : 0) + ((b[1]<-0.6f && y>1) ? -1 : 0);
    if (ddx==0 && ddy==0)
      break;
  }
  float val = cur[y*p + x] + 0.5f*(AddRN(AddRN(MulRN(Dx, b[0]), MulRN(Dy, b[1])), MulRN(Ds, b[2])));
  float score = (Dxx + Dyy)*(Dxx + Dyy)/SubRN(MulRN(Dxx, Dyy), MulRN(Dxy, Dxy));
  float xn = x + b[0], yn = y + b[1], sn = s + b[2];
  const float te = dp.edgeThreshold;
  // The level is bounded by the lowest one, s_min (lower with the extension), and s_max
  bool good = fabs(val)>dp.peakThreshold && score<(te + 1)*(te + 1)/te && score>=0 &&
    fabs(b[0])<1.5f && fabs(b[1])<1.5f && fabs(b[2])<1.5f &&
    xn>=0 && xn<=w - 1 && yn>=0 && yn<=h - 1 && sn>=dp.s_min - P.lev[o] && sn<=dp.s_max;
  if (!good)
    return false;
  const float xper = P.pix[o];
  kp.raster = raster;
  kp.o = o;
  kp.is = s;
  kp.x = xn*xper;
  kp.y = yn*xper;
  kp.sigma = dp.sigma0*precise::exp2(sn/(float)dp.S)*xper;
  kp.angle = 0.0f;
  kp.val = val;
  kp.score = score;
  kp.ori = 0;
  return true;
}

// ---------------------------------------------------------------------------------------
// The kernels of the pipeline and their per-mode steps

// What differs between the modes besides the per-mode steps: the keypoint type, the border
// without extrema, the SIMD groups (keypoints) per threadgroup of OrientationsCompat, the
// threads per threadgroup (keypoint) of DescriptorsCompat, and the sizes of their histograms and
// of VLFeat's descriptor window
template<int M> struct ModeTraits;
template<> struct ModeTraits<MODE_OPENCV> {
  typedef OpenCVKeypoint Keypoint;
  enum { border = SIFT_IMG_BORDER, oriSG = ORI_SG_OPENCV, descNT = DESC_NT_OPENCV,
         descHist = (SIFT_DESCR_WIDTH + 2)*(SIFT_DESCR_WIDTH + 2)*(SIFT_DESCR_HIST_BINS + 2), window = 1 };
};
template<> struct ModeTraits<MODE_VLFEAT> {
  typedef VLFeatKeypointM Keypoint;
  enum { border = 1, oriSG = ORI_SG_VLFEAT, descNT = DESC_NT_VLFEAT,
         descHist = SIFT_DESCR_HIST_BINS*SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH, window = 2*VL_MAX_W + 1 };
};

// The first octave's input image, upsampled by 2 or not. OpenCV: bilinear (warpAffine with
// INTER_LINEAR), values 0 to 255. VLFeat (copy_and_upsample_rows): rows first, then columns,
// odd samples average two pixels except the last one that repeats the last pixel, values 0 to 1.
template<int M>
kernel void InputCompat(device float *ss [[buffer(0)]], constant CompatImageArgs &a [[buffer(1)]],
                        device const float *src [[buffer(2)]], uint2 pos [[thread_position_in_grid]])
{
  const int x = pos.x, y = pos.y;
  if (x>=a.w || y>=a.h)
    return;
  device float *dst = ss + a.dst;
  const int dp = a.p, sw = a.sw, sh = a.sh, sp = a.sp;
  if (M==MODE_VLFEAT) {
    if (!a.upsample) {
      dst[y*dp + x] = UnitVLFeat(src[y*sp + x]);
      return;
    }
    int x0 = x>>1, y0 = y>>1;
    float h0 = UpsampleRowVLFeat(src + y0*sp, x0, x&1, sw);
    if ((y&1) && y0<sh - 1)
      h0 = 0.5f*AddRN(h0, UpsampleRowVLFeat(src + (y0 + 1)*sp, x0, x&1, sw));
    dst[y*dp + x] = h0;
  } else {
    if (!a.upsample) {
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

template [[host_name("InputCompat_0")]] kernel void InputCompat<0>(device float *, constant CompatImageArgs &, device const float *, uint2);
template [[host_name("InputCompat_1")]] kernel void InputCompat<1>(device float *, constant CompatImageArgs &, device const float *, uint2);

// Next octave's first level: every other pixel (resize INTER_NEAREST to half size)
kernel void DownsampleCompat(device float *ss [[buffer(0)]], constant CompatImageArgs &a [[buffer(1)]],
                             uint2 pos [[thread_position_in_grid]])
{
  const int x = pos.x, y = pos.y;
  if (x>=a.w || y>=a.h)
    return;
  ss[a.dst + y*a.p + x] = ss[a.src + 2*y*a.sp + 2*x];
}

// Extrema of level `layer` of octave o over the 26 neighbours, refined and tested. OpenCV:
// above its threshold, and not smaller (larger for minima) than any neighbour. VLFeat: at least
// 0.8 times the peak threshold, and strictly larger (smaller) than all the neighbours.
template<int M>
kernel void FindExtremaCompat(device float *ss [[buffer(0)]], constant CompatPyramidM &P [[buffer(1)]],
                              constant CompatDetectParamsM &dp [[buffer(2)]], constant int2 &ol [[buffer(3)]],
                              device typename ModeTraits<M>::Keypoint *cands [[buffer(4)]],
                              device atomic_uint *counter [[buffer(5)]], constant int &maxCands [[buffer(6)]],
                              uint2 pos [[thread_position_in_grid]])
{
  const int B = ModeTraits<M>::border;
  const int o = ol.x, layer = ol.y;
  const Levels Lv = {ss, &P};
  const int w = P.w[o], h = P.h[o], p = P.p[o];
  int c = pos.x + B, r = pos.y + B;
  if (c>=w - B || r>=h - B)
    return;
  const DogView prev = Dog(Lv, o, layer-1) + (r*p + c);
  const DogView cur = Dog(Lv, o, layer) + (r*p + c);
  const DogView next = Dog(Lv, o, layer+1) + (r*p + c);
  const float v = cur[0];
  bool isMax, isMin;
  if (M==MODE_VLFEAT) {
    isMax = (v>=dp.peakPrefilter);
    isMin = (v<=-dp.peakPrefilter);
  } else {
    if (!(fabs(v)>dp.preThreshold))
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
    if (M==MODE_VLFEAT) {
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
  typename ModeTraits<M>::Keypoint kp;
  if (!Refine(Lv, o, layer, r, c, dp, kp))
    return;
  uint k = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
  if (k<(uint)maxCands)
    cands[k] = kp;
}

template [[host_name("FindExtremaCompat_0")]] kernel void FindExtremaCompat<0>(device float *, constant CompatPyramidM &, constant CompatDetectParamsM &, constant int2 &, device OpenCVKeypoint *, device atomic_uint *, constant int &, uint2);
template [[host_name("FindExtremaCompat_1")]] kernel void FindExtremaCompat<1>(device float *, constant CompatPyramidM &, constant CompatDetectParamsM &, constant int2 &, device VLFeatKeypointM *, device atomic_uint *, constant int &, uint2);

// The orientation histogram of one keypoint, one partial histogram per lane: OpenCV's
// (calcOrientationHist) and VLFeat's (vl_sift_calc_keypoint_orientations; false if the keypoint
// is outside its level). dx2 and dy2: VLFeat's squared offsets, per SIMD group. Both overloads
// take the same arguments, which OrientationsCompat calls for both modes (MSL has no
// `if constexpr`): OpenCV's uses neither dp, dx2, dy2 nor tab.
inline bool OrientationHist(Levels Lv, constant CompatDetectParamsM &dp, thread const OpenCVKeypoint &kp, threadgroup float *hist,
                            threadgroup float *dx2, threadgroup float *dy2, uint lane, device const float *tab)
{
  constant CompatPyramidM &P = *Lv.P;
  const int n = SIFT_ORI_HIST_BINS;
  const int o = kp.o;
  device const float *img = Lv.Get(o, kp.layer);
  const int rows = P.h[o], cols = P.w[o], p = P.p[o];
  float scl_octv = kp.size*0.5f*(1.0f/P.pix[o]);
  int radius = (int)rint(SIFT_ORI_RADIUS*scl_octv);
  float sigma = SIFT_ORI_SIG_FCTR*scl_octv;
  float expf_scale = -1.f/(2.f*sigma*sigma);
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
    float mag = precise::sqrt(fma(dx, dx, dy*dy));
    int bin = (int)rint((n/360.f)*ori);
    if (bin>=n)
      bin -= n;
    if (bin<0)
      bin += n;
    hist[bin] += wgt*mag;
  }
  return true;
}

inline bool OrientationHist(Levels Lv, constant CompatDetectParamsM &dp, thread const VLFeatKeypointM &kp, threadgroup float *hist,
                            threadgroup float *dx2, threadgroup float *dy2, uint lane, device const float *tab)
{
  constant CompatPyramidM &P = *Lv.P;
  const int n = SIFT_ORI_HIST_BINS;
  const int o = kp.o;
  const int w = P.w[o], h = P.h[o], p = P.p[o];
  device const float *gauss = Lv.Get(o, LevelIndexVLFeat(P, dp, kp));
  const float xper = P.pix[o];
  float x = kp.x/xper, y = kp.y/xper, sigma = kp.sigma/xper;
  int xi = (int)(x + 0.5f), yi = (int)(y + 0.5f);
  const float sigmaw = 1.5f*sigma;
  int W = min(max((int)floor(3.0f*sigmaw), 1), VL_MAX_W);
  if (xi<0 || xi>w - 1 || yi<0 || yi>h - 1)
    return false;
  // Squared offsets, as VLFeat rounds them
  for (int t=lane;t<2*W+1;t+=32) {
    float dx = (float)(xi + t - W) - x, dy = (float)(yi + t - W) - y;
    dx2[t] = MulRN(dx, dx);
    dy2[t] = MulRN(dy, dy);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  int ys0 = max(-W, -yi), ys1 = min(W, h - 1 - yi);
  int xs0 = max(-W, -xi), xs1 = min(W, w - 1 - xi);
  int nx = xs1 - xs0 + 1, ny = ys1 - ys0 + 1;
  // Here and below, reciprocals where VLFeat divides: faster, and the features stay as close
  const float invw = 1/(2*sigmaw*sigmaw);
  for (int t=lane;t<nx*ny;t+=32) {
    int ys = ys0 + t/nx, xs = xs0 + t%nx;
    float r2 = AddRN(dx2[xs + W], dy2[ys + W]);
    if (r2>=W*W + 0.6f)
      continue;
    float wgt = FastExpnVLFeat(r2*invw, tab);
    float2 g = GradientVLFeat(gauss, xi + xs, yi + ys, w, h, p);
    float mod = g.x, ang = g.y;
    float fbin = n*ang*(1/(2*VL_PI_F));
    int bin = (int)floor(fbin - 0.5f);   // VL_SIFT_BILINEAR_ORIENTATIONS, set in sift.c
    float rbin = fbin - bin - 0.5f;
    hist[(bin + n)%n] += MulRN(MulRN(1 - rbin, mod), wgt);
    hist[(bin + 1)%n] += MulRN(MulRN(rbin, mod), wgt);
  }
  return true;
}

// The keypoints of the peaks of the summed histogram t (2 bins of padding on each side):
// OpenCV's ([1 4 6 4 1] smoothing into smooth, all the peaks of at least 0.8 times the
// maximum) and VLFeat's (six box filters, the first VL_MAX_ORIENTATION_PEAKS peaks above 0.8
// times the maximum, of which the first maxOrientations are kept)
inline void OrientationPeaks(threadgroup float *t, thread const OpenCVKeypoint &kp, uint lane, threadgroup float *smooth, int maxOrientations,
                             device OpenCVKeypoint *kpts, device atomic_uint *counter, int maxKpts)
{
  const int n = SIFT_ORI_HIST_BINS;
  if (lane==0) {
    t[-1] = t[n-1];
    t[-2] = t[n-2];
    t[n] = t[0];
    t[n+1] = t[1];
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (int b=lane;b<n;b+=32)
    smooth[b] = fma(t[b-2] + t[b+2], 1.0f/16.0f, fma(t[b-1] + t[b+1], 4.0f/16.0f, t[b]*(6.0f/16.0f)));
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (lane==0) {
    float omax = smooth[0];
    for (int b=1;b<n;b++)
      omax = fmax(omax, smooth[b]);
    float mag_thr = omax*SIFT_ORI_PEAK_RATIO;
    for (int j=0;j<n;j++) {
      int l = j>0 ? j - 1 : n - 1;
      int r2 = j<n-1 ? j + 1 : 0;
      if (smooth[j]>smooth[l] && smooth[j]>smooth[r2] && smooth[j]>=mag_thr) {
        float bin = j + 0.5f*(smooth[l] - smooth[r2])/(smooth[l] - 2*smooth[j] + smooth[r2]);
        bin = bin<0 ? n + bin : bin>=n ? bin - n : bin;
        OpenCVKeypoint out = kp;
        out.angle = 360.f - (360.0f/n)*bin;
        if (fabs(out.angle - 360.f)<FLT_EPSILON)
          out.angle = 0.f;
        uint m = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        if (m<(uint)maxKpts)
          kpts[m] = out;
      }
    }
  }
}

inline void OrientationPeaks(threadgroup float *t, thread const VLFeatKeypointM &kp, uint lane, threadgroup float *smooth, int maxOrientations,
                             device VLFeatKeypointM *kpts, device atomic_uint *counter, int maxKpts)
{
  const int n = SIFT_ORI_HIST_BINS;
  // Six circular box filters: each bin from the previous values of its neighbours, as VLFeat
  for (int iter=0;iter<6;iter++) {
    float nh[2];
    for (int k=0;k<2;k++) {
      int i = lane + 32*k;
      if (i<n)
        nh[k] = MulRN(AddRN(AddRN(t[(i + n - 1)%n], t[i]), t[(i + 1)%n]), 1.0f/3.0f);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int k=0;k<2;k++)
      if (lane + 32*k<(uint)n)
        t[lane + 32*k] = nh[k];
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane==0) {
    float maxh = 0;
    for (int i=0;i<n;i++)
      maxh = fmax(maxh, t[i]);
    int nangles = 0;
    for (int i=0;i<n && nangles<VL_MAX_ORIENTATION_PEAKS;i++) {
      float h0 = t[i], hm = t[(i - 1 + n)%n], hp = t[(i + 1 + n)%n];
      if (h0>0.8f*maxh && h0>hm && h0>hp) {
        float di = (-0.5f*SubRN(hp, hm))/SubRN(AddRN(hp, hm), 2*h0);
        float th = (2*VL_PI_F*(i + di + 0.5f))/(float)n;
        if (nangles<maxOrientations) {
          VLFeatKeypointM out = kp;
          out.angle = th;
          out.ori = nangles;
          uint m = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
          if (m<(uint)maxKpts)
            kpts[m] = out;
        }
        nangles++;
      }
    }
  }
}

// The arguments of the orientations' indirect dispatch, from FindExtremaCompat's counter: the
// number of candidates (at most maxCands) and the threadgroups of perGroup candidates
kernel void OrientationArgs(device const uint *counter [[buffer(0)]], constant int &maxCands [[buffer(1)]],
                            constant int &perGroup [[buffer(2)]], device uint *args [[buffer(3)]])
{
  const uint n = min(counter[0], (uint)maxCands);
  args[0] = (n + perGroup - 1)/perGroup;   // MTLDispatchThreadgroupsIndirectArguments
  args[1] = 1;
  args[2] = 1;
  args[3] = n;                             // OrientationsCompat's numCands
}

// Orientations, one SIMD group per candidate and one keypoint per peak of its histogram of 36 bins
template<int M>
kernel void OrientationsCompat(device float *ss [[buffer(0)]], constant CompatPyramidM &P [[buffer(1)]],
                               constant CompatDetectParamsM &dp [[buffer(2)]], constant int &maxOrientations [[buffer(3)]],
                               device const typename ModeTraits<M>::Keypoint *cands [[buffer(4)]], constant int &numCands [[buffer(5)]],
                               device typename ModeTraits<M>::Keypoint *kpts [[buffer(6)]], device atomic_uint *counter [[buffer(7)]],
                               constant int &maxKpts [[buffer(8)]], device const float *tab [[buffer(9)]],
                               uint group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]])
{
  const int NW = ModeTraits<M>::oriSG;
  const int n = SIFT_ORI_HIST_BINS;
  const int TW = (M==MODE_VLFEAT ? 2*VL_MAX_W + 1 : 1);   // VLFeat: squared offsets
  threadgroup float lanehist[NW][32][n + 1];
  threadgroup float hist[NW][n + 4];
  threadgroup float smooth[NW][n];
  threadgroup float dx2[NW][TW], dy2[NW][TW];
  const int idx = group*NW + sg;
  if (idx>=numCands)
    return;
  const Levels Lv = {ss, &P};
  const typename ModeTraits<M>::Keypoint kp = cands[idx];
  for (int b=0;b<n;b++)
    lanehist[sg][lane][b] = 0.0f;
  if (!OrientationHist(Lv, dp, kp, lanehist[sg][lane], dx2[sg], dy2[sg], lane, tab))
    return;
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (int b=lane;b<n;b+=32) {
    float s = 0.0f;
    for (int l=0;l<32;l++)
      s += lanehist[sg][l][b];
    hist[sg][b + 2] = s;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  OrientationPeaks(hist[sg] + 2, kp, lane, smooth[sg], maxOrientations, kpts, counter, maxKpts);
}

template [[host_name("OrientationsCompat_0")]] kernel void OrientationsCompat<0>(device float *, constant CompatPyramidM &, constant CompatDetectParamsM &, constant int &, device const OpenCVKeypoint *, constant int &, device OpenCVKeypoint *, device atomic_uint *, constant int &, device const float *, uint, uint, uint);
template [[host_name("OrientationsCompat_1")]] kernel void OrientationsCompat<1>(device float *, constant CompatPyramidM &, constant CompatDetectParamsM &, constant int &, device const VLFeatKeypointM *, constant int &, device VLFeatKeypointM *, device atomic_uint *, constant int &, device const float *, uint, uint, uint);

// VLFeat's descriptor window, row by row: in row r, the columns [rowLo[r], rowLo[r] + rowLen[r])
// where the rotated window can be (WindowRange), and the prefix sums rowStart of the row lengths
// (WindowPrefix), to find the row of the t-th sample (WindowRow)
#define WINDOW_MAX_ROWS (2*VL_MAX_W + 1)

// Range of the integers j in [jmin, jmax] where a*j + b can be in (vlo, vhi), widened by one
// on each side (the exact test of each sample follows)
inline void WindowRange(float a, float b, float vlo, float vhi, thread int &jmin, thread int &jmax)
{
  if (fabs(a)<1e-6f)
    return;
  float j1 = (vlo - b)/a, j2 = (vhi - b)/a;
  if (j1>j2) {
    float t = j1;
    j1 = j2;
    j2 = t;
  }
  jmin = max(jmin, (int)floor(j1) - 1);
  jmax = min(jmax, (int)ceil(j2) + 1);
}

// Prefix sums of the row lengths (one thread)
inline void WindowPrefix(threadgroup const int *rowLen, threadgroup int *rowStart, int numRows)
{
  int s = 0;
  for (int r=0;r<numRows;r++) {
    rowStart[r] = s;
    s += rowLen[r];
  }
  rowStart[numRows] = s;
}

// The row of the t-th sample
inline int WindowRow(threadgroup const int *rowStart, int numRows, int t)
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

// The threadgroup memory of the descriptor histograms: the histogram of the thread's group
// (hist); VLFeat's window (the terms of the rotated offsets, and the rows
// of the samples near the rotated window); the cosine and sine of the orientation (rot)
struct DescriptorShared {
  threadgroup float *hist;
  threadgroup float *tabA, *tabB, *tabC, *tabD;
  threadgroup int *rowLo, *rowLen, *rowStart;
  threadgroup float *rot;
};

// The descriptor histograms of one keypoint, one per group of DESC_GROUP lanes, from the gradients in a
// rotated window, in an order that doesn't vary between runs. OpenCV (calcSIFTDescriptor):
// Gaussian weights, trilinear interpolation, histograms of (d + 2)^2 x (n + 2) bins with
// borders. VLFeat (vl_sift_calc_keypoint_descriptor): fast_expn weights, its bilinear spatial
// and orientation interpolation, the samples near the rotated window only (WindowRange).
inline void DescriptorRotation(thread const OpenCVKeypoint &kp, threadgroup float *rot)
{
  // OpenCV's descriptor orientation (360 - angle, 0 for 360) and its cosine and sine
  float ori = 360.f - kp.angle;
  if (fabs(ori - 360.f)<FLT_EPSILON)
    ori = 0.f;
  const float a = ori*0.0174532924f;   // (float)(M_PI/180)
  rot[0] = precise::cos(a);
  rot[1] = precise::sin(a);
}

inline void DescriptorRotation(thread const VLFeatKeypointM &kp, threadgroup float *rot)
{
  rot[0] = precise::cos(kp.angle);
  rot[1] = precise::sin(kp.angle);
}

inline void DescriptorHist(Levels Lv, constant CompatDetectParamsM &dp, thread const OpenCVKeypoint &kp, DescriptorShared sh,
                           uint tid, uint ntg, uint lane, device const float *tab)
{
  constant CompatPyramidM &P = *Lv.P;
  const int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;
  const int o = kp.o;
  device const float *img = Lv.Get(o, kp.layer);
  const int rows = P.h[o], cols = P.w[o], p = P.p[o];
  // Position and size in the octave, as OpenCV computes them from the final keypoint
  const float inv = 1.0f/P.pix[o];
  float ptx = kp.x*inv, pty = kp.y*inv;
  int ptxi = (int)rint(ptx), ptyi = (int)rint(pty);
  float scl = kp.size*inv*0.5f;
  float ori = 360.f - kp.angle;
  if (fabs(ori - 360.f)<FLT_EPSILON)
    ori = 0.f;
  float cos_t = sh.rot[0], sin_t = sh.rot[1];
  float bins_per_rad = n/360.f;
  float exp_scale = -1.f/(d*d*0.5f);
  float hist_width = SIFT_DESCR_SCL_FCTR*scl;
  int radius = (int)rint(hist_width*1.4142135623730951f*(d + 1)*0.5f);
  radius = min(radius, (int)precise::sqrt((float)cols*cols + (float)rows*rows));
  cos_t = cos_t/hist_width;
  sin_t = sin_t/hist_width;
  const int side = 2*radius + 1, total = side*side;
  const int hoffs[8] = {0, 1, n + 2, n + 3, (d + 2)*(n + 2), (d + 2)*(n + 2) + 1, (d + 3)*(n + 2), (d + 3)*(n + 2) + 1};
  for (int base=0;base<total;base+=ntg) {   // the same number of rounds for all SIMD groups
    const int k = base + tid;
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
      float mag = precise::sqrt(fma(dx, dx, dy*dy))*ExpfOpenCV((c_rot*c_rot + r_rot*r_rot)*exp_scale);
      float obin = (ang - ori)*bins_per_rad;
      int r0 = (int)floor(rbin);
      int c0 = (int)floor(cbin);
      int o0 = (int)floor(obin);
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
    if (simd_any(h0>=0))   // most SIMD groups of the square's corners have nothing to add
      for (int q=0;q<8;q++)
        GroupHistAdd(sh.hist, h0>=0 ? h0 + hoffs[q] : -1, h0>=0 ? contrib[q] : 0.0f, lane);
  }
}

inline void DescriptorHist(Levels Lv, constant CompatDetectParamsM &dp, thread const VLFeatKeypointM &kp, DescriptorShared sh,
                           uint tid, uint ntg, uint lane, device const float *tab)
{
  constant CompatPyramidM &P = *Lv.P;
  const int NBO = SIFT_DESCR_HIST_BINS, NBP = SIFT_DESCR_WIDTH;
  const int o = kp.o;
  const int w = P.w[o], h = P.h[o], p = P.p[o];
  device const float *gauss = Lv.Get(o, LevelIndexVLFeat(P, dp, kp));
  const float xper = P.pix[o];
  const float x = kp.x/xper, y = kp.y/xper, sigma = kp.sigma/xper;
  const int xi = (int)(x + 0.5f), yi = (int)(y + 0.5f);
  const float st0 = sh.rot[1], ct0 = sh.rot[0];
  const float SBP = 3.0f*sigma + VL_EPSILON_F;
  const int W = min((int)floor(1.41421356f*SBP*(NBP + 1)/2.0f + 0.5f), VL_MAX_W);
  // The rotated offsets' terms, as VLFeat rounds them: nx = (A[dx] + B[dy])/SBP, ny = (C[dx] + D[dy])/SBP
  for (int t=tid;t<2*W+1;t+=ntg) {
    float dx = (float)(xi + t - W) - x;
    float dy = (float)(yi + t - W) - y;
    sh.tabA[t] = MulRN(ct0, dx);
    sh.tabC[t] = MulRN(-st0, dx);
    sh.tabB[t] = MulRN(st0, dy);
    sh.tabD[t] = MulRN(ct0, dy);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (xi<0 || xi>=w || yi<0 || yi>=h - 1)   // the same for the whole threadgroup
    return;
  int dy0 = max(-W, 1 - yi), dy1 = min(W, h - yi - 2);
  int dx0 = max(-W, 1 - xi), dx1 = min(W, w - xi - 2);
  const int ny = dy1 - dy0 + 1;   // at most 2*VL_MAX_W + 1 = WINDOW_MAX_ROWS
  const float invSBP = 1/SBP;   // VLFeat divides by SBP: the reciprocal is faster
  // Only the samples near the rotated window (|nx|, |ny| < 2.5): about half of the square
  const float cs = ct0/SBP, sn = st0/SBP, ox = (float)(xi - x);
  for (int r=tid;r<ny;r+=ntg) {
    const float dyf = (float)(yi + dy0 + r) - y;
    int jmin = dx0, jmax = dx1;
    WindowRange(cs, cs*ox + sn*dyf, -2.5f, 2.5f, jmin, jmax);
    WindowRange(-sn, -sn*ox + cs*dyf, -2.5f, 2.5f, jmin, jmax);
    sh.rowLo[r] = jmin;
    sh.rowLen[r] = (jmax>=jmin ? jmax - jmin + 1 : 0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid==0)
    WindowPrefix(sh.rowLen, sh.rowStart, ny);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int total = sh.rowStart[ny];
  for (int base=0;base<total;base+=ntg) {   // the same number of rounds for all SIMD groups
    const int t = base + tid;
    const bool valid = (t<total);
    const int row = WindowRow(sh.rowStart, ny, min(t, total - 1));
    const int dyi = dy0 + row;
    const int dxi = sh.rowLo[row] + min(t, total - 1) - sh.rowStart[row];
    float fnx = MulRN(AddRN(sh.tabA[dxi + W], sh.tabB[dyi + W]), invSBP);
    float fny = MulRN(AddRN(sh.tabC[dxi + W], sh.tabD[dyi + W]), invSBP);
    // Samples whose 8 bins are all outside the descriptor add nothing: skip their gradient and
    // their weights, and whole SIMD groups of them
    const bool inside = valid && fnx>=-2.5f && fnx<2.5f && fny>=-2.5f && fny<2.5f;
    if (!simd_any(inside))
      continue;
    float2 g = (inside ? GradientVLFeat(gauss, xi + dxi, yi + dyi, w, h, p) : float2(0.0f, 0.0f));
    float mod = g.x, angle = g.y;
    float theta = Mod2PiVLFeat(SubRN(angle, kp.angle));
    float nt = MulRN(MulRN((float)NBO, theta), 1/(2*VL_PI_F));
    const float wsigma = (float)(NBP/2);
    float win = FastExpnVLFeat(MulRN(AddRN(MulRN(fnx, fnx), MulRN(fny, fny)), 1/(2.0f*wsigma*wsigma)), tab);
    // VLFeat subtracts in double here, exactly: the float operations round the same way
    int binx = (int)floor(SubRN(fnx, 0.5f));
    int biny = (int)floor(SubRN(fny, 0.5f));
    int bint = (int)floor(nt);
    float rbinx = SubRN(fnx, (float)binx + 0.5f);
    float rbiny = SubRN(fny, (float)biny + 0.5f);
    float rbint = SubRN(nt, (float)bint);
    for (int dbinx=0;dbinx<2;dbinx++)
      for (int dbiny=0;dbiny<2;dbiny++)
        for (int dbint=0;dbint<2;dbint++) {
          int bin = -1;
          float weight = 0.0f;
          if (valid && binx + dbinx>=-(NBP/2) && binx + dbinx<(NBP/2) && biny + dbiny>=-(NBP/2) && biny + dbiny<(NBP/2)) {
            weight = MulRN(MulRN(MulRN(MulRN(win, mod), fabs(SubRN((float)(1 - dbinx), rbinx))),
                                 fabs(SubRN((float)(1 - dbiny), rbiny))),
                           fabs(SubRN((float)(1 - dbint), rbint)));
            bin = (biny + dbiny + NBP/2)*NBO*NBP + (binx + dbinx + NBP/2)*NBO + (bint + dbint)%NBO;
          }
          GroupHistAdd(sh.hist, bin, weight, lane);
        }
  }
}

// The descriptor raw (128 values) from the groups' histograms whist (NH of size HS):
// OpenCV's (wrap the orientation bins, clip at 0.2 of the norm, renormalize, quantize to 8 bits)
// and VLFeat's (normalize_histogram) or, with rootSift, COLMAP's (L1-root, 8 bits)
inline void DescriptorValues(thread const OpenCVKeypoint &, threadgroup const float *whist, int NH, int HS, threadgroup float *raw,
                             threadgroup float *norm, bool rootSift, uint tid, uint ntg)
{
  const int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;
  for (int k=tid;k<d*d*n;k+=ntg) {
    int i = k/(d*n), j = (k/n)%d, b = k%n;
    int h0 = ((i + 1)*(d + 2) + (j + 1))*(n + 2) + b;
    float v = 0.0f, v2 = 0.0f;
    for (int g=0;g<NH;g++) {
      v += whist[g*HS + h0];
      if (b<2)
        v2 += whist[g*HS + h0 + n];
    }
    raw[k] = (b<2 ? v + v2 : v);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid==0) {
    float nrm2 = 0;
    for (int k=0;k<d*d*n;k++)
      nrm2 = fma(raw[k], raw[k], nrm2);
    *norm = precise::sqrt(nrm2)*SIFT_DESCR_MAG_THR;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid<(uint)(d*d*n))
    raw[tid] = fmin(raw[tid], *norm);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid==0) {
    float nrm2 = 0;
    for (int k=0;k<d*d*n;k++)
      nrm2 += raw[k]*raw[k];
    *norm = SIFT_INT_DESCR_FCTR/fmax(precise::sqrt(nrm2), FLT_EPSILON);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // OpenCV's 8-bit values, scaled to unit length for MatchSiftData
  for (int k=tid;k<d*d*n;k+=ntg)
    raw[k] = fmin(fmax(rint(raw[k]*(*norm)), 0.0f), 255.0f);
}

inline void DescriptorValues(thread const VLFeatKeypointM &, threadgroup const float *whist, int NH, int HS, threadgroup float *raw,
                             threadgroup float *norm, bool rootSift, uint tid, uint ntg)
{
  const int NBO = SIFT_DESCR_HIST_BINS, NBP = SIFT_DESCR_WIDTH;
  for (int t=tid;t<NBO*NBP*NBP;t+=ntg) {
    float v = 0.0f;
    for (int g=0;g<NH;g++)
      v += whist[g*HS + t];
    raw[t] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Normalize, truncate at 0.2 and normalize again (normalize_histogram): the sums in VLFeat's
  // order, the rest in parallel
  for (int pass=0;pass<2;pass++) {
    if (tid==0) {
      float s = 0.0f;
      for (int k=0;k<NBO*NBP*NBP;k++)
        s = AddRN(s, MulRN(raw[k], raw[k]));
      *norm = AddRN(FastSqrtVLFeat(s), VL_EPSILON_F);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid<(uint)(NBO*NBP*NBP)) {
      float v = raw[tid]/(*norm);
      raw[tid] = (pass==0 && v>0.2f ? 0.2f : v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (rootSift) {
    // COLMAP: L1RootNormalizeFeatureDescriptors, TransformVLFeatToUBCFeatureDescriptors,
    // FeatureDescriptorsToUnsignedByte
    if (tid==0) {
      float l1 = 0.0f;
      for (int k=0;k<NBO*NBP*NBP;k++)
        l1 += fabs(raw[k]);
      *norm = (l1>0 ? 1.0f/l1 : 0.0f);   // no descriptor (keypoint at the border): zero, not COLMAP's NaN
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // COLMAP's 8-bit values, scaled to unit length for MatchSiftData
    if (tid<(uint)(NBO*NBP*NBP)) {
      float v = precise::sqrt(MulRN(raw[tid], *norm));
      raw[tid] = fmin(255.0f, round(512.0f*v));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// 1 / the norm of the 128 8-bit values of raw (summed in order by one thread)
inline float InvNorm(threadgroup const float *raw, threadgroup float *norm, uint tid)
{
  if (tid==0) {
    float n2 = 0.0f;
    for (int k=0;k<128;k++)
      n2 += raw[k]*raw[k];
    *norm = (n2>0 ? precise::rsqrt(n2) : 0.0f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return *norm;
}

// The fields of a SiftPoint, its descriptor apart
inline void WritePoint(device SiftPoint &pt, float x, float y, float scale, float sharpness, float edgeness,
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

// The point: OpenCV's descriptor and fields (from the coordinates of the first octave to the
// input's, outScale), or VLFeat's (COLMAP's descriptor in OpenCV's bin order)
inline void WriteDescriptor(thread const OpenCVKeypoint &kp, constant CompatPyramidM &P, threadgroup float *raw, threadgroup float *norm,
                            bool rootSift, float outScale, device SiftPoint &pt, uint tid, uint ntg)
{
  const float inv = InvNorm(raw, norm, tid);
  for (int k=tid;k<128;k+=ntg)
    pt.data[k] = raw[k]*inv;
  if (tid==0)
    WritePoint(pt, kp.x*outScale, kp.y*outScale, kp.size*outScale*0.5f, kp.contr, kp.edge, kp.angle, P.pix[kp.o]*outScale);
}

inline void WriteDescriptor(thread const VLFeatKeypointM &kp, constant CompatPyramidM &P, threadgroup float *raw, threadgroup float *norm,
                            bool rootSift, float outScale, device SiftPoint &pt, uint tid, uint ntg)
{
  if (rootSift) {
    const float inv = InvNorm(raw, norm, tid);
    if (tid<128)
      pt.data[tid - tid%8 + (8 - tid%8)%8] = raw[tid]*inv;   // OpenCV's bin order
  } else if (tid<128)
    pt.data[tid] = raw[tid];
  if (tid==0) {
    float deg = kp.angle*180.0f/VL_PI_F;   // VLFeat's angles are in [0, 2 pi]
    WritePoint(pt, kp.x, kp.y, kp.sigma, kp.val, kp.score, deg>=360.0f ? deg - 360.0f : deg, P.pix[kp.o]);
  }
}

// Descriptors, one threadgroup per keypoint: 4x4 cells of 8 orientation bins, of unit length
template<int M>
kernel void DescriptorsCompat(device float *ss [[buffer(0)]], constant CompatPyramidM &P [[buffer(1)]],
                              constant CompatDetectParamsM &dp [[buffer(2)]], constant float &outScale [[buffer(3)]],
                              constant int &rootSift [[buffer(4)]], device const typename ModeTraits<M>::Keypoint *kpts [[buffer(5)]],
                              constant int &numKpts [[buffer(6)]], device SiftPoint *out [[buffer(7)]], device const float *tab [[buffer(8)]],
                              uint idx [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
  const int NT = ModeTraits<M>::descNT, NH = NT/DESC_GROUP, HS = ModeTraits<M>::descHist;
  const int TW = ModeTraits<M>::window, WR = (M==MODE_VLFEAT ? WINDOW_MAX_ROWS : 0);
  threadgroup float whist[NH*HS];
  threadgroup float raw[128];
  threadgroup float norm[1];
  threadgroup float tabA[TW], tabB[TW], tabC[TW], tabD[TW];
  threadgroup int rowLo[WR + 1], rowLen[WR + 1], rowStart[WR + 1];
  threadgroup float rot[2];
  if ((int)idx>=numKpts)
    return;
  const Levels Lv = {ss, &P};
  const typename ModeTraits<M>::Keypoint kp = kpts[idx];
  for (int k=tid;k<NH*HS;k+=NT)
    whist[k] = 0.0f;
  if (tid==0)
    DescriptorRotation(kp, rot);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const DescriptorShared sh = {whist + (tid/DESC_GROUP)*HS, tabA, tabB, tabC, tabD, rowLo, rowLen, rowStart, rot};
  DescriptorHist(Lv, dp, kp, sh, tid, NT, lane, tab);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  DescriptorValues(kp, whist, NH, HS, raw, norm, rootSift!=0, tid, NT);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  WriteDescriptor(kp, P, raw, norm, rootSift!=0, outScale, out[idx], tid, NT);
}

template [[host_name("DescriptorsCompat_0")]] kernel void DescriptorsCompat<0>(device float *, constant CompatPyramidM &, constant CompatDetectParamsM &, constant float &, constant int &, device const OpenCVKeypoint *, constant int &, device SiftPoint *, device const float *, uint, uint, uint, uint);
template [[host_name("DescriptorsCompat_1")]] kernel void DescriptorsCompat<1>(device float *, constant CompatPyramidM &, constant CompatDetectParamsM &, constant float &, constant int &, device const VLFeatKeypointM *, constant int &, device SiftPoint *, device const float *, uint, uint, uint, uint);
