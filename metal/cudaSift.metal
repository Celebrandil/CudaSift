//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//              Apple Metal backend                       //
//********************************************************//

// Metal versions of the CudaSift kernels (cudaSiftD.cu and matching.cu). Most are
// close to line by line ports of the CUDA kernels. A few are restructured, as
// explained in their comments: fused DoG and extrema detection, histograms summed
// in a fixed order instead of with atomics (so that results are deterministic), and
// a matching kernel based on SIMD group matrix operations. cudaSiftD.h,
// cudaSiftMetalShared.h and cudaSiftRandom.h are prepended to this file before it is
// compiled.
//
// CUDA -> Metal: __syncthreads -> threadgroup_barrier, ShiftUp/ShiftDown/Shuffle ->
// simd_shuffle_up/down/simd_shuffle (Apple SIMD groups are 32 wide, like warps),
// __any_sync -> simd_any, tex2D -> sample() with a pixel coordinate, clamp-to-edge,
// linear sampler, __constant__ memory -> constant buffers, d_PointCounter -> a
// device atomic_uint buffer.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
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

#define NDIM 128

// Keypoint before its descriptor is computed. Each octave has its own region of
// maxPts candidates, so that the octaves can be processed concurrently.
struct CandPoint {
  float xpos;
  float ypos;
  float scale;
  float sharpness;
  float edgeness;
  float orientation;
  float subsampling;
  float pad;
};

constexpr sampler texSampler(coord::pixel, address::clamp_to_edge, filter::linear);

static inline uint LoadCounter(device atomic_uint *counter, int i)
{
  return atomic_load_explicit(&counter[i], memory_order_relaxed);
}

static inline void MaxCounter(device atomic_uint *counter, int i, int j)
{
  atomic_fetch_max_explicit(&counter[i], LoadCounter(counter, j), memory_order_relaxed);
}

static inline uint IncCounter(device atomic_uint *counter, int i)
{
  return atomic_fetch_add_explicit(&counter[i], 1u, memory_order_relaxed);
}

///////////////////////////////////////////////////////////////////////////////
// Lowpass filter and subsample image
///////////////////////////////////////////////////////////////////////////////

// As ScaleDown in cudaSiftD.cu (same 5-tap filter, operations and edge clamping),
// with one thread per output pixel instead of per-row block synchronization.
kernel void ScaleDownDirect(device float *d_Result [[buffer(0)]], device const float *d_Data [[buffer(1)]],
                            constant ImageParams &p [[buffer(2)]], constant float *d_ScaleDownKernel [[buffer(3)]],
                            uint2 gid [[thread_position_in_grid]])
{
  const int x = gid.x;
  const int y = gid.y;
  const int width = p.width, pitch = p.pitch, height = p.height;
  if (x>=width/2 || y>=height/2)
    return;
  float k0 = d_ScaleDownKernel[0];
  float k1 = d_ScaleDownKernel[1];
  float k2 = d_ScaleDownKernel[2];
  int xc[5];
#pragma unroll
  for (int j=0;j<5;j++)
    xc[j] = max(min(2*x + j - 2, width-1), 0);
  float h[5];
#pragma unroll
  for (int i=0;i<5;i++) {
    device const float *row = d_Data + max(min(2*y + i - 2, height-1), 0)*pitch;
    h[i] = k0*(row[xc[0]]+row[xc[4]]) + k1*(row[xc[1]]+row[xc[3]]) + k2*row[xc[2]];
  }
  d_Result[y*p.newpitch + x] = k2*h[2] + k0*(h[0]+h[4]) + k1*(h[1]+h[3]);
}

kernel void ScaleUp(device float *d_Result [[buffer(0)]], device const float *d_Data [[buffer(1)]],
                    constant ImageParams &p [[buffer(2)]],
                    uint2 bid [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]])
{
  const int tx = tid.x;
  const int ty = tid.y;
  const int width = p.width, pitch = p.pitch, height = p.height, newpitch = p.newpitch;
  int x = bid.x*SCALEUP_W + 2*tx;
  int y = bid.y*SCALEUP_H + 2*ty;
  if (x<2*width && y<2*height) {
    int xl = bid.x*(SCALEUP_W/2) + tx;
    int yu = bid.y*(SCALEUP_H/2) + ty;
    int xr = min(xl + 1, width - 1);
    int yd = min(yu + 1, height - 1);
    float vul = d_Data[yu*pitch + xl];
    float vur = d_Data[yu*pitch + xr];
    float vdl = d_Data[yd*pitch + xl];
    float vdr = d_Data[yd*pitch + xr];
    d_Result[(y + 0)*newpitch + x + 0] = vul;
    d_Result[(y + 0)*newpitch + x + 1] = 0.50f*(vul + vur);
    d_Result[(y + 1)*newpitch + x + 0] = 0.50f*(vul + vdl);
    d_Result[(y + 1)*newpitch + x + 1] = 0.25f*(vul + vur + vdl + vdr);
  }
}

kernel void LowPassBlock(device const float *d_Image [[buffer(0)]], device float *d_Result [[buffer(1)]],
                         constant ImageParams &p [[buffer(2)]], constant float *k [[buffer(3)]],
                         uint2 bid [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]])
{
  threadgroup float xrows[16][32];
  const int tx = tid.x;
  const int ty = tid.y;
  const int width = p.width, pitch = p.pitch, height = p.height, newpitch = p.newpitch;   // source, result pitches
  const int xp = bid.x*LOWPASS_W + tx;
  const int yp = bid.y*LOWPASS_H + ty;
  const int N = 16;
  int xl = max(min(xp - 4, width-1), 0);
#pragma unroll
  for (int l=-8;l<4;l+=4) {
    int ly = l + ty;
    int yl = max(min(yp + l + 4, height-1), 0);
    float val = d_Image[yl*pitch + xl];
    val = k[4]*simd_shuffle_down(val, 4) +
      k[3]*(simd_shuffle_down(val, 5) + simd_shuffle_down(val, 3)) +
      k[2]*(simd_shuffle_down(val, 6) + simd_shuffle_down(val, 2)) +
      k[1]*(simd_shuffle_down(val, 7) + simd_shuffle_down(val, 1)) +
      k[0]*(simd_shuffle_down(val, 8) + val);
    xrows[ly + 8][tx] = val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
  for (int l=4;l<LOWPASS_H;l+=4) {
    int ly = l + ty;
    int yl = min(yp + l + 4, height-1);
    float val = d_Image[yl*pitch + xl];
    val = k[4]*simd_shuffle_down(val, 4) +
      k[3]*(simd_shuffle_down(val, 5) + simd_shuffle_down(val, 3)) +
      k[2]*(simd_shuffle_down(val, 6) + simd_shuffle_down(val, 2)) +
      k[1]*(simd_shuffle_down(val, 7) + simd_shuffle_down(val, 1)) +
      k[0]*(simd_shuffle_down(val, 8) + val);
    xrows[(ly + 8)%N][tx] = val;
    int ys = yp + l - 4;
    if (xp<width && ys<height && tx<LOWPASS_W)
      d_Result[ys*newpitch + xp] = k[4]*xrows[(ly + 0)%N][tx] +
        k[3]*(xrows[(ly - 1)%N][tx] + xrows[(ly + 1)%N][tx]) +
        k[2]*(xrows[(ly - 2)%N][tx] + xrows[(ly + 2)%N][tx]) +
        k[1]*(xrows[(ly - 3)%N][tx] + xrows[(ly + 3)%N][tx]) +
        k[0]*(xrows[(ly - 4)%N][tx] + xrows[(ly + 4)%N][tx]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  int ly = LOWPASS_H + ty;
  int ys = yp + LOWPASS_H - 4;
  if (xp<width && ys<height && tx<LOWPASS_W)
    d_Result[ys*newpitch + xp] = k[4]*xrows[(ly + 0)%N][tx] +
      k[3]*(xrows[(ly - 1)%N][tx] + xrows[(ly + 1)%N][tx]) +
      k[2]*(xrows[(ly - 2)%N][tx] + xrows[(ly + 2)%N][tx]) +
      k[1]*(xrows[(ly - 3)%N][tx] + xrows[(ly + 3)%N][tx]) +
      k[0]*(xrows[(ly - 4)%N][tx] + xrows[(ly + 4)%N][tx]);
}

///////////////////////////////////////////////////////////////////////////////
// LaplaceMulti and FindPointsMulti (cudaSiftD.cu) fused into one kernel: the DoG
// layers are computed tile by tile in threadgroup memory and never written to device
// memory. Each threadgroup covers MINMAX_W columns and FUSED_FY*MINMAX_H rows, i.e.
// FUSED_FY FindPointsMultiNew blocks, and reproduces their logic, including the limit
// of 32 candidates per block. The filters use the same operations, order and edge
// clamping as LaplaceMultiMem.
///////////////////////////////////////////////////////////////////////////////

// FUSED_FY (FindPoints tiles per threadgroup) and FUSED_NT (threads) are in cudaSiftMetalShared.h
#define FUSED_RH   (MINMAX_H*FUSED_FY + 2)    // DoG region rows, incl. 1-row halo
#define FUSED_RW   (MINMAX_W + 2)             // DoG region columns, incl. 1-column halo (one SIMD group)
#define FUSED_VW   (FUSED_RW + 2*LAPLACE_R)   // vertically filtered columns
#define FUSED_SH   5                          // rows per thread in vertical pass
static_assert(FUSED_RW==32, "one SIMD group per row of the DoG region");
static_assert(FUSED_NT%32==0 && FUSED_NT/32>=FUSED_FY, "one SIMD group per FindPoints tile");
static_assert(FUSED_NT>=FUSED_RH*FUSED_RW/4, "horizontal pass: 4 values per thread");
static_assert(FUSED_NT>=FUSED_VW*((FUSED_RH + FUSED_SH - 1)/FUSED_SH), "vertical pass: FUSED_SH rows per thread");

kernel void LaplaceFindPoints(device const float *d_Image [[buffer(0)]], device CandPoint *d_Cand [[buffer(1)]],
                              constant FindPointsParams &p [[buffer(2)]], constant float *d_LaplaceKernel [[buffer(3)]],
                              device atomic_uint *d_PointCounter [[buffer(4)]],
                              uint2 bid [[threadgroup_position_in_grid]], uint ltid [[thread_index_in_threadgroup]])
{
  threadgroup float4 vbuf4[FUSED_RH*FUSED_VW/4];     // vertically filtered image, current scale
  threadgroup float4 dogs4[3][FUSED_RH*FUSED_RW/4];  // ring buffer of the three latest DoG layers
  threadgroup float *vbuf = (threadgroup float*)vbuf4;
  threadgroup float (*dogs)[FUSED_RH*FUSED_RW] = (threadgroup float (*)[FUSED_RH*FUSED_RW])dogs4;
  threadgroup ushort points[FUSED_FY][2*FUSED_RW];
  const int width = p.width, pitch = p.pitch, height = p.height, octave = p.octave;
  const float thresh = p.thresh;
  const int tid = ltid;
  const int minx = bid.x*MINMAX_W;
  const int y0 = bid.y*MINMAX_H*FUSED_FY;

  // Vertical pass: column vc (image column clamp(minx-1-LAPLACE_R+vc)), region rows
  // r0..r0+FUSED_SH-1 (image rows y0-1+r, out-of-image rows are computed but never used)
  const int vc = tid%FUSED_VW;
  const int r0 = (tid/FUSED_VW)*FUSED_SH;
  const bool vact = (r0<FUSED_RH);
  float src[FUSED_SH + 2*LAPLACE_R];
  if (vact) {
    device const float *data = d_Image + max(min(minx - 1 - LAPLACE_R + vc, width-1), 0);
#pragma unroll
    for (int i=0;i<FUSED_SH+2*LAPLACE_R;i++)
      src[i] = data[max(0, min(y0 - 1 + r0 - LAPLACE_R + i, height - 1))*pitch];
  }
  // Horizontal pass: thread handles region row hr, columns 4*hg..4*hg+3. Region position
  // (rr, cc) holds the DoG at image position (clamp(y0-1+rr), clamp(minx-1+cc)), filtered
  // around that clamped position. Without clamping, the 4 outputs read 12 consecutive values.
  const int hr = tid/(FUSED_RW/4);
  const int hg = tid%(FUSED_RW/4);
  const bool hact = (hr<FUSED_RH);
  const int vrow = (max(0, min(y0 - 1 + hr, height - 1)) - (y0 - 1))*FUSED_VW;
  int hcol[4];
#pragma unroll
  for (int i=0;i<4;i++)
    hcol[i] = max(0, min(minx - 1 + 4*hg + i, width - 1)) - (minx - 1 - LAPLACE_R);
  const bool hcontig = (hcol[0]==4*hg + LAPLACE_R && hcol[3]==4*hg + 3 + LAPLACE_R);
  float prev[4];
  for (int scale=0;scale<LAPLACE_S;scale++) {
    constant float *kptr = d_LaplaceKernel + octave*12*16 + scale*16;
    float kern[LAPLACE_R + 1];
#pragma unroll
    for (int i=0;i<=LAPLACE_R;i++)
      kern[i] = kptr[i];
    if (vact) {
#pragma unroll
      for (int k=0;k<FUSED_SH;k++) {
        float sum = kern[0]*src[k + LAPLACE_R];
#pragma unroll
        for (int j=1;j<=LAPLACE_R;j++)
          sum += kern[j]*(src[k + LAPLACE_R - j] + src[k + LAPLACE_R + j]);
        if (r0 + k<FUSED_RH)
          vbuf[(r0 + k)*FUSED_VW + vc] = sum;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (hact) {
      float res[4];
      if (hcontig) {
        float w[4 + 2*LAPLACE_R];
        threadgroup const float4 *v4 = (threadgroup const float4*)(vbuf + vrow + 4*hg);
#pragma unroll
        for (int i=0;i<(4 + 2*LAPLACE_R)/4;i++) {
          float4 v = v4[i];
          w[4*i + 0] = v.x;
          w[4*i + 1] = v.y;
          w[4*i + 2] = v.z;
          w[4*i + 3] = v.w;
        }
#pragma unroll
        for (int i=0;i<4;i++) {
          res[i] = kern[0]*w[i + LAPLACE_R];
#pragma unroll
          for (int j=1;j<=LAPLACE_R;j++)
            res[i] += kern[j]*(w[i + LAPLACE_R - j] + w[i + LAPLACE_R + j]);
        }
      } else {
#pragma unroll
        for (int i=0;i<4;i++) {
          threadgroup const float *v = vbuf + vrow + hcol[i];
          res[i] = kern[0]*v[0];
#pragma unroll
          for (int j=1;j<=LAPLACE_R;j++)
            res[i] += kern[j]*(v[-j] + v[j]);
        }
      }
      if (scale>0)
        *(threadgroup float4*)&dogs[(scale + 2)%3][hr*FUSED_RW + 4*hg] =  // DoG layer scale-1
          float4(res[0] - prev[0], res[1] - prev[1], res[2] - prev[2], res[3] - prev[3]);
#pragma unroll
      for (int i=0;i<4;i++)
        prev[i] = res[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (scale<3)
      continue;

    // Extrema detection in DoG layer s+1, as FindPointsMultiNew, one SIMD group per MINMAX_H rows
    const int s = scale - 3;
    const int warp = tid/32;
    const int tx = tid%32;
    const int ystart = y0 + MINMAX_H*warp;
    const int yloops = min(height - ystart, MINMAX_H);
    if (warp>=FUSED_FY || yloops<=0)
      continue;
    threadgroup const float *D0 = dogs[s%3];
    threadgroup const float *D1 = dogs[(s + 1)%3];
    threadgroup const float *D2 = dogs[(s + 2)%3];
    const int maxx = min(minx + MINMAX_W, width);
    const int xpos = minx + tx;
    int ptbits = 0;
    for (int y=0;y<yloops;y++) {
      const int r1 = (MINMAX_H*warp + y + 1)*FUSED_RW + tx;
      float d11 = D1[r1];
      if (simd_any(fabs(d11)>thresh)) {
        const int r0 = r1 - FUSED_RW;
        const int r2 = r1 + FUSED_RW;
        float d01 = D0[r1];
        float d10 = D1[r0];
        float d12 = D1[r2];
        float d21 = D2[r1];

        float d00 = D0[r0];
        float d02 = D0[r2];
        float ymin1 = fmin(fmin(d00, d01), d02);
        float ymax1 = fmax(fmax(d00, d01), d02);
        float d20 = D2[r0];
        float d22 = D2[r2];
        float ymin3 = fmin(fmin(d20, d21), d22);
        float ymax3 = fmax(fmax(d20, d21), d22);
        float ymin2 = fmin(fmin(ymin1, fmin(fmin(d10, d12), d11)), ymin3);
        float ymax2 = fmax(fmax(ymax1, fmax(fmax(d10, d12), d11)), ymax3);

        float nmin2 = fmin(simd_shuffle_up(ymin2, 1), simd_shuffle_down(ymin2, 1));
        float nmax2 = fmax(simd_shuffle_up(ymax2, 1), simd_shuffle_down(ymax2, 1));
        float minv = fmin(fmin(nmin2, ymin1), ymin3);
        minv = fmin(fmin(minv, d10), d12);
        float maxv = fmax(fmax(nmax2, ymax1), ymax3);
        maxv = fmax(fmax(maxv, d10), d12);

        if (tx>0 && tx<MINMAX_W+1 && xpos<=maxx)
          ptbits |= ((d11 < fmin(-thresh, minv)) | (d11 > fmax(thresh, maxv))) << y;
      }
    }

    uint totbits = popcount(ptbits);
    uint numbits = totbits;
    for (ushort d=1;d<32;d<<=1) {
      uint num = simd_shuffle_up(totbits, d);
      if (tx >= d)
        totbits += num;
    }
    int pos = totbits - numbits;
    threadgroup ushort *wpoints = points[warp];
    for (int y=0;y<yloops;y++) {
      int ypos = ystart + y;
      if (ptbits & (1 << y) && pos<FUSED_RW) {
        wpoints[2*pos + 0] = xpos - 1;
        wpoints[2*pos + 1] = ypos;
        pos ++;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    totbits = simd_shuffle(totbits, 31);
    if (tx<(int)totbits) {
      int xpos = wpoints[2*tx + 0];
      int ypos = wpoints[2*tx + 1];
      int ptr = (ypos - y0 + 1)*FUSED_RW + xpos - (minx - 1);
      const int pitch = FUSED_RW;
      float val = D1[ptr];
      threadgroup const float *data1 = &D1[ptr];
      // Negated Hessian (dxx = -d2/dx2, dxy = -d2/dxdy, ...): the offset is its inverse times
      // the gradient
      float dxx = 2.0f*val - data1[-1] - data1[1];
      float dyy = 2.0f*val - data1[-pitch] - data1[pitch];
      float dxy = 0.25f*(data1[-pitch+1] + data1[+pitch-1] - data1[+pitch+1] - data1[-pitch-1]);
      float tra = dxx + dyy;
      float det = dxx*dyy - dxy*dxy;
      if (tra*tra<p.edgeLimit*det) {
        float edge = tra*tra/det;
        float dx = 0.5f*(data1[1] - data1[-1]);
        float dy = 0.5f*(data1[pitch] - data1[-pitch]);
        threadgroup const float *data0 = &D0[ptr];
        threadgroup const float *data2 = &D2[ptr];
        float ds = 0.5f*(data2[0] - data0[0]);
        float dss = 2.0f*val - data2[0] - data0[0];
        float dxs = 0.25f*(data0[1] + data2[-1] - data2[1] - data0[-1]);
        float dys = 0.25f*(data0[pitch] + data2[-pitch] - data2[pitch] - data0[-pitch]);
        float idxx = dyy*dss - dys*dys;
        float idxy = dys*dxs - dxy*dss;
        float idxs = dxy*dys - dyy*dxs;
        float idet = 1.0f/(idxx*dxx + idxy*dxy + idxs*dxs);
        float idyy = dxx*dss - dxs*dxs;
        float idys = dxy*dxs - dxx*dys;
        float idss = dxx*dyy - dxy*dxy;
        float pdx = idet*(idxx*dx + idxy*dy + idxs*ds);
        float pdy = idet*(idxy*dx + idyy*dy + idys*ds);
        float pds = idet*(idxs*dx + idys*dy + idss*ds);
        if (pdx<-0.5f || pdx>0.5f || pdy<-0.5f || pdy>0.5f || pds<-0.5f || pds>0.5f) {
          pdx = dx/dxx;
          pdy = dy/dyy;
          pds = ds/dss;
        }
        float dval = 0.5f*(dx*pdx + dy*pdy + ds*pds);
        int maxPts = p.maxPts;
        float sc = pow(2.0f, (float)s/NUM_SCALES) * exp2(pds*p.factor);
        if (sc>=p.lowestScale) {
          uint idx = IncCounter(d_PointCounter, 2*octave+0);   // candidates of this octave
          if (idx<(uint)maxPts) {
            d_Cand[idx].xpos = xpos + pdx;
            d_Cand[idx].ypos = ypos + pdy;
            d_Cand[idx].scale = sc;
            d_Cand[idx].sharpness = val + dval;
            d_Cand[idx].edgeness = edge;
            d_Cand[idx].subsampling = p.subsampling;
          }
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Orientations and descriptors
///////////////////////////////////////////////////////////////////////////////

// Same orientations as ComputeOrientationsCONST, with one SIMD group per point (8 points
// per threadgroup) instead of one threadgroup, and no threadgroup barriers: each lane
// samples 4 of the 121 gradients, then lane b sums bin b over the samples in the same
// order (fetched with simd_shuffle), and smoothing, peak search and interpolation are
// done with shuffles, with the same operations. The results are bit-identical.

kernel void ComputeOrientationsSimd(texture2d<float, access::sample> tex [[texture(0)]], device CandPoint *d_Sift [[buffer(0)]],
                                    constant OctaveParams &p [[buffer(1)]], device atomic_uint *d_PointCounter [[buffer(2)]],
                                    uint bid [[threadgroup_position_in_grid]], uint ngroups [[threadgroups_per_grid]],
                                    uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
  const int octave = p.octave;
  const uint maxPts = p.maxPts;
  const int tx = lane;
  int totPts = min(LoadCounter(d_PointCounter, 2*octave+0), maxPts);   // candidates of this octave
  for (int bx = bid*ORIENT_SG + sgid; bx < totPts; bx += ngroups*ORIENT_SG) {
    const float scale = d_Sift[bx].scale;
    const float i2sigma2 = -1.0f/(2.0f*1.5f*1.5f*scale*scale);
    const float xp = d_Sift[bx].xpos - 4.5f;
    const float yp = d_Sift[bx].ypos - 4.5f;
    float sgrad[4];
    int sbin[4];
#pragma unroll
    for (int j=0;j<4;j++) {
      const int t = 32*j + tx;          // sample index, as tx in ComputeOrientationsCONST
      int yd = t/11;
      int xd = t - yd*11;
      sgrad[j] = 0.0f;
      sbin[j] = -1;
      if (yd<11) {
        float xf = xp + xd;
        float yf = yp + yd;
        float dx = tex.sample(texSampler, float2(xf+1.0f, yf)).x - tex.sample(texSampler, float2(xf-1.0f, yf)).x;
        float dy = tex.sample(texSampler, float2(xf, yf+1.0f)).x - tex.sample(texSampler, float2(xf, yf-1.0f)).x;
        int bin = 16.0f*atan2(dy, dx)/3.1416f + 16.5f;
        if (bin>31)
          bin = 0;
        float grad = sqrt(dx*dx + dy*dy);
        float gx = exp(i2sigma2*(xd-5)*(xd-5));
        float gy = exp(i2sigma2*(yd-5)*(yd-5));
        sgrad[j] = grad*gx*gy;
        sbin[j] = bin;
      }
    }
    // Histogram, lane tx = bin tx, summed over samples 0..120 in order
    float hist = 0.0f;
#pragma unroll
    for (int j=0;j<4;j++)
      for (ushort l=0;l<32;l++) {
        float g = simd_shuffle(sgrad[j], l);
        int b = simd_shuffle(sbin[j], l);
        if (32*j + l<11*11)
          hist += (b==tx ? g : 0.0f);
      }
    const ushort x1m = (tx>=1 ? tx-1 : tx+31);
    const ushort x1p = (tx<=30 ? tx+1 : tx-31);
    const ushort x2m = (tx>=2 ? tx-2 : tx+30);
    const ushort x2p = (tx<=29 ? tx+2 : tx-30);
    float smooth = 6.0f*hist + 4.0f*(simd_shuffle(hist, x1m) + simd_shuffle(hist, x1p)) +
      (simd_shuffle(hist, x2m) + simd_shuffle(hist, x2p));
    float peakv = (smooth>simd_shuffle(smooth, x1m) && smooth>=simd_shuffle(smooth, x1p) ? smooth : 0.0f);
    float maxval1 = 0.0;
    float maxval2 = 0.0;
    int i1 = -1;
    int i2 = -1;
    for (ushort i=0;i<32;i++) {
      float v = simd_shuffle(peakv, i);
      if (v>maxval1) {
        maxval2 = maxval1;
        maxval1 = v;
        i2 = i1;
        i1 = i;
      } else if (v>maxval2) {
        maxval2 = v;
        i2 = i;
      }
    }
    float val1 = simd_shuffle(smooth, (ushort)((i1+1)&31));
    float val2 = simd_shuffle(smooth, (ushort)((i1+31)&31));
    float val3 = simd_shuffle(smooth, (ushort)((i2+1)&31));
    float val4 = simd_shuffle(smooth, (ushort)((i2+31)&31));
    if (tx==0) {
      float peak = i1 + 0.5f*(val1-val2) / (2.0f*maxval1-val1-val2);
      d_Sift[bx].orientation = 11.25f*(peak<0.0f ? peak+32.0f : peak);
      MaxCounter(d_PointCounter, 2*octave+1, 2*octave+0);
      if (maxval2>0.8f*maxval1) {
        float peak = i2 + 0.5f*(val3-val4) / (2.0f*maxval2-val3-val4);
        uint idx = IncCounter(d_PointCounter, 2*octave+1);
        if (idx<maxPts) {
          d_Sift[idx].xpos = d_Sift[bx].xpos;
          d_Sift[idx].ypos = d_Sift[bx].ypos;
          d_Sift[idx].scale = d_Sift[bx].scale;
          d_Sift[idx].sharpness = d_Sift[bx].sharpness;
          d_Sift[idx].edgeness = d_Sift[bx].edgeness;
          d_Sift[idx].orientation = 11.25f*(peak<0.0f ? peak+32.0f : peak);
          d_Sift[idx].subsampling = d_Sift[bx].subsampling;
        }
      }
    }
  }
}

static inline float FastAtan2(float y, float x)
{
  float absx = fabs(x);
  float absy = fabs(y);
  float maxxy = max(absx, absy);
  float a = (maxxy>0.0f ? precise::divide(min(absx, absy), maxxy) : 0.0f);   // 0 for a zero gradient, not NaN
  float s = a*a;
  float r = ((-0.0464964749f*s + 0.15931422f)*s - 0.327622764f)*s*a + a;
  r = (absy>absx ? 1.57079637f - r : r);
  r = (x<0 ? 3.14159274f - r : r);
  r = (y<0 ? -r : r);
  return r;
}

// Same descriptor as ExtractSiftDescriptorsCONSTNew in cudaSiftD.cu, without atomics:
// each thread first computes the 4 spatially weighted gradients of its 2 samples (same
// operations). Then the 8 threads of a spatial cell each take one row of the cell's 8x8
// samples, accumulate the 8 angle bins in registers, and sum the 8 partial histograms
// with simd_shuffle_xor, in a fixed order. The result is deterministic.
kernel void ExtractSiftDescriptorsGather(texture2d<float, access::sample> tex [[texture(0)]], device SiftPoint *d_sift [[buffer(0)]],
                                         constant OctaveParams &p [[buffer(1)]], device atomic_uint *d_PointCounter [[buffer(2)]],
                                         device const CandPoint *d_Cand [[buffer(3)]],
                                         uint bid [[threadgroup_position_in_grid]], uint ngroups [[threadgroups_per_grid]],
                                         uint ltid [[thread_index_in_threadgroup]])
{
  threadgroup float gauss[16];
  threadgroup float sgrad[4][256];   // [2*dv + dh][16*y + x]: weights for bins (veri+dv, hori+dh)
  threadgroup float sangle[256];     // angle bin, integer part angi and fractional part angf
  threadgroup float sums[4];

  const int tx = ltid%16; // 0 -> 16
  const int ty = ltid/16; // 0 -> 8
  const int idx = ty*16 + tx;
  const int octave = p.octave;
  const uint maxPts = p.maxPts;
  const float subsampling = p.subsampling;
  if (ty==0)
    gauss[tx] = exp(-(tx-7.5f)*(tx-7.5f)/128.0f);
  // Histogram bin of this thread, as hist = 8*(4*veri + hori) + angle
  const int bv = idx/32;
  const int bh = (idx/8)%4;
  const int ba = idx%8;

  // The points of this octave go after those of the smaller octaves (1..octave-1)
  uint first = 0;
  for (int o=1;o<octave;o++)
    first += min(LoadCounter(d_PointCounter, 2*o+1), maxPts);
  int totPts = min(LoadCounter(d_PointCounter, 2*octave+1), maxPts);
  for (int bx = bid; bx < totPts; bx += ngroups) {
    const uint out = first + bx;   // >= maxPts: computed, but not written
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute angles and gradients
    float theta = 2.0f*3.1415f/360.0f*d_Cand[bx].orientation;
    float sina = sin(theta);           // cosa -sina
    float cosa = cos(theta);           // sina  cosa
    float scale = 12.0f/16.0f*d_Cand[bx].scale;
    float ssina = scale*sina;
    float scosa = scale*cosa;

#pragma unroll
    for (int y=ty;y<16;y+=8) {
      float xpos = d_Cand[bx].xpos + (tx-7.5f)*scosa - (y-7.5f)*ssina + 0.5f;
      float ypos = d_Cand[bx].ypos + (tx-7.5f)*ssina + (y-7.5f)*scosa + 0.5f;
      float dx = tex.sample(texSampler, float2(xpos+cosa, ypos+sina)).x -
        tex.sample(texSampler, float2(xpos-cosa, ypos-sina)).x;
      float dy = tex.sample(texSampler, float2(xpos-sina, ypos+cosa)).x -
        tex.sample(texSampler, float2(xpos+sina, ypos-cosa)).x;
      float grad = gauss[y]*gauss[tx] * precise::sqrt(dx*dx + dy*dy);
      float angf = 4.0f/M_PI_F*FastAtan2(dy, dx) + 4.0f;
      angf = (angf>=8.0f ? angf - 8.0f : angf);   // atan2 = pi is angle 0, not 8

      int hori = (tx + 2)/4 - 1;      // Convert from (tx,y,angle) to bins
      float horf = (tx - 1.5f)/4.0f - hori;
      float ihorf = 1.0f - horf;
      int veri = (y + 2)/4 - 1;
      float verf = (y - 1.5f)/4.0f - veri;
      float iverf = 1.0f - verf;

      const int s = 16*y + tx;
      float grad1 = ihorf*grad;
      sgrad[0][s] = iverf*grad1;    // Upper left
      sgrad[2][s] = verf*grad1;     // Lower left
      grad1 = horf*grad;
      sgrad[1][s] = iverf*grad1;    // Upper right
      sgrad[3][s] = verf*grad1;     // Lower right
      sangle[s] = angf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Samples with hori==bh (x in 4*bh+2..4*bh+5) contribute with dh=0, samples with
    // hori==bh-1 (x in 4*bh-2..4*bh+1) with dh=1. Same for rows. The 8 threads of a cell
    // (bv,bh) take one row y = 4*bv-2+ba of these 8x8 samples each, accumulate all 8 angle
    // bins in registers, and the 8 partial histograms are summed with SIMD shuffles.
    float bins[8];
#pragma unroll
    for (int a=0;a<8;a++)
      bins[a] = 0.0f;
    {
      const int y = 4*bv - 2 + ba;
      if (y>=0 && y<16) {
        const int dv = (y<4*bv+2 ? 2 : 0);
        for (int x=max(4*bh-2, 0);x<min(4*bh+6, 16);x++) {
          const int s = 16*y + x;
          const float grad2 = sgrad[dv + (x<4*bh+2 ? 1 : 0)][s];
          const float angle = sangle[s];
          const int angi = angle;
          const float angf = angle - angi;
          const int angp = (angi<7 ? angi+1 : 0);
          const float g1 = (1.0f - angf)*grad2;
          const float g2 = angf*grad2;
#pragma unroll
          for (int a=0;a<8;a++)
            bins[a] += (angi==a ? g1 : 0.0f) + (angp==a ? g2 : 0.0f);
        }
      }
    }
#pragma unroll
    for (ushort d=1;d<8;d<<=1)
#pragma unroll
      for (int a=0;a<8;a++)
        bins[a] += simd_shuffle_xor(bins[a], d);
    float bin = bins[0];
#pragma unroll
    for (int a=1;a<8;a++)
      bin = (ba==a ? bins[a] : bin);

    // Normalize twice and suppress peaks first time
    float sum = bin*bin;
    for (ushort i=16;i>0;i/=2)
      sum += simd_shuffle_down(sum, i);
    if ((idx&31)==0)
      sums[idx/32] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tsum1 = sums[0] + sums[1] + sums[2] + sums[3];
    tsum1 = min(bin * rsqrt(tsum1), 0.2f);

    sum = tsum1*tsum1;
    for (ushort i=16;i>0;i/=2)
      sum += simd_shuffle_down(sum, i);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((idx&31)==0)
      sums[idx/32] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float tsum2 = sums[0] + sums[1] + sums[2] + sums[3];
    if (out<maxPts) {
      d_sift[out].data[idx] = tsum1 * rsqrt(tsum2);
      if (idx==0) {
        const CandPoint c = d_Cand[bx];
        d_sift[out].xpos = c.xpos*subsampling;
        d_sift[out].ypos = c.ypos*subsampling;
        d_sift[out].scale = c.scale*subsampling;
        d_sift[out].sharpness = c.sharpness;
        d_sift[out].edgeness = c.edgeness;
        d_sift[out].orientation = c.orientation;
        d_sift[out].subsampling = c.subsampling;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Helpers to keep point counts on the GPU, so that ExtractSift and MatchSiftData calls
// can be recorded in one batch (cudaSiftBatch.h)
kernel void ClearCounters(device atomic_uint *d_PointCounter [[buffer(0)]], uint i [[thread_position_in_grid]])
{
  if (i<NUM_COUNTERS)
    atomic_store_explicit(&d_PointCounter[i], 0u, memory_order_relaxed);
}

// Number of points of ExtractSift: sum of the octaves' points (at most maxPts each), at most maxPts
kernel void CountPoints(device atomic_uint *d_PointCounter [[buffer(0)]], constant int2 &octMax [[buffer(1)]],
                        device int *numPts [[buffer(2)]], uint i [[thread_position_in_grid]])
{
  if (i==0) {
    uint n = 0;
    for (int o=1;o<=octMax.x;o++)
      n += min(LoadCounter(d_PointCounter, 2*o+1), (uint)octMax.y);
    *numPts = min(n, (uint)octMax.y);
  }
}

// Threadgroup counts of FindMaxCorr12 and MergeMaxCorr12, for indirect dispatches
// No matching if a point count is 0, as outside a batch (the match fields are unchanged)
kernel void MatchGrid(constant int &n1 [[buffer(0)]], device uint *args [[buffer(1)]], constant int &numChunks [[buffer(2)]],
                      constant int &n2 [[buffer(3)]], uint i [[thread_position_in_grid]])
{
  if (i==0) {
    const int n = (n2>0 ? n1 : 0);
    uint g = (n + M12_T - 1)/M12_T;
    args[0] = g;  args[1] = (g ? numChunks : 0);  args[2] = 1;          // FindMaxCorr12
    args[3] = (n + MERGE_NT - 1)/MERGE_NT;  args[4] = 1;  args[5] = 1;   // MergeMaxCorr12
  }
}

kernel void RescalePositions(device SiftPoint *d_sift [[buffer(0)]], constant float &scale [[buffer(1)]],
                             constant int &numPts [[buffer(2)]], uint num [[thread_position_in_grid]])
{
  if ((int)num<numPts) {
    d_sift[num].xpos *= scale;
    d_sift[num].ypos *= scale;
    d_sift[num].scale *= scale;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Matching, see MatchSiftData in matching.cu
///////////////////////////////////////////////////////////////////////////////

// Brute-force matching with the outputs of FindMaxCorr11 in matching.cu (best score,
// ambiguity = second best / best, matched point), using SIMD group matrices. Each
// threadgroup correlates 64 sift1 points with the sift2 tiles of one of numChunks
// chunks, 64 points at a time: 8 SIMD groups compute 16x32 correlations each, as 2x4
// 8x8 matrices (more accumulators spill registers and run an order of magnitude
// slower), then each row is reduced to its best and second best score. With a fixed
// number of chunks, the scratch memory does not depend on the number of sift2 points.
// MergeMaxCorr12 merges the per-chunk results; each sift2 point is in exactly one
// chunk, so the result is exact. Out-of-range points are zero padded (zero descriptors
// score 0 and can never become best or second best), and a point without any positive
// score gets match -1.
#define M12_K   32            // descriptor dimensions per threadgroup memory stage
#define M12_P   (M12_K + 4)   // padded stage row length, keeps float4 alignment

kernel void FindMaxCorr12(device const SiftPoint *sift1 [[buffer(0)]], device const SiftPoint *sift2 [[buffer(1)]],
                          constant MatchParams &p [[buffer(2)]], device float4 *d_best [[buffer(3)]],
                          constant int &n1 [[buffer(4)]], constant int &n2 [[buffer(5)]],
                          uint2 gbid [[threadgroup_position_in_grid]], uint ltid [[thread_index_in_threadgroup]],
                          uint sgid [[simdgroup_index_in_threadgroup]])
{
  threadgroup float4 smem4[2*M12_T*M12_P/4];   // A and B stages, then the 64x64 correlations
  threadgroup float *As = (threadgroup float*)smem4;
  threadgroup float *Bs = As + M12_T*M12_P;
  const int numPts1 = n1, numPts2 = n2;   // point counts, possibly computed on the GPU
  const int tiles2 = (numPts2 + M12_T - 1)/M12_T;
  const int bp1 = gbid.x*M12_T;
  const int chunk = gbid.y;     // sift2 tiles chunk*perChunk ... (chunk+1)*perChunk-1
  const int perChunk = (tiles2 + p.numChunks - 1)/p.numChunks;
  const int r0 = (sgid/2)*16;   // rows (sift1) and columns (sift2) of this SIMD group
  const int c0 = (sgid%2)*32;
  const float4 zero = float4(0.0f);
  const int row = ltid/4;       // reduction: four threads per row, 16 columns each
  const int part = ltid%4;
  float max_score = 0.0f, sec_score = 0.0f;
  int index = -1;
  for (int t2=chunk*perChunk;t2<min((chunk + 1)*perChunk, tiles2);t2++) {
    const int bp2 = t2*M12_T;

    simdgroup_float8x8 acc[2][4];
#pragma unroll
    for (int i=0;i<2;i++)
#pragma unroll
      for (int j=0;j<4;j++)
        acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k0=0;k0<NDIM;k0+=M12_K) {
#pragma unroll
      for (int r=0;r<M12_T*M12_K/4/M12_NT;r++) {
        int idx = ltid + M12_NT*r;  // 64 points x M12_K/4 float4
        int q = idx/(M12_K/4);
        int f = idx%(M12_K/4);
        // Zero padding: zero descriptors score 0 and can never become best or second best
        float4 a = (bp1 + q < numPts1 ? ((device const float4*)sift1[bp1 + q].data)[k0/4 + f] : zero);
        float4 b = (bp2 + q < numPts2 ? ((device const float4*)sift2[bp2 + q].data)[k0/4 + f] : zero);
        *(threadgroup float4*)&As[q*M12_P + 4*f] = a;
        *(threadgroup float4*)&Bs[q*M12_P + 4*f] = b;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
      for (int k=0;k<M12_K;k+=8) {
        simdgroup_float8x8 a[2], b[4];
#pragma unroll
        for (int i=0;i<2;i++)
          simdgroup_load(a[i], &As[(r0 + 8*i)*M12_P + k], M12_P);
#pragma unroll
        for (int j=0;j<4;j++)
          simdgroup_load(b[j], &Bs[(c0 + 8*j)*M12_P + k], M12_P, ulong2(0, 0), true);
#pragma unroll
        for (int i=0;i<2;i++)
#pragma unroll
          for (int j=0;j<4;j++)
            simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Correlations to threadgroup memory, then best and second best per row
    threadgroup float *corr = As;   // M12_T x (M12_T + 4) floats, fits in smem
    const int cp = M12_T + 4;
#pragma unroll
    for (int i=0;i<2;i++)
#pragma unroll
      for (int j=0;j<4;j++)
        simdgroup_store(acc[i][j], &corr[(r0 + 8*i)*cp + c0 + 8*j], cp);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int c=16*part;c<16*part+16;c++) {
      float v = corr[row*cp + c];
      if (v>max_score) {
        sec_score = max_score;
        max_score = v;
        index = bp2 + c;
      } else if (v>sec_score)
        sec_score = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);   // corr aliases the next tile's stages
  }
  for (ushort d=1;d<4;d<<=1) {
    float m = simd_shuffle_xor(max_score, d);
    float s = simd_shuffle_xor(sec_score, d);
    int ix = simd_shuffle_xor(index, d);
    if (m>max_score || (m==max_score && ix>=0 && (index<0 || ix<index))) {  // same result in all 4 lanes
      sec_score = fmax(max_score, s);
      max_score = m;
      index = ix;
    } else
      sec_score = fmax(sec_score, m);
  }
  if (part==0 && bp1 + row<numPts1)
    d_best[chunk*numPts1 + bp1 + row] = float4(max_score, sec_score, as_type<float>(index), 0.0f);
}

kernel void MergeMaxCorr12(device SiftPoint *sift1 [[buffer(0)]], device const SiftPoint *sift2 [[buffer(1)]],
                           constant MatchParams &p [[buffer(2)]], device const float4 *d_best [[buffer(3)]],
                           constant int &n1 [[buffer(4)]], constant int &n2 [[buffer(5)]],
                           uint gid [[thread_position_in_grid]])
{
  const int p1 = gid;
  const int numPts1 = n1;
  if (p1>=numPts1)
    return;
  float max_score = 0.0f, sec_score = 0.0f;
  int index = -1;
  for (int t=0;t<p.numChunks;t++) {
    float4 b = d_best[t*numPts1 + p1];
    if (b.x>max_score) {
      sec_score = fmax(max_score, b.y);
      max_score = b.x;
      index = as_type<int>(b.z);
    } else
      sec_score = fmax(sec_score, b.x);
  }
  sift1[p1].score = max_score;
  sift1[p1].ambiguity = sec_score / (max_score + 1e-6f);
  if (index>=0 && index<n2) {
    sift1[p1].match = index;
    sift1[p1].match_xpos = sift2[index].xpos;
    sift1[p1].match_ypos = sift2[index].ypos;
  } else {
    sift1[p1].match = -1;
    sift1[p1].match_xpos = 0.0f;
    sift1[p1].match_ypos = 0.0f;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Homography estimation, see ComputeHomographies and TestHomographies in matching.cu
///////////////////////////////////////////////////////////////////////////////

#define HOMO_SIZE 8

static void InvertMatrix(thread float (&elem)[HOMO_SIZE][HOMO_SIZE], thread float (&res)[HOMO_SIZE][HOMO_SIZE])
{
  const int size = HOMO_SIZE;
  int indx[size];
  float b[size];
  float vv[size];
  for (int i=0;i<size;i++)
    indx[i] = 0;
  int imax = 0;
  float d = 1.0;
  for (int i=0;i<size;i++) { // find biggest element for each row
    float big = 0.0;
    for (int j=0;j<size;j++) {
      float temp = fabs(elem[i][j]);
      if (temp>big)
        big = temp;
    }
    if (big>0.0)
      vv[i] = 1.0/big;
    else
      vv[i] = 1e16;
  }
  for (int j=0;j<size;j++) {
    for (int i=0;i<j;i++) { // i<j
      float sum = elem[i][j]; // i<j (lower left)
      for (int k=0;k<i;k++) // k<i<j
        sum -= elem[i][k]*elem[k][j]; // i>k (upper right), k<j (lower left)
      elem[i][j] = sum; // i<j (lower left)
    }
    float big = 0.0;
    for (int i=j;i<size;i++) { // i>=j
      float sum = elem[i][j]; // i>=j (upper right)
      for (int k=0;k<j;k++) // k<j<=i
        sum -= elem[i][k]*elem[k][j]; // i>k (upper right), k<j (lower left)
      elem[i][j] = sum; // i>=j (upper right)
      float dum = vv[i]*fabs(sum);
      if (dum>=big) {
        big = dum;
        imax = i;
      }
    }
    if (j!=imax) { // imax>j
      for (int k=0;k<size;k++) {
        float dum = elem[imax][k]; // upper right and lower left
        elem[imax][k] = elem[j][k];
        elem[j][k] = dum;
      }
      d = -d;
      vv[imax] = vv[j];
    }
    indx[j] = imax;
    if (elem[j][j]==0.0)  // j==j (upper right)
      elem[j][j] = 1e-16;
    if (j!=(size-1)) {
      float dum = 1.0/elem[j][j];
      for (int i=j+1;i<size;i++) // i>j
        elem[i][j] *= dum; // i>j (upper right)
    }
  }
  for (int j=0;j<size;j++) {
    for (int k=0;k<size;k++)
      b[k] = 0.0;
    b[j] = 1.0;
    int ii = -1;
    for (int i=0;i<size;i++) {
      int ip = indx[i];
      float sum = b[ip];
      b[ip] = b[i];
      if (ii!=-1)
        for (int j=ii;j<i;j++)
          sum -= elem[i][j]*b[j]; // i>j (upper right)
      else if (sum!=0.0)
        ii = i;
      b[i] = sum;
    }
    for (int i=size-1;i>=0;i--) {
      float sum = b[i];
      for (int j=i+1;j<size;j++)
        sum -= elem[i][j]*b[j]; // i<j (lower left)
      b[i] = sum/elem[i][i]; // i==i (upper right)
    }
    for (int i=0;i<size;i++)
      res[i][j] = b[i];
  }
}

kernel void ComputeHomographies(device const float *coord [[buffer(0)]], device const int *randPts [[buffer(1)]],
                                device float *homo [[buffer(2)]], constant HomographyParams &p [[buffer(3)]],
                                uint gid [[thread_position_in_grid]])
{
  float a[8][8], ia[8][8];
  float b[8];
  const int idx = gid;
  const int numLoops = p.numLoops;
  const int numPts = p.numPts;
  for (int i=0;i<4;i++) {
    int pt = randPts[i*numLoops+idx];
    float x1 = coord[pt+0*numPts];
    float y1 = coord[pt+1*numPts];
    float x2 = coord[pt+2*numPts];
    float y2 = coord[pt+3*numPts];
    thread float *row1 = a[2*i+0];
    row1[0] = x1;
    row1[1] = y1;
    row1[2] = 1.0;
    row1[3] = row1[4] = row1[5] = 0.0;
    row1[6] = -x2*x1;
    row1[7] = -x2*y1;
    thread float *row2 = a[2*i+1];
    row2[0] = row2[1] = row2[2] = 0.0;
    row2[3] = x1;
    row2[4] = y1;
    row2[5] = 1.0;
    row2[6] = -y2*x1;
    row2[7] = -y2*y1;
    b[2*i+0] = x2;
    b[2*i+1] = y2;
  }
  InvertMatrix(a, ia);
  for (int j=0;j<8;j++) {
    float sum = 0.0f;
    for (int i=0;i<8;i++)
      sum += ia[j][i]*b[i];
    homo[j*numLoops+idx] = sum;
  }
}


// CUDA uses round-toward-zero multiplications (__fmul_rz) here; Metal has no equivalent,
// so inlier counts can differ marginally for points at the threshold.
kernel void TestHomographies(device const float *d_coord [[buffer(0)]], device const float *d_homo [[buffer(1)]],
                             device int *d_counts [[buffer(2)]], constant HomographyParams &p [[buffer(3)]],
                             uint2 bid [[threadgroup_position_in_grid]], uint2 ngroups [[threadgroups_per_grid]],
                             uint2 tid [[thread_position_in_threadgroup]])
{
  threadgroup float homo[8*TESTHOMO_LOOPS];
  threadgroup int cnts[TESTHOMO_TESTS*TESTHOMO_LOOPS];
  const int tx = tid.x;
  const int ty = tid.y;
  const int numPts = p.numPts;
  const int idx = bid.y*TESTHOMO_LOOPS + tx;
  const int numLoops = TESTHOMO_LOOPS*ngroups.y;
  if (ty<8 && tx<TESTHOMO_LOOPS)
    homo[tx*8+ty] = d_homo[idx+ty*numLoops];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float a[8];
#pragma unroll
  for (int i=0;i<8;i++)
    a[i] = homo[ty*8+i];
  int cnt = 0;
  for (int i=tx;i<p.count;i+=TESTHOMO_TESTS) {   // the actual points, not the padding
    float x1 = d_coord[i+0*numPts];
    float y1 = d_coord[i+1*numPts];
    float x2 = d_coord[i+2*numPts];
    float y2 = d_coord[i+3*numPts];
    float nomx = a[0]*x1 + a[1]*y1 + a[2];
    float nomy = a[3]*x1 + a[4]*y1 + a[5];
    float deno = a[6]*x1 + a[7]*y1 + 1.0f;
    float errx = x2*deno - nomx;
    float erry = y2*deno - nomy;
    float err2 = errx*errx + erry*erry;
    if (err2<p.thresh2*(deno*deno))
      cnt ++;
  }
  int kty = TESTHOMO_TESTS*ty;
  cnts[kty + tx] = cnt;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int len = TESTHOMO_TESTS/2;
  while (len>0) {
    if (tx<len)
      cnts[kty + tx] += cnts[kty + tx + len];
    len /= 2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tx<TESTHOMO_LOOPS && ty==0)
    d_counts[idx] = cnts[TESTHOMO_TESTS*tx];
}

///////////////////////////////////////////////////////////////////////////////
// Seeded homography estimation on the GPU (FindHomographyGPU, ImproveHomographyGPU).
// cudaSiftRandom.h is prepended to this file.
///////////////////////////////////////////////////////////////////////////////

// Coordinates of all points, and indices of the valid points (score > minScore and
// ambiguity < maxAmbiguity) in increasing order, as in FindHomography. One threadgroup.
kernel void HomographyPrepare(device const SiftPoint *sift [[buffer(0)]], constant int &numPts [[buffer(1)]],
                              device float *coord [[buffer(2)]], device int *valid [[buffer(3)]],
                              device int *numValid [[buffer(4)]], constant HomographyGPUParams &p [[buffer(5)]],
                              uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                              uint ntid [[threads_per_threadgroup]])
{
  threadgroup int sgCount[32];
  threadgroup int sgFirst[32];
  int base = 0;
  const int S = p.stride;
  for (int c0=0;c0<numPts;c0+=ntid) {
    const int i = c0 + tid;
    bool ok = false;
    if (i<numPts) {
      coord[i + 0*S] = sift[i].xpos;
      coord[i + 1*S] = sift[i].ypos;
      coord[i + 2*S] = sift[i].match_xpos;
      coord[i + 3*S] = sift[i].match_ypos;
      ok = (sift[i].score>p.minScore && sift[i].ambiguity<p.maxAmbiguity);
    }
    const int pre = simd_prefix_exclusive_sum((int)ok);
    if (lane==31)
      sgCount[sg] = pre + (int)ok;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg==0) {
      int c = (lane<nsg ? sgCount[lane] : 0);
      sgFirst[lane] = simd_prefix_exclusive_sum(c);
      if (lane==31)
        sgCount[0] = sgFirst[31] + c;   // total of this chunk
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ok)
      valid[base + sgFirst[sg] + pre] = i;
    base += sgCount[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid==0)
    *numValid = base;
}

// The coordinates of the valid points, in the order of the points (valid holds their
// indices in increasing order): RANSAC samples and counts the inliers among these only. The
// samples depend on the order of the points, as those of FindHomographySeeded; SortSiftData
// makes it reproducible.
kernel void HomographyGather(device const float *coord [[buffer(0)]], device const int *valid [[buffer(1)]],
                             device const int *numValid [[buffer(2)]], device float *validCoord [[buffer(3)]],
                             constant HomographyGPUParams &p [[buffer(4)]], uint i [[thread_position_in_grid]])
{
  if ((int)i>=*numValid)
    return;
  const int S = p.stride;
  const int a = valid[i];
  validCoord[i] = coord[a];
  validCoord[i + S] = coord[a + S];
  validCoord[i + 2*S] = coord[a + 2*S];
  validCoord[i + 3*S] = coord[a + 3*S];
}

// Four random valid points per hypothesis, from the seeded generator, and whether they
// pass SiftSameOrientation (in float: Metal has no double precision, so for nearly
// collinear samples the result can differ from that of the host code)
kernel void HomographySample(device const float *validCoord [[buffer(0)]], device const int *numValid [[buffer(1)]],
                             device int *randPts [[buffer(2)]], constant HomographyGPUParams &p [[buffer(3)]],
                             device int *sameOrientation [[buffer(4)]], uint i [[thread_position_in_grid]])
{
  if ((int)i>=p.numLoops)
    return;
  const int nv = *numValid;
  const int S = p.stride;
  int q[4] = {0, 0, 0, 0};
  if (nv>=8)
    SiftRandomSample4(p.seed, i, nv, q);
  float x1[4], y1[4], x2[4], y2[4];
  for (int k=0;k<4;k++) {
    randPts[i + k*p.numLoops] = q[k];
    x1[k] = validCoord[q[k]];
    y1[k] = validCoord[q[k] + S];
    x2[k] = validCoord[q[k] + 2*S];
    y2[k] = validCoord[q[k] + 3*S];
  }
  sameOrientation[i] = (nv>=8 && SiftSameOrientation(x1, y1, x2, y2));
}

// As TestHomographies, over the *numValid valid points (the count is read on the GPU),
// and with 4 hypotheses per thread (TESTHOMO_PER hypotheses per row of threads), so that
// each point is loaded once for 4 hypotheses. 16 x 16 threads: 64 hypotheses per
// threadgroup, the 16 threads of a row take every 16th point.
kernel void TestHomographiesGPU(device const float *d_coord [[buffer(0)]], device const float *d_homo [[buffer(1)]],
                                device int *d_counts [[buffer(2)]], constant HomographyGPUParams &p [[buffer(3)]],
                                device const int *numValid [[buffer(4)]],
                                uint2 bid [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]])
{
  const int numPts = *numValid;
  const int tx = tid.x;
  const int ty = tid.y;
  const int S = p.stride;
  const int numLoops = p.numLoops;
  const int h0 = (bid.y*16 + ty)*TESTHOMO_PER;   // first hypothesis of this thread
  float a[TESTHOMO_PER][8];
  for (int h=0;h<TESTHOMO_PER;h++)
    for (int i=0;i<8;i++)
      a[h][i] = (h0 + h<numLoops ? d_homo[h0 + h + i*numLoops] : 0.0f);
  int cnt[TESTHOMO_PER] = {0};
  const float thresh2 = p.thresh2;
  for (int i=tx;i<numPts;i+=16) {
    float x1 = d_coord[i+0*S];
    float y1 = d_coord[i+1*S];
    float x2 = d_coord[i+2*S];
    float y2 = d_coord[i+3*S];
#pragma unroll
    for (int h=0;h<TESTHOMO_PER;h++) {
      float nomx = a[h][0]*x1 + a[h][1]*y1 + a[h][2];
      float nomy = a[h][3]*x1 + a[h][4]*y1 + a[h][5];
      float deno = a[h][6]*x1 + a[h][7]*y1 + 1.0f;
      float errx = x2*deno - nomx;
      float erry = y2*deno - nomy;
      float err2 = errx*errx + erry*erry;
      if (err2<thresh2*(deno*deno))
        cnt[h] ++;
    }
  }
  // Sum over the 16 threads of the row (half a SIMD group)
#pragma unroll
  for (int h=0;h<TESTHOMO_PER;h++)
    for (ushort d=1;d<16;d<<=1)
      cnt[h] += simd_shuffle_xor(cnt[h], d);
  if (tx<TESTHOMO_PER && h0 + tx<numLoops)
    d_counts[h0 + tx] = cnt[tx];
}

// Hypothesis with the most inliers among those from samples that pass SiftSameOrientation
// (the first one in case of ties, as FindHomography). Identity and 0 matches if there are
// fewer than 8 valid points or no sample passes. One threadgroup.
kernel void HomographyBest(device const int *d_counts [[buffer(0)]], device const float *d_homo [[buffer(1)]],
                           device const int *numValid [[buffer(2)]], device HomographyResult *res [[buffer(3)]],
                           constant HomographyGPUParams &p [[buffer(4)]], device const int *sameOrientation [[buffer(5)]],
                           uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                           uint ntid [[threads_per_threadgroup]])
{
  threadgroup int sgMax[32];
  threadgroup int sgIdx[32];
  int maxCount = -1, maxIndex = -1;
  for (int i=tid;i<p.numLoops;i+=ntid)
    if (sameOrientation[i] && d_counts[i]>maxCount) {
      maxCount = d_counts[i];
      maxIndex = i;
    }
  for (ushort d=16;d>0;d/=2) {
    int c = simd_shuffle_down(maxCount, d), j = simd_shuffle_down(maxIndex, d);
    if (lane+d<32 && (c>maxCount || (c==maxCount && j>=0 && (maxIndex<0 || j<maxIndex)))) {
      maxCount = c;
      maxIndex = j;
    }
  }
  if (lane==0) {
    sgMax[sg] = maxCount;
    sgIdx[sg] = maxIndex;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid==0) {
    for (uint s=1;s<nsg;s++)
      if (sgMax[s]>maxCount || (sgMax[s]==maxCount && sgIdx[s]>=0 && (maxIndex<0 || sgIdx[s]<maxIndex))) {
        maxCount = sgMax[s];
        maxIndex = sgIdx[s];
      }
    if (*numValid<8 || maxIndex<0) {
      for (int j=0;j<9;j++)
        res->h[j] = (j%4==0 ? 1.0f : 0.0f);
      res->count = 0;
    } else {
      for (int j=0;j<8;j++)
        res->h[j] = d_homo[maxIndex + j*p.numLoops];
      res->h[8] = 1.0f;
      res->count = maxCount;
    }
  }
}

// Double-float arithmetic (about 48 bits of mantissa): Metal has no double precision.
// Must be compiled without fast math (this kernel uses a separate library compiled with
// MTLMathModeSafe), which would break the error-free transformations.
struct df { float hi, lo; };
static inline df DF(float a) { return df{a, 0.0f}; }
static inline df QuickTwoSum(float a, float b) { float s = a + b; return df{s, b - (s - a)}; }
static inline df TwoSum(float a, float b) { float s = a + b, bb = s - a; return df{s, (a - (s - bb)) + (b - bb)}; }
static inline df operator+(df a, df b) {
  df s = TwoSum(a.hi, b.hi), t = TwoSum(a.lo, b.lo);
  s.lo += t.hi;
  s = QuickTwoSum(s.hi, s.lo);
  s.lo += t.lo;
  return QuickTwoSum(s.hi, s.lo);
}
static inline df operator-(df a) { return df{-a.hi, -a.lo}; }
static inline df operator-(df a, df b) { return a + (-b); }
static inline df operator*(df a, df b) {
  float p = a.hi*b.hi;
  float e = fma(a.hi, b.hi, -p);
  e = fma(a.hi, b.lo, e);
  e = fma(a.lo, b.hi, e);
  return QuickTwoSum(p, e);
}
static inline df operator/(df a, df b) {
  float q1 = a.hi/b.hi;
  df r = a - b*DF(q1);
  float q2 = r.hi/b.hi;
  r = r - b*DF(q2);
  float q3 = r.hi/b.hi;
  return QuickTwoSum(q1, q2) + DF(q3);
}
static inline df Sqrt(df a) {
  if (a.hi<=0.0f)
    return DF(0.0f);
  float x = sqrt(a.hi);
  df xx = DF(x)*DF(x);
  return DF(x) + DF(((a - xx).hi)*(0.5f/x));
}
static inline float Float(df a) { return a.hi + a.lo; }
static inline df ShuffleXor(df a, ushort d) { return df{simd_shuffle_xor(a.hi, d), simd_shuffle_xor(a.lo, d)}; }

// ImproveHomography (geomFuncs.cpp, in double precision on the CPU) on the GPU:
// numLoops iterations of least squares over the inliers, then the inlier count and
// match_error of all points. Each iteration: ImproveAccumulate sums the normal equations
// over the points in IMPROVE_GROUPS threadgroups, then ImproveSolve adds the partial sums
// (in a fixed order) and solves them (Cholesky). ImproveFinal and ImproveCount compute
// the inliers and match errors. The residuals are rounded to float as in the CPU code;
// the products -x*mx, -y*mx, -x*my, -y*my are not (the CPU code computes them in float).
// Only the non-zero terms of the equations are summed: a point's two rows have non-zero
// coefficients 0,1,2,6,7 and 3,4,5,6,7.

struct ImproveState {        // device state between the kernels of ImproveHomographyGPU
  df A[8];                   // current homography
  df partial[IMPROVE_GROUPS][IMPROVE_NV];
  int counts[IMPROVE_GROUPS];
};
static_assert(sizeof(ImproveState)==IMPROVE_STATE_SIZE, "IMPROVE_STATE_SIZE (cudaSiftMetalShared.h)");

static inline float ImproveError(thread const df *a, float x, float y, float mx, float my)
{
  float den = Float(a[6]*DF(x) + a[7]*DF(y) + DF(1.0f));
  float dx = Float((a[0]*DF(x) + a[1]*DF(y) + a[2])/DF(den) - DF(mx));
  float dy = Float((a[3]*DF(x) + a[4]*DF(y) + a[5])/DF(den) - DF(my));
  return dx*dx + dy*dy;
}

kernel void ImproveInit(device const HomographyResult *in [[buffer(0)]], device ImproveState *st [[buffer(1)]],
                        uint i [[thread_position_in_grid]])
{
  if (i<8)
    st->A[i] = DF(in->h[i]/in->h[8]);
}

kernel void ImproveAccumulate(device const SiftPoint *sift [[buffer(0)]], constant int &numPts [[buffer(1)]],
                              device ImproveState *st [[buffer(2)]], constant HomographyGPUParams &p [[buffer(3)]],
                              uint gid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]])
{
  threadgroup df red[IMPROVE_NT/32][IMPROVE_NV];
  threadgroup int counts[IMPROVE_NT/32];
  df a[8];
  for (int k=0;k<8;k++)
    a[k] = st->A[k];
  df acc[IMPROVE_NV];
  for (int k=0;k<IMPROVE_NV;k++)
    acc[k] = DF(0.0f);
  int numfit = 0;
  for (int i=gid*IMPROVE_NT + tid;i<numPts;i+=IMPROVE_GROUPS*IMPROVE_NT) {
    const float x = sift[i].xpos, y = sift[i].ypos, mx = sift[i].match_xpos, my = sift[i].match_ypos;
    if (sift[i].score<p.minScore || sift[i].ambiguity>p.maxAmbiguity)
      continue;
    if (!(ImproveError(a, x, y, mx, my)<p.thresh2))
      continue;
    numfit++;
    // Row for match_xpos: Y = (x, y, 1, 0, 0, 0, -x*mx, -y*mx), for match_ypos: (0, 0, 0, x, y, 1, -x*my, -y*my)
    const df X = DF(x), Yv = DF(y), One = DF(1.0f);
    const df u6 = -(X*DF(mx)), u7 = -(Yv*DF(mx)), v6 = -(X*DF(my)), v7 = -(Yv*DF(my));
    const df U[5] = {X, Yv, One, u6, u7}, V[5] = {X, Yv, One, v6, v7};
    int k = 0;
    for (int r=0;r<3;r++)                     // entries (r, c), r<=c, of U: 0..11 (without (6,6), (6,7), (7,7))
      for (int c=r;c<5;c++) {
        acc[k] = acc[k] + U[c]*U[r];
        k++;
      }
    for (int r=0;r<3;r++)                     // entries of V: 12..23
      for (int c=r;c<5;c++) {
        acc[k] = acc[k] + V[c]*V[r];
        k++;
      }
    acc[24] = acc[24] + u6*u6 + v6*v6;        // (6,6), (6,7), (7,7): both rows
    acc[25] = acc[25] + u7*u6 + v7*v6;
    acc[26] = acc[26] + u7*u7 + v7*v7;
    for (int r=0;r<3;r++) {                   // vector: Y*match
      acc[27 + r] = acc[27 + r] + U[r]*DF(mx);
      acc[30 + r] = acc[30 + r] + V[r]*DF(my);
    }
    acc[33] = acc[33] + u6*DF(mx) + v6*DF(my);
    acc[34] = acc[34] + u7*DF(mx) + v7*DF(my);
  }
  for (int k=0;k<IMPROVE_NV;k++) {
    df v = acc[k];
    for (ushort d=16;d>0;d/=2)
      v = v + ShuffleXor(v, d);
    if (lane==0)
      red[sg][k] = v;
  }
  numfit = simd_sum(numfit);
  if (lane==0)
    counts[sg] = numfit;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid<IMPROVE_NV) {
    df v = red[0][tid];
    for (int s=1;s<IMPROVE_NT/32;s++)
      v = v + red[s][tid];
    st->partial[gid][tid] = v;
  }
  if (tid==0) {
    int n = 0;
    for (int s=0;s<IMPROVE_NT/32;s++)
      n += counts[s];
    st->counts[gid] = n;   // inliers of this iteration (ImproveFinal overwrites them)
  }
}

kernel void ImproveSolve(device ImproveState *st [[buffer(0)]], uint tid [[thread_index_in_threadgroup]])
{
  // One SIMD group: sums of the partial results, then the normal equations M A = b in
  // threadgroup memory, and their Cholesky factorization with the column updates spread
  // over the lanes. With fewer than 4 inliers, or if M is not positive definite, the
  // homography is kept (cv::solve in ImproveHomography sets it to 0), and so are the
  // next iterations, which have the same inliers.
  int numfit = 0;
  for (int g=0;g<IMPROVE_GROUPS;g++)
    numfit += st->counts[g];
  if (numfit<4)
    return;
  threadgroup df L[8][8];
  threadgroup df b[8];
  threadgroup df sum[IMPROVE_NV];
  threadgroup bool ok;
  for (int k=tid;k<IMPROVE_NV;k+=32) {
    df v = st->partial[0][k];
    for (int g=1;g<IMPROVE_GROUPS;g++)
      v = v + st->partial[g][k];
    sum[k] = v;
  }
  for (int k=tid;k<64;k+=32)
    L[k/8][k%8] = DF(0.0f);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (tid==0) {
    const int map[5] = {0, 1, 2, 6, 7}, map2[5] = {3, 4, 5, 6, 7};
    int k = 0;
    for (int r=0;r<3;r++)
      for (int c=r;c<5;c++, k++)
        L[map[r]][map[c]] = L[map[c]][map[r]] = sum[k];
    for (int r=0;r<3;r++)
      for (int c=r;c<5;c++, k++)
        L[map2[r]][map2[c]] = L[map2[c]][map2[r]] = sum[k];
    L[6][6] = sum[24];
    L[6][7] = L[7][6] = sum[25];
    L[7][7] = sum[26];
    for (int r=0;r<3;r++) {
      b[map[r]] = sum[27 + r];
      b[map2[r]] = sum[30 + r];
    }
    b[6] = sum[33];
    b[7] = sum[34];
    ok = true;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (int j=0;j<8;j++) {
    if (tid==0) {
      df s = L[j][j];
      for (int k=0;k<j;k++)
        s = s - L[j][k]*L[j][k];
      if (!(s.hi>0.0f))
        ok = false;
      L[j][j] = Sqrt(s);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (!ok)
      break;
    const int i = j + 1 + tid;
    if (i<8) {
      df t = L[i][j];
      for (int k=0;k<j;k++)
        t = t - L[i][k]*L[j][k];
      L[i][j] = t/L[j][j];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid!=0 || !ok)
    return;
  df A[8];
  {
    df z[8];
    for (int i=0;i<8;i++) {
      df t = b[i];
      for (int k=0;k<i;k++)
        t = t - L[i][k]*z[k];
      z[i] = t/L[i][i];
    }
    for (int i=7;i>=0;i--) {
      df t = z[i];
      for (int k=i+1;k<8;k++)
        t = t - L[k][i]*A[k];
      A[i] = t/L[i][i];
    }
  }
  for (int i=0;i<8;i++)
    st->A[i] = A[i];
}

kernel void ImproveFinal(device SiftPoint *sift [[buffer(0)]], constant int &numPts [[buffer(1)]],
                         device ImproveState *st [[buffer(2)]], constant HomographyGPUParams &p [[buffer(3)]],
                         uint gid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]])
{
  threadgroup int counts[IMPROVE_NT/32];
  df a[8];
  for (int k=0;k<8;k++)
    a[k] = st->A[k];
  int numfit = 0;
  for (int i=gid*IMPROVE_NT + tid;i<numPts;i+=IMPROVE_GROUPS*IMPROVE_NT) {
    float err = ImproveError(a, sift[i].xpos, sift[i].ypos, sift[i].match_xpos, sift[i].match_ypos);
    if (err<p.thresh2)
      numfit++;
    sift[i].match_error = sqrt(err);
  }
  numfit = simd_sum(numfit);
  if (lane==0)
    counts[sg] = numfit;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid==0) {
    int n = 0;
    for (int s=0;s<IMPROVE_NT/32;s++)
      n += counts[s];
    st->counts[gid] = n;
  }
}

kernel void ImproveCount(device const ImproveState *st [[buffer(0)]], device HomographyResult *res [[buffer(1)]],
                         uint i [[thread_position_in_grid]])
{
  if (i!=0)
    return;
  int n = 0;
  for (int g=0;g<IMPROVE_GROUPS;g++)
    n += st->counts[g];
  for (int k=0;k<8;k++)
    res->h[k] = Float(st->A[k]);
  res->h[8] = 1.0f;
  res->count = n;
}
